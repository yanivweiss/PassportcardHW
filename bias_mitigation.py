"""
Bias mitigation module for insurance claim predictions

This module provides techniques for detecting and mitigating unfair bias
in machine learning models for regression problems.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.optimizers import Adam

def calculate_group_statistics(X, y, group_col):
    """
    Calculate statistics for different groups in the dataset.
    
    Parameters:
    -----------
    X : pandas DataFrame
        Feature matrix
    y : array-like
        Target values
    group_col : str
        Column name for group membership
        
    Returns:
    --------
    pandas DataFrame
        DataFrame with group statistics
    """
    if group_col not in X.columns:
        raise ValueError(f"Group column '{group_col}' not found in the data")
    
    # Create a DataFrame with target and group information
    data = pd.DataFrame({
        'target': y,
        'group': X[group_col]
    })
    
    # Calculate statistics by group
    stats = data.groupby('group').agg({
        'target': ['count', 'mean', 'std', 'min', 'max']
    }).reset_index()
    
    # Flatten column names
    stats.columns = ['group' if col == 'group' else f'target_{agg}' for col, agg in stats.columns]
    
    # Calculate overall statistics
    overall = pd.DataFrame({
        'group': ['overall'],
        'target_count': [len(y)],
        'target_mean': [y.mean()],
        'target_std': [y.std()],
        'target_min': [y.min()],
        'target_max': [y.max()]
    })
    
    # Calculate proportion of samples in each group (excluding 'overall')
    stats['proportion'] = stats['target_count'] / len(y)
    
    # Calculate relative mean (compared to overall)
    overall_mean = y.mean()
    stats['relative_mean'] = stats['target_mean'] / overall_mean
    
    # Combine group and overall statistics
    combined_stats = pd.concat([stats, overall], ignore_index=True)
    
    # Set proportion for 'overall' to 1.0
    combined_stats.loc[combined_stats['group'] == 'overall', 'proportion'] = 1.0
    combined_stats.loc[combined_stats['group'] == 'overall', 'relative_mean'] = 1.0
    
    return combined_stats

def create_sample_weights(X, y, group_col, method='balanced'):
    """
    Create sample weights to balance the training distribution.
    
    Parameters:
    -----------
    X : pandas DataFrame
        Feature matrix
    y : array-like
        Target values
    group_col : str
        Column name for group membership
    method : str, optional
        Method for calculating weights:
        'balanced' - balance group representation
        'balanced_target' - balance both group and target representation
        
    Returns:
    --------
    numpy array
        Sample weights for training
    """
    if group_col not in X.columns:
        raise ValueError(f"Group column '{group_col}' not found in the data")
    
    # Get group information
    groups = X[group_col].values
    unique_groups = np.unique(groups)
    n_samples = len(y)
    
    if method == 'balanced':
        # Calculate weights to balance group representation
        group_counts = pd.Series(groups).value_counts()
        weights = np.ones(n_samples)
        
        for group in unique_groups:
            # Weight is inversely proportional to group frequency
            group_weight = n_samples / (len(unique_groups) * group_counts[group])
            weights[groups == group] = group_weight
            
    elif method == 'balanced_target':
        # Create bins for the continuous target variable
        n_bins = 5
        y_binned = pd.qcut(y, n_bins, labels=False, duplicates='drop')
        
        # Calculate weights for group and target bin combinations
        data = pd.DataFrame({
            'group': groups,
            'target_bin': y_binned
        })
        
        # Count samples in each group-bin combination
        group_target_counts = data.groupby(['group', 'target_bin']).size().reset_index(name='count')
        
        # Calculate expected count in a perfectly balanced dataset
        expected_count = n_samples / (len(unique_groups) * n_bins)
        
        # Create weights dictionary
        weight_dict = {}
        for _, row in group_target_counts.iterrows():
            weight_dict[(row['group'], row['target_bin'])] = expected_count / row['count']
        
        # Apply weights to samples
        weights = np.ones(n_samples)
        for i in range(n_samples):
            weights[i] = weight_dict.get((groups[i], y_binned[i]), 1.0)
    
    # Normalize weights to have mean=1
    weights = weights / weights.mean()
    
    return weights

def train_weighted_model(X_train, y_train, X_test, y_test, group_col, 
                         weighting_method='balanced', model_type='xgboost'):
    """
    Train a model with sample weights to mitigate bias.
    
    Parameters:
    -----------
    X_train : pandas DataFrame
        Training feature matrix
    y_train : array-like
        Training target values
    X_test : pandas DataFrame
        Testing feature matrix
    y_test : array-like
        Testing target values
    group_col : str
        Column name for group membership
    weighting_method : str, optional
        Method for calculating weights
    model_type : str, optional
        Type of model to train
        
    Returns:
    --------
    tuple
        Trained model and performance metrics by group
    """
    # Create output directory for visualizations
    os.makedirs('visualizations/bias_mitigation', exist_ok=True)
    
    # Calculate sample weights
    sample_weights = create_sample_weights(X_train, y_train, group_col, method=weighting_method)
    
    # Get group information for later evaluation
    group_train = X_train[group_col].values
    group_test = X_test[group_col].values
    
    # Create a copy of training data without the group column
    X_train_model = X_train.drop(columns=[group_col]).copy()
    X_test_model = X_test.drop(columns=[group_col]).copy()
    
    # Train a model with sample weights
    if model_type == 'xgboost':
        model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            n_jobs=-1
        )
        
        # Train with sample weights
        model.fit(
            X_train_model, y_train,
            sample_weight=sample_weights,
            verbose=False
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Make predictions on test data
    y_pred = model.predict(X_test_model)
    
    # Calculate overall performance metrics
    metrics = {
        'overall': {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
    }
    
    # Calculate performance metrics by group
    for group in np.unique(group_test):
        group_mask = (group_test == group)
        if sum(group_mask) > 0:  # Ensure we have samples for this group
            metrics[group] = {
                'rmse': np.sqrt(mean_squared_error(y_test[group_mask], y_pred[group_mask])),
                'mae': mean_absolute_error(y_test[group_mask], y_pred[group_mask]),
                'r2': r2_score(y_test[group_mask], y_pred[group_mask])
            }
    
    # Create a visualization of performance by group
    plt.figure(figsize=(12, 6))
    
    # Plot RMSE by group
    plt.subplot(1, 2, 1)
    groups = list(metrics.keys())
    rmse_values = [metrics[g]['rmse'] for g in groups]
    
    # Plot bar chart
    sns.barplot(x=groups, y=rmse_values)
    plt.title('RMSE by Group')
    plt.ylabel('RMSE')
    plt.axhline(y=metrics['overall']['rmse'], linestyle='--', color='r', label='Overall')
    plt.legend()
    
    # Plot % difference from overall
    plt.subplot(1, 2, 2)
    overall_rmse = metrics['overall']['rmse']
    pct_diff = [(metrics[g]['rmse'] - overall_rmse) / overall_rmse * 100 for g in groups]
    
    sns.barplot(x=groups, y=pct_diff)
    plt.title('% Difference from Overall RMSE')
    plt.ylabel('% Difference')
    plt.axhline(y=0, linestyle='--', color='r')
    
    plt.tight_layout()
    plt.savefig(f'visualizations/bias_mitigation/weighted_model_{weighting_method}_performance.png')
    plt.close()
    
    return model, metrics

def adversarial_debiasing(X_train, y_train, X_test, y_test, group_col, 
                          adversary_loss_weight=1.0, batch_size=32, epochs=10):
    """
    Implement adversarial debiasing to mitigate unfair bias.
    
    This technique trains a main model to predict the target while an adversary
    model attempts to predict the sensitive attribute from the main model's representations.
    
    Parameters:
    -----------
    X_train : pandas DataFrame
        Training feature matrix
    y_train : array-like
        Training target values
    X_test : pandas DataFrame
        Testing feature matrix
    y_test : array-like
        Testing target values
    group_col : str
        Column name for group membership
    adversary_loss_weight : float, optional
        Weight for the adversary loss
    batch_size : int, optional
        Batch size for training
    epochs : int, optional
        Number of training epochs
        
    Returns:
    --------
    tuple
        Trained model and performance metrics by group
    """
    # Create output directory for visualizations
    os.makedirs('visualizations/bias_mitigation', exist_ok=True)
    
    # Extract group information
    group_train = X_train[group_col].values
    group_test = X_test[group_col].values
    
    # Create a copy of training data without the group column
    X_train_model = X_train.drop(columns=[group_col]).copy()
    X_test_model = X_test.drop(columns=[group_col]).copy()
    
    # Convert group labels to numeric if they are categorical
    if not np.issubdtype(group_train.dtype, np.number):
        group_encoder = {group: idx for idx, group in enumerate(np.unique(group_train))}
        group_train_numeric = np.array([group_encoder[g] for g in group_train])
        group_test_numeric = np.array([group_encoder[g] for g in group_test])
        n_groups = len(group_encoder)
    else:
        group_train_numeric = group_train
        group_test_numeric = group_test
        n_groups = len(np.unique(group_train))
    
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_model)
    X_test_scaled = scaler.transform(X_test_model)
    
    # Build the adversarial model using TensorFlow
    n_features = X_train_scaled.shape[1]
    
    # Define the predictor model
    input_features = Input(shape=(n_features,))
    predictor_hidden = Dense(32, activation='relu')(input_features)
    predictor_hidden = Dense(16, activation='relu')(predictor_hidden)
    predictor_output = Dense(1, activation='linear', name='predictor_output')(predictor_hidden)
    
    # Define the adversary model - takes predictor's hidden representation
    adversary_hidden = Dense(16, activation='relu')(predictor_hidden)
    
    # Output layer depends on whether group is binary or multi-class
    if n_groups == 2:
        adversary_output = Dense(1, activation='sigmoid', name='adversary_output')(adversary_hidden)
    else:
        adversary_output = Dense(n_groups, activation='softmax', name='adversary_output')(adversary_hidden)
    
    # Combined model
    combined_model = Model(inputs=input_features, outputs=[predictor_output, adversary_output])
    
    # Predictor model (for inference)
    predictor_model = Model(inputs=input_features, outputs=predictor_output)
    
    # Compile the models
    if n_groups == 2:
        adversary_loss = 'binary_crossentropy'
    else:
        adversary_loss = 'sparse_categorical_crossentropy'
    
    combined_model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss={
            'predictor_output': 'mse',
            'adversary_output': adversary_loss
        },
        loss_weights={
            'predictor_output': 1.0,
            'adversary_output': -adversary_loss_weight  # Negative to maximize this loss
        }
    )
    
    # Train the combined model
    history = combined_model.fit(
        X_train_scaled,
        {
            'predictor_output': y_train,
            'adversary_output': group_train_numeric
        },
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        validation_split=0.2
    )
    
    # Make predictions using the predictor model
    y_pred = predictor_model.predict(X_test_scaled).flatten()
    
    # Calculate overall performance metrics
    metrics = {
        'overall': {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
    }
    
    # Calculate performance metrics by group
    for group in np.unique(group_test):
        group_mask = (group_test == group)
        if sum(group_mask) > 0:  # Ensure we have samples for this group
            metrics[group] = {
                'rmse': np.sqrt(mean_squared_error(y_test[group_mask], y_pred[group_mask])),
                'mae': mean_absolute_error(y_test[group_mask], y_pred[group_mask]),
                'r2': r2_score(y_test[group_mask], y_pred[group_mask])
            }
    
    # Visualize training history
    plt.figure(figsize=(12, 4))
    
    # Plot predictor loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['predictor_output_loss'], label='train')
    plt.plot(history.history['val_predictor_output_loss'], label='validation')
    plt.title('Predictor Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    
    # Plot adversary loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['adversary_output_loss'], label='train')
    plt.plot(history.history['val_adversary_output_loss'], label='validation')
    plt.title('Adversary Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'visualizations/bias_mitigation/adversarial_training_history.png')
    plt.close()
    
    # Create a visualization of performance by group
    plt.figure(figsize=(12, 6))
    
    # Plot RMSE by group
    plt.subplot(1, 2, 1)
    groups = list(metrics.keys())
    rmse_values = [metrics[g]['rmse'] for g in groups]
    
    # Plot bar chart
    sns.barplot(x=groups, y=rmse_values)
    plt.title('RMSE by Group')
    plt.ylabel('RMSE')
    plt.axhline(y=metrics['overall']['rmse'], linestyle='--', color='r', label='Overall')
    plt.legend()
    
    # Plot % difference from overall
    plt.subplot(1, 2, 2)
    overall_rmse = metrics['overall']['rmse']
    pct_diff = [(metrics[g]['rmse'] - overall_rmse) / overall_rmse * 100 for g in groups]
    
    sns.barplot(x=groups, y=pct_diff)
    plt.title('% Difference from Overall RMSE')
    plt.ylabel('% Difference')
    plt.axhline(y=0, linestyle='--', color='r')
    
    plt.tight_layout()
    plt.savefig(f'visualizations/bias_mitigation/adversarial_debiasing_performance.png')
    plt.close()
    
    return predictor_model, metrics

def post_processing_calibration(model, X_test, y_test, group_col):
    """
    Apply post-processing calibration to equalize error rates across groups.
    
    Parameters:
    -----------
    model : estimator
        Trained model with predict method
    X_test : pandas DataFrame
        Test feature matrix
    y_test : array-like
        Test target values
    group_col : str
        Column name for group membership
        
    Returns:
    --------
    dict
        Calibration parameters for each group
    """
    # Create output directory for visualizations
    os.makedirs('visualizations/bias_mitigation', exist_ok=True)
    
    # Get group information
    group_test = X_test[group_col].values
    unique_groups = np.unique(group_test)
    
    # Create a copy of test data without the group column for prediction
    X_test_model = X_test.drop(columns=[group_col]).copy()
    
    # Get uncalibrated predictions
    y_pred = model.predict(X_test_model)
    
    # Calculate error statistics for each group
    error_stats = {}
    for group in unique_groups:
        group_mask = (group_test == group)
        if sum(group_mask) > 0:
            y_test_group = y_test[group_mask]
            y_pred_group = y_pred[group_mask]
            
            # Calculate mean error
            error_mean = np.mean(y_pred_group - y_test_group)
            
            # Calculate error standard deviation
            error_std = np.std(y_pred_group - y_test_group)
            
            error_stats[group] = {
                'mean': error_mean,
                'std': error_std,
                'samples': sum(group_mask)
            }
    
    # Calculate overall error statistics
    overall_error_mean = np.mean(y_pred - y_test)
    overall_error_std = np.std(y_pred - y_test)
    
    # Calculate calibration parameters for each group
    calibration_params = {}
    for group, stats in error_stats.items():
        # We'll calibrate by subtracting the group-specific mean error
        # to bring each group's mean error close to the overall mean error
        calibration = stats['mean'] - overall_error_mean
        calibration_params[group] = calibration
    
    # Apply calibration and calculate metrics
    calibrated_metrics = {}
    uncalibrated_metrics = {}
    
    for group in unique_groups:
        group_mask = (group_test == group)
        if sum(group_mask) > 0:
            y_test_group = y_test[group_mask]
            y_pred_group = y_pred[group_mask]
            
            # Calculate uncalibrated metrics
            uncalibrated_metrics[group] = {
                'rmse': np.sqrt(mean_squared_error(y_test_group, y_pred_group)),
                'mae': mean_absolute_error(y_test_group, y_pred_group),
                'bias': np.mean(y_pred_group - y_test_group)
            }
            
            # Apply calibration
            y_pred_calibrated = y_pred_group - calibration_params[group]
            
            # Calculate calibrated metrics
            calibrated_metrics[group] = {
                'rmse': np.sqrt(mean_squared_error(y_test_group, y_pred_calibrated)),
                'mae': mean_absolute_error(y_test_group, y_pred_calibrated),
                'bias': np.mean(y_pred_calibrated - y_test_group)
            }
    
    # Overall metrics
    uncalibrated_metrics['overall'] = {
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'bias': np.mean(y_pred - y_test)
    }
    
    # Apply calibration to get overall metrics
    y_pred_calibrated = np.zeros_like(y_pred)
    for group in unique_groups:
        group_mask = (group_test == group)
        y_pred_calibrated[group_mask] = y_pred[group_mask] - calibration_params[group]
    
    calibrated_metrics['overall'] = {
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_calibrated)),
        'mae': mean_absolute_error(y_test, y_pred_calibrated),
        'bias': np.mean(y_pred_calibrated - y_test)
    }
    
    # Visualize calibration results
    plt.figure(figsize=(15, 5))
    
    # Plot bias (before and after)
    plt.subplot(1, 3, 1)
    groups = list(uncalibrated_metrics.keys())
    uncalibrated_bias = [uncalibrated_metrics[g]['bias'] for g in groups]
    calibrated_bias = [calibrated_metrics[g]['bias'] for g in groups]
    
    x = np.arange(len(groups))
    width = 0.35
    
    plt.bar(x - width/2, uncalibrated_bias, width, label='Uncalibrated')
    plt.bar(x + width/2, calibrated_bias, width, label='Calibrated')
    
    plt.axhline(y=0, linestyle='--', color='r')
    plt.xlabel('Group')
    plt.ylabel('Bias (Mean Error)')
    plt.title('Bias Before and After Calibration')
    plt.xticks(x, groups)
    plt.legend()
    
    # Plot RMSE (before and after)
    plt.subplot(1, 3, 2)
    uncalibrated_rmse = [uncalibrated_metrics[g]['rmse'] for g in groups]
    calibrated_rmse = [calibrated_metrics[g]['rmse'] for g in groups]
    
    plt.bar(x - width/2, uncalibrated_rmse, width, label='Uncalibrated')
    plt.bar(x + width/2, calibrated_rmse, width, label='Calibrated')
    
    plt.xlabel('Group')
    plt.ylabel('RMSE')
    plt.title('RMSE Before and After Calibration')
    plt.xticks(x, groups)
    plt.legend()
    
    # Plot MAE (before and after)
    plt.subplot(1, 3, 3)
    uncalibrated_mae = [uncalibrated_metrics[g]['mae'] for g in groups]
    calibrated_mae = [calibrated_metrics[g]['mae'] for g in groups]
    
    plt.bar(x - width/2, uncalibrated_mae, width, label='Uncalibrated')
    plt.bar(x + width/2, calibrated_mae, width, label='Calibrated')
    
    plt.xlabel('Group')
    plt.ylabel('MAE')
    plt.title('MAE Before and After Calibration')
    plt.xticks(x, groups)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('visualizations/bias_mitigation/post_processing_calibration.png')
    plt.close()
    
    return {
        'calibration_params': calibration_params,
        'uncalibrated_metrics': uncalibrated_metrics,
        'calibrated_metrics': calibrated_metrics
    }

def fairness_constrained_optimization(X_train, y_train, X_test, y_test, group_col, fairness_constraint=0.1):
    """
    Train a model with fairness constraints to ensure similar performance across groups.
    
    For regression, we implement a constraint that the difference in prediction error 
    between any two groups should not exceed a specified threshold.
    
    Parameters:
    -----------
    X_train : pandas DataFrame
        Training feature matrix
    y_train : array-like
        Training target values
    X_test : pandas DataFrame
        Testing feature matrix
    y_test : array-like
        Testing target values
    group_col : str
        Column name for group membership
    fairness_constraint : float, optional
        Maximum allowed difference in error between groups
        
    Returns:
    --------
    tuple
        Trained model and performance metrics by group
    """
    # Create a simple implementation using iterative reweighting
    # In each iteration, adjust sample weights based on error disparities
    
    # Create output directory for visualizations
    os.makedirs('visualizations/bias_mitigation', exist_ok=True)
    
    # Extract group information
    group_train = X_train[group_col].values
    group_test = X_test[group_col].values
    unique_groups = np.unique(group_train)
    
    # Create a copy of training data without the group column
    X_train_model = X_train.drop(columns=[group_col]).copy()
    X_test_model = X_test.drop(columns=[group_col]).copy()
    
    # Initialize sample weights
    sample_weights = np.ones(len(y_train))
    
    # Track error disparities across iterations
    error_disparities = []
    max_iterations = 10  # Maximum number of iterations
    
    for iteration in range(max_iterations):
        # Train model with current weights
        model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train_model, y_train, sample_weight=sample_weights, verbose=False)
        
        # Get predictions
        y_train_pred = model.predict(X_train_model)
        
        # Calculate errors for each group
        group_errors = {}
        for group in unique_groups:
            mask = (group_train == group)
            if sum(mask) > 0:
                errors = np.abs(y_train_pred[mask] - y_train[mask])
                group_errors[group] = errors.mean()
        
        # Calculate error disparity
        max_error = max(group_errors.values())
        min_error = min(group_errors.values())
        error_disparity = max_error - min_error
        error_disparities.append(error_disparity)
        
        # Check if fairness constraint is satisfied
        if error_disparity <= fairness_constraint:
            print(f"Fairness constraint satisfied at iteration {iteration+1}")
            break
        
        # Adjust sample weights based on group errors
        for group in unique_groups:
            group_mask = (group_train == group)
            
            # If group error is high, increase weights
            relative_error = group_errors[group] / max_error
            adjustment = 1 + (relative_error - 1) * 0.5  # Moderate adjustment
            
            sample_weights[group_mask] *= adjustment
        
        # Normalize weights
        sample_weights = sample_weights / sample_weights.mean() * 1.0
    
    # Evaluate on test data
    y_pred = model.predict(X_test_model)
    
    # Calculate overall performance metrics
    metrics = {
        'overall': {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
    }
    
    # Calculate performance metrics by group
    group_metrics = {}
    for group in np.unique(group_test):
        group_mask = (group_test == group)
        if sum(group_mask) > 0:
            group_metrics[group] = {
                'rmse': np.sqrt(mean_squared_error(y_test[group_mask], y_pred[group_mask])),
                'mae': mean_absolute_error(y_test[group_mask], y_pred[group_mask]),
                'r2': r2_score(y_test[group_mask], y_pred[group_mask])
            }
            metrics[group] = group_metrics[group]
    
    # Visualize error disparity across iterations
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(error_disparities) + 1), error_disparities, 'o-')
    plt.axhline(y=fairness_constraint, linestyle='--', color='r', 
                label=f'Constraint: {fairness_constraint}')
    plt.xlabel('Iteration')
    plt.ylabel('Error Disparity')
    plt.title('Error Disparity Across Iterations')
    plt.legend()
    plt.tight_layout()
    plt.savefig('visualizations/bias_mitigation/fairness_constrained_convergence.png')
    plt.close()
    
    # Visualize final group performance
    plt.figure(figsize=(12, 6))
    
    # Plot RMSE by group
    plt.subplot(1, 2, 1)
    groups = list(group_metrics.keys())
    rmse_values = [group_metrics[g]['rmse'] for g in groups]
    
    # Plot bar chart
    sns.barplot(x=groups, y=rmse_values)
    plt.title('RMSE by Group')
    plt.ylabel('RMSE')
    plt.axhline(y=metrics['overall']['rmse'], linestyle='--', color='r', label='Overall')
    plt.legend()
    
    # Plot % difference from overall
    plt.subplot(1, 2, 2)
    overall_rmse = metrics['overall']['rmse']
    pct_diff = [(group_metrics[g]['rmse'] - overall_rmse) / overall_rmse * 100 for g in groups]
    
    sns.barplot(x=groups, y=pct_diff)
    plt.title('% Difference from Overall RMSE')
    plt.ylabel('% Difference')
    plt.axhline(y=0, linestyle='--', color='r')
    
    plt.tight_layout()
    plt.savefig('visualizations/bias_mitigation/fairness_constrained_performance.png')
    plt.close()
    
    return model, metrics

def evaluate_bias_mitigation_methods(X, y, group_col, test_size=0.2, random_state=42):
    """
    Evaluate different bias mitigation methods on the same dataset.
    
    Parameters:
    -----------
    X : pandas DataFrame
        Feature matrix
    y : array-like
        Target values
    group_col : str
        Column name for group membership
    test_size : float, optional
        Proportion of data to use for testing
    random_state : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    dict
        Dictionary with results of each method
    """
    # Create output directory for visualizations
    os.makedirs('visualizations/bias_mitigation', exist_ok=True)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Ensure group_col is in X
    if group_col not in X.columns:
        raise ValueError(f"Group column '{group_col}' not found in the data")
    
    # Train baseline model (no bias mitigation)
    print("Training baseline model...")
    baseline_model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=random_state,
        n_jobs=-1
    )
    
    X_train_model = X_train.drop(columns=[group_col]).copy()
    X_test_model = X_test.drop(columns=[group_col]).copy()
    
    baseline_model.fit(X_train_model, y_train, verbose=False)
    
    # Get predictions
    baseline_pred = baseline_model.predict(X_test_model)
    
    # Calculate overall performance metrics
    baseline_metrics = {
        'overall': {
            'rmse': np.sqrt(mean_squared_error(y_test, baseline_pred)),
            'mae': mean_absolute_error(y_test, baseline_pred),
            'r2': r2_score(y_test, baseline_pred)
        }
    }
    
    # Calculate performance by group
    for group in np.unique(X_test[group_col]):
        group_mask = (X_test[group_col] == group)
        if sum(group_mask) > 0:
            baseline_metrics[group] = {
                'rmse': np.sqrt(mean_squared_error(y_test[group_mask], baseline_pred[group_mask])),
                'mae': mean_absolute_error(y_test[group_mask], baseline_pred[group_mask]),
                'r2': r2_score(y_test[group_mask], baseline_pred[group_mask])
            }
    
    # Train weighted model
    print("Training weighted model...")
    weighted_model, weighted_metrics = train_weighted_model(
        X_train, y_train, X_test, y_test, group_col, weighting_method='balanced'
    )
    
    # Train adversarial debiasing model
    print("Training adversarial debiasing model...")
    adversarial_model, adversarial_metrics = adversarial_debiasing(
        X_train, y_train, X_test, y_test, group_col, epochs=5  # Reduced epochs for demo
    )
    
    # Apply post-processing calibration to baseline model
    print("Applying post-processing calibration...")
    calibration_params = post_processing_calibration(baseline_model, X_test, y_test, group_col)
    
    # Train fairness constrained model
    print("Training fairness constrained model...")
    fairness_model, fairness_metrics = fairness_constrained_optimization(
        X_train, y_train, X_test, y_test, group_col
    )
    
    # Collect all results
    results = {
        'baseline': {
            'model': baseline_model,
            'metrics': baseline_metrics
        },
        'weighted': {
            'model': weighted_model,
            'metrics': weighted_metrics
        },
        'adversarial': {
            'model': adversarial_model,
            'metrics': adversarial_metrics
        },
        'fairness_constrained': {
            'model': fairness_model,
            'metrics': fairness_metrics
        }
    }
    
    # Compare all methods
    methods = list(results.keys())
    groups = list(baseline_metrics.keys())
    
    # Prepare data for visualization
    comparison_data = []
    
    for method in methods:
        for group in groups:
            if group in results[method]['metrics']:
                comparison_data.append({
                    'method': method,
                    'group': group,
                    'rmse': results[method]['metrics'][group]['rmse'],
                    'mae': results[method]['metrics'][group]['mae'],
                    'r2': results[method]['metrics'][group]['r2']
                })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Create visualization comparing all methods
    plt.figure(figsize=(15, 10))
    
    # Plot RMSE by method and group
    plt.subplot(2, 1, 1)
    sns.barplot(x='group', y='rmse', hue='method', data=comparison_df)
    plt.title('RMSE by Method and Group')
    plt.ylabel('RMSE')
    plt.legend(title='Method')
    
    # Plot R² by method and group
    plt.subplot(2, 1, 2)
    sns.barplot(x='group', y='r2', hue='method', data=comparison_df)
    plt.title('R² by Method and Group')
    plt.ylabel('R²')
    plt.legend(title='Method')
    
    plt.tight_layout()
    plt.savefig('visualizations/bias_mitigation/methods_comparison.png')
    plt.close()
    
    # Calculate fairness metrics for each method
    fairness_scores = {}
    
    for method in methods:
        # Calculate max disparity in error across groups
        method_metrics = results[method]['metrics']
        group_rmse = [method_metrics[g]['rmse'] for g in groups if g != 'overall']
        
        if len(group_rmse) > 1:
            fairness_scores[method] = {
                'max_disparity': max(group_rmse) - min(group_rmse),
                'disparity_ratio': max(group_rmse) / min(group_rmse) if min(group_rmse) > 0 else float('inf'),
                'std_deviation': np.std(group_rmse)
            }
    
    # Visualize fairness scores
    fairness_df = pd.DataFrame({
        'method': list(fairness_scores.keys()),
        'max_disparity': [fairness_scores[m]['max_disparity'] for m in fairness_scores],
        'disparity_ratio': [fairness_scores[m]['disparity_ratio'] for m in fairness_scores],
        'std_deviation': [fairness_scores[m]['std_deviation'] for m in fairness_scores]
    })
    
    plt.figure(figsize=(15, 5))
    
    # Plot max disparity
    plt.subplot(1, 3, 1)
    sns.barplot(x='method', y='max_disparity', data=fairness_df)
    plt.title('Maximum RMSE Disparity')
    plt.ylabel('Max RMSE Difference')
    plt.xticks(rotation=45)
    
    # Plot disparity ratio
    plt.subplot(1, 3, 2)
    sns.barplot(x='method', y='disparity_ratio', data=fairness_df)
    plt.title('RMSE Disparity Ratio')
    plt.ylabel('Max/Min RMSE Ratio')
    plt.xticks(rotation=45)
    
    # Plot standard deviation
    plt.subplot(1, 3, 3)
    sns.barplot(x='method', y='std_deviation', data=fairness_df)
    plt.title('RMSE Standard Deviation')
    plt.ylabel('RMSE Std Dev')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('visualizations/bias_mitigation/fairness_comparison.png')
    plt.close()
    
    return {
        'results': results,
        'comparison': comparison_df,
        'fairness_scores': fairness_scores
    }

def generate_bias_mitigation_report(evaluation_results, output_path='reports/bias_mitigation_report.md'):
    """
    Generate a comprehensive report on bias mitigation.
    
    Parameters:
    -----------
    evaluation_results : dict
        Results from evaluate_bias_mitigation_methods
    output_path : str, optional
        Path to save the report
        
    Returns:
    --------
    str
        Path to the generated report
    """
    # Create reports directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Extract results
    results = evaluation_results['results']
    comparison = evaluation_results['comparison']
    fairness_scores = evaluation_results['fairness_scores']
    
    # Start building the report
    report = [
        "# Bias Mitigation Evaluation Report\n",
        "## Overview\n",
        "This report compares different bias mitigation techniques for reducing unfair disparities in model performance across different groups.\n",
        
        "## Methods Evaluated\n",
        "1. **Baseline**: Standard model without bias mitigation\n",
        "2. **Sample Weighting**: Model trained with balanced sample weights to give equal importance to all groups\n",
        "3. **Adversarial Debiasing**: Model trained with an adversary that attempts to predict the sensitive attribute\n",
        "4. **Fairness Constraints**: Model trained with explicit constraints on performance disparities\n"
    ]
    
    # Add overall performance comparison
    report.append("## Overall Performance Comparison\n")
    
    # Create performance table
    report.append("| Method | Overall RMSE | Overall MAE | Overall R² |")
    report.append("| ------ | ------------ | ----------- | ---------- |")
    
    for method in results:
        metrics = results[method]['metrics']['overall']
        report.append(f"| {method.capitalize()} | {metrics['rmse']:.4f} | {metrics['mae']:.4f} | {metrics['r2']:.4f} |")
    
    # Add fairness metrics comparison
    report.append("\n## Fairness Metrics Comparison\n")
    report.append("The following metrics indicate how equitably each model performs across different groups:\n")
    
    # Create fairness table
    report.append("| Method | Max RMSE Disparity | Disparity Ratio | Std Deviation |")
    report.append("| ------ | ------------------ | --------------- | ------------- |")
    
    for method in fairness_scores:
        scores = fairness_scores[method]
        report.append(f"| {method.capitalize()} | {scores['max_disparity']:.4f} | {scores['disparity_ratio']:.4f} | {scores['std_deviation']:.4f} |")
    
    # Add performance by group
    report.append("\n## Performance by Group\n")
    
    # Get unique groups (excluding 'overall')
    groups = sorted(set(comparison['group']) - {'overall'})
    
    for group in groups:
        report.append(f"\n### Group: {group}\n")
        
        group_data = comparison[comparison['group'] == group]
        
        # Create group performance table
        report.append("| Method | RMSE | MAE | R² |")
        report.append("| ------ | ---- | --- | -- |")
        
        for _, row in group_data.iterrows():
            report.append(f"| {row['method'].capitalize()} | {row['rmse']:.4f} | {row['mae']:.4f} | {row['r2']:.4f} |")
    
    # Add visualizations
    report.append("\n## Visualizations\n")
    
    report.append("### Performance Comparison\n")
    report.append("![Methods Comparison](../visualizations/bias_mitigation/methods_comparison.png)\n")
    
    report.append("### Fairness Metrics\n")
    report.append("![Fairness Comparison](../visualizations/bias_mitigation/fairness_comparison.png)\n")
    
    # Add methods details
    report.append("\n## Method Details\n")
    
    report.append("### Sample Weighting\n")
    report.append("![Weighted Model Performance](../visualizations/bias_mitigation/weighted_model_balanced_performance.png)\n")
    report.append("This method assigns higher weights to underrepresented groups during training to ensure they have equal influence on the model.\n")
    
    report.append("### Adversarial Debiasing\n")
    report.append("![Adversarial Training History](../visualizations/bias_mitigation/adversarial_training_history.png)\n")
    report.append("![Adversarial Performance](../visualizations/bias_mitigation/adversarial_debiasing_performance.png)\n")
    report.append("This method uses an adversarial neural network to prevent the model from learning to discriminate based on sensitive attributes.\n")
    
    report.append("### Fairness Constraints\n")
    report.append("![Fairness Convergence](../visualizations/bias_mitigation/fairness_constrained_convergence.png)\n")
    report.append("![Fairness Performance](../visualizations/bias_mitigation/fairness_constrained_performance.png)\n")
    report.append("This method explicitly constrains the model to maintain similar error rates across different groups.\n")
    
    report.append("### Post-Processing Calibration\n")
    report.append("![Post-Processing Calibration](../visualizations/bias_mitigation/post_processing_calibration.png)\n")
    report.append("This method adjusts predictions after training to equalize error rates across groups.\n")
    
    # Add recommendations
    report.append("\n## Recommendations\n")
    
    # Find the best method based on fairness metrics
    fairness_df = pd.DataFrame(fairness_scores).T
    best_method = fairness_df['max_disparity'].idxmin()
    
    report.append(f"Based on the evaluation, the **{best_method.capitalize()}** method provides the best balance between overall performance and fairness across groups.\n")
    
    # Add specific recommendations
    report.append("### Implementation Recommendations:\n")
    report.append("1. **Data Collection**: Collect more diverse and representative data from all groups.\n")
    report.append("2. **Feature Engineering**: Review features that may be proxies for sensitive attributes.\n")
    report.append("3. **Regular Auditing**: Continuously monitor model performance across different groups.\n")
    report.append("4. **Transparent Reporting**: Clearly report performance metrics broken down by group.\n")
    
    # Add conclusion
    report.append("\n## Conclusion\n")
    report.append("Mitigating bias in machine learning models is crucial for ensuring fair and equitable outcomes. ")
    report.append("The techniques evaluated in this report provide different approaches to addressing bias, ")
    report.append("with trade-offs between overall performance and fairness across groups.\n")
    
    report.append("By implementing the recommended bias mitigation techniques, we can develop models ")
    report.append("that make more equitable predictions while maintaining good overall performance.")
    
    # Write the report to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))
    
    return output_path

def main():
    """
    Demo function to show how to use the bias mitigation module
    """
    # Create synthetic data with biased outcomes
    np.random.seed(42)
    n_samples = 1000
    
    # Group membership (binary for simplicity)
    group = np.random.choice(['A', 'B'], n_samples, p=[0.7, 0.3])  # Imbalanced groups
    
    # Create features
    X1 = np.random.normal(0, 1, n_samples)
    X2 = np.random.normal(0, 1, n_samples)
    
    # Biased feature that correlates with group
    X3 = np.random.normal(0, 1, n_samples)
    X3[group == 'A'] += 1  # Group A has higher values on average
    
    # Create biased target
    # Group B has a different relationship between features and target
    y = 2 * X1 + 3 * X2
    y[group == 'A'] += 0.5 * X3[group == 'A']  # X3 affects group A more
    y[group == 'B'] += np.random.normal(0, 5, sum(group == 'B'))  # More noise for group B
    
    # Create DataFrame
    X = pd.DataFrame({
        'X1': X1,
        'X2': X2, 
        'X3': X3,
        'group': group
    })
    
    # Evaluate bias mitigation methods
    evaluation_results = evaluate_bias_mitigation_methods(X, y, 'group')
    
    # Generate report
    report_path = generate_bias_mitigation_report(evaluation_results)
    print(f"Report generated: {report_path}")

if __name__ == '__main__':
    main() 