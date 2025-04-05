import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
import os
import joblib
import warnings

def calculate_regression_metrics(y_true, y_pred):
    """
    Calculate regression metrics
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
        
    Returns:
    --------
    dict
        Dictionary of regression metrics
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Calculate MAPE with better handling of zero/small values
    # Use a threshold to avoid division by very small numbers
    threshold = 10.0  # Define a minimal threshold for denominator
    abs_diff = np.abs(y_true - y_pred)
    
    # For MAPE calculation, only consider values where true value is above threshold
    valid_indices = y_true > threshold
    if np.any(valid_indices):
        mape = np.mean(abs_diff[valid_indices] / y_true[valid_indices]) * 100
    else:
        # If no valid indices, use MAE as a proportion of mean prediction
        mape = (mae / (np.mean(y_pred) + 1e-8)) * 100
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape
    }

def select_features(X, y, method='xgboost', threshold=0.01, k=10, visualize=True):
    """
    Select top features based on different methods
    
    Parameters:
    -----------
    X : pandas DataFrame
        Feature matrix
    y : array-like
        Target values
    method : str, optional
        Method to use for feature selection
        ('xgboost', 'mutual_info', 'f_regression', 'random_forest')
    threshold : float, optional
        Importance threshold for XGBoost and Random Forest methods
    k : int, optional
        Number of top features to select for mutual_info and f_regression
    visualize : bool, optional
        Whether to create visualization
        
    Returns:
    --------
    list
        List of selected feature names
    """
    # Get feature names
    feature_names = X.columns.tolist()
    
    # Replace infinity with NaN
    X = X.replace([np.inf, -np.inf], np.nan)
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # For test safety, drop non-numeric columns
    numeric_cols = X.select_dtypes(include=['number']).columns
    if len(numeric_cols) < len(X.columns):
        print(f"Warning: Dropping {len(X.columns) - len(numeric_cols)} non-numeric columns for feature selection")
        X = X[numeric_cols]
        feature_names = numeric_cols.tolist()
    
    # Initialize feature importances
    feature_importances = pd.DataFrame({'feature': feature_names})
    
    if method == 'xgboost':
        # Use XGBoost feature importance
        model = xgb.XGBRegressor()
        model.fit(X, y)
        importance = model.feature_importances_
        
        # Create feature importance DataFrame
        feature_importances['importance'] = importance
        feature_importances = feature_importances.sort_values('importance', ascending=False)
        
        # Select features above threshold
        selected_features = feature_importances[feature_importances['importance'] > threshold]['feature'].tolist()
        
    elif method == 'mutual_info':
        # Use mutual information
        selector = SelectKBest(mutual_info_regression, k=k)
        selector.fit(X, y)
        
        # Get scores
        feature_importances['importance'] = selector.scores_
        feature_importances = feature_importances.sort_values('importance', ascending=False)
        
        # Get selected features
        mask = selector.get_support()
        selected_features = [feature_names[i] for i in range(len(feature_names)) if mask[i]]
        
    elif method == 'f_regression':
        # Use f_regression
        selector = SelectKBest(f_regression, k=k)
        selector.fit(X, y)
        
        # Get scores
        feature_importances['importance'] = selector.scores_
        feature_importances = feature_importances.sort_values('importance', ascending=False)
        
        # Get selected features
        mask = selector.get_support()
        selected_features = [feature_names[i] for i in range(len(feature_names)) if mask[i]]
        
    elif method == 'random_forest':
        # Use Random Forest feature importance
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        importance = model.feature_importances_
        
        # Create feature importance DataFrame
        feature_importances['importance'] = importance
        feature_importances = feature_importances.sort_values('importance', ascending=False)
        
        # Select features above threshold
        selected_features = feature_importances[feature_importances['importance'] > threshold]['feature'].tolist()
        
    else:
        raise ValueError(f"Unsupported feature selection method: {method}")
    
    # Ensure at least 1 feature is selected
    if len(selected_features) == 0:
        print("Warning: No features were selected. Using top feature instead.")
        selected_features = [feature_importances.iloc[0]['feature']]
    
    # Create visualization
    if visualize:
        # Create directory if it doesn't exist
        os.makedirs('visualizations/feature_selection', exist_ok=True)
        
        plt.figure(figsize=(10, 8))
        
        # Get top 20 features (or all if less than 20)
        top_n = min(20, len(feature_importances))
        top_features = feature_importances.head(top_n)
        
        # Plot feature importance
        plt.barh(top_features['feature'], top_features['importance'])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(f'Feature Importance ({method})')
        plt.tight_layout()
        
        # Save plot
        plt.savefig(f'visualizations/feature_selection/feature_importance_{method}.png')
        plt.close()
    
    print(f"Selected {len(selected_features)} features using {method}")
    
    return selected_features

def temporal_cross_validation(X, y, date_index, model, n_splits=5, gap=30, window_type='expanding'):
    """
    Perform temporal cross-validation
    
    Parameters:
    -----------
    X : pandas DataFrame
        Feature matrix
    y : array-like
        Target values
    date_index : array-like
        Date index for each sample (datetime or date objects)
    model : estimator
        Model to evaluate
    n_splits : int, optional
        Number of splits for temporal CV
    gap : int, optional
        Gap in days between train and test sets
    window_type : str, optional
        Type of window ('expanding' or 'rolling')
        
    Returns:
    --------
    tuple
        (cv_scores, cv_predictions)
    """
    # Ensure date_index is a numpy array
    date_array = np.array(date_index)
    
    # Sort unique dates
    unique_dates = np.sort(np.unique(date_array))
    
    # Calculate splits
    date_points = np.linspace(0, len(unique_dates) - 1, n_splits + 1).astype(int)
    split_dates = unique_dates[date_points]
    
    # Initialize results
    cv_predictions = []
    cv_scores = []
    
    # Loop through splits
    for i in range(len(split_dates) - 1):
        # For expanding window, start from the beginning
        # For rolling window, start from the previous split
        if window_type == 'expanding':
            train_start_date = unique_dates[0]
        else:  # rolling window
            if i == 0:
                train_start_date = unique_dates[0]
            else:
                train_start_date = split_dates[i-1]
        
        # Define train and test dates
        train_end_date = split_dates[i]
        test_start_date = split_dates[i]
        test_end_date = split_dates[i+1]
        
        # Define train and test indices
        train_idx = np.where((date_array >= train_start_date) & (date_array <= train_end_date))[0]
        test_idx = np.where((date_array > test_start_date) & (date_array <= test_end_date))[0]
        
        # Add gap between train and test
        if gap > 0:
            # Find the latest date in train
            max_train_date = date_array[train_idx].max()
            # Add gap days
            gap_date = max_train_date + np.timedelta64(gap, 'D')
            # Adjust test indices to start after gap
            test_idx = np.array([idx for idx in test_idx if date_array[idx] >= gap_date])
        
        # Check if we have enough data
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        
        # Get train and test sets
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = calculate_regression_metrics(y_test, y_pred)
        
        # Store results
        cv_scores.append(metrics)
        cv_predictions.append({
            'train_start': train_start_date,
            'train_end': train_end_date,
            'test_start': test_start_date,
            'test_end': test_end_date,
            'y_true': y_test,
            'y_pred': y_pred,
            'train_size': len(train_idx),
            'test_size': len(test_idx)
        })
    
    return cv_scores, cv_predictions

def create_regression_confusion_matrix(y_true, y_pred, thresholds=None, visualize=True, filename='regression_confusion_matrix.png'):
    """
    Create a confusion matrix for regression predictions
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    thresholds : list, optional
        List of thresholds for binning the values
        If None, quartiles are used
    visualize : bool, optional
        Whether to create visualization
    filename : str, optional
        Filename for saving the visualization
        
    Returns:
    --------
    pandas DataFrame
        Confusion matrix as a DataFrame
    """
    # Create bins based on quartiles of true values if not provided
    if thresholds is None:
        thresholds = [
            np.percentile(y_true, 25),
            np.percentile(y_true, 50),
            np.percentile(y_true, 75)
        ]
    
    # Ensure thresholds are sorted
    thresholds = sorted(thresholds)
    
    # Create bin edges
    bin_edges = [-np.inf] + thresholds + [np.inf]
    
    # Create bin labels
    bin_labels = [f'Q{i+1}' for i in range(len(bin_edges) - 1)]
    
    # Bin the values
    y_true_binned = pd.cut(y_true, bins=bin_edges, labels=bin_labels)
    y_pred_binned = pd.cut(y_pred, bins=bin_edges, labels=bin_labels)
    
    # Create confusion matrix
    confusion = pd.crosstab(
        y_true_binned, y_pred_binned, 
        rownames=['Actual'], colnames=['Predicted'], 
        normalize='index'
    )
    
    # Create visualization
    if visualize:
        # Create directory if it doesn't exist
        os.makedirs('visualizations', exist_ok=True)
        
        plt.figure(figsize=(10, 8))
        sns = __import__('seaborn')
        
        # Plot confusion matrix
        sns.heatmap(confusion, annot=True, cmap='Blues', fmt='.2f', cbar=True)
        plt.title('Regression Confusion Matrix (Normalized by Row)')
        plt.tight_layout()
        
        # Save plot
        plt.savefig(f'visualizations/{filename}')
        plt.close()
    
    return confusion

def run_advanced_modeling_pipeline(X, y, date_index=None, feature_selection_method='xgboost', 
                                  model_type='xgboost', temporal_cv=True):
    """
    Run a complete advanced modeling pipeline with feature selection and temporal CV
    
    Parameters:
    -----------
    X : pandas DataFrame
        Feature matrix
    y : array-like
        Target values
    date_index : array-like, optional
        Date index for each sample (required if temporal_cv=True)
    feature_selection_method : str, optional
        Method to use for feature selection
    model_type : str, optional
        Type of model to train ('xgboost', 'random_forest')
    temporal_cv : bool, optional
        Whether to use temporal cross-validation
        
    Returns:
    --------
    dict
        Dictionary with results
    """
    print("Starting advanced modeling pipeline...")
    
    # Convert y to pandas Series if it's not already
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    
    # Check if date_index is provided for temporal CV
    if temporal_cv and date_index is None:
        print("Warning: date_index is required for temporal CV. Falling back to standard train/test split.")
        temporal_cv = False
    
    # Step 1: Feature selection
    print("\nPerforming feature selection...")
    selected_features = select_features(X, y, method=feature_selection_method)
    X_selected = X[selected_features]
    
    # Step 2: Split data for evaluation
    if not temporal_cv:
        # Use standard train/test split
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
    
    # Step 3: Create and train model
    print("\nTraining model...")
    
    if model_type == 'xgboost':
        model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
    elif model_type == 'random_forest':
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Step 4: Evaluate model
    if temporal_cv:
        print("\nPerforming temporal cross-validation...")
        cv_scores, cv_predictions = temporal_cross_validation(
            X_selected, y, date_index, model
        )
        
        # Calculate average metrics
        avg_metrics = {}
        for metric in cv_scores[0].keys():
            avg_metrics[metric] = np.mean([score[metric] for score in cv_scores])
        
        print("\nAverage CV Metrics:")
        for metric, value in avg_metrics.items():
            print(f"{metric}: {value:.4f}")
            
        # Train final model on all data
        model.fit(X_selected, y)
        
        # Make predictions on all data for visualizations
        y_pred_all = model.predict(X_selected)
        
    else:
        # Train model on training set
        model.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = model.predict(X_test)
        metrics = calculate_regression_metrics(y_test, y_pred)
        
        print("\nTest Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # Make predictions on all data for visualizations
        y_pred_all = model.predict(X_selected)
    
    # Step 5: Create visualizations
    print("\nCreating visualizations...")
    
    # Create directory if it doesn't exist
    os.makedirs('visualizations/model_results', exist_ok=True)
    
    # Create predicted vs actual scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(y, y_pred_all, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'{model_type.upper()}: Predicted vs Actual')
    plt.tight_layout()
    plt.savefig(f'visualizations/model_results/{model_type}_predicted_vs_actual.png')
    plt.close()
    
    # Create QQ plot
    plt.figure(figsize=(10, 8))
    plt.scatter(np.sort(y), np.sort(y_pred_all), alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel('Actual Quantiles')
    plt.ylabel('Predicted Quantiles')
    plt.title(f'{model_type.upper()}: Q-Q Plot')
    plt.tight_layout()
    plt.savefig(f'visualizations/model_results/{model_type}_qq_plot.png')
    plt.close()
    
    # Create error histogram
    plt.figure(figsize=(10, 8))
    plt.hist(y_pred_all - y, bins=30, alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title(f'{model_type.upper()}: Prediction Error Distribution')
    plt.tight_layout()
    plt.savefig(f'visualizations/model_results/{model_type}_error_distribution.png')
    plt.close()
    
    # Create regression confusion matrix
    confusion = create_regression_confusion_matrix(
        y, y_pred_all, 
        visualize=True, 
        filename=f'model_results/{model_type}_confusion_matrix.png'
    )
    
    # Step 6: Save model
    print("\nSaving model...")
    os.makedirs('models', exist_ok=True)
    model_path = f'models/{model_type}_advanced_model.pkl'
    
    # Save model with metadata
    model_info = {
        'model': model,
        'feature_selection_method': feature_selection_method,
        'selected_features': selected_features,
        'model_type': model_type,
        'temporal_cv': temporal_cv,
        'date_saved': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    joblib.dump(model_info, model_path)
    print(f"Model saved to {model_path}")
    
    # Return results
    if temporal_cv:
        return {
            'model': model,
            'selected_features': selected_features,
            'cv_scores': cv_scores,
            'cv_predictions': cv_predictions,
            'avg_metrics': avg_metrics,
            'confusion_matrix': confusion,
            'model_path': model_path
        }
    else:
        return {
            'model': model,
            'selected_features': selected_features,
            'metrics': metrics,
            'y_pred': y_pred,
            'y_test': y_test,
            'confusion_matrix': confusion,
            'model_path': model_path
        }

if __name__ == "__main__":
    # Example usage
    print("Running advanced modeling pipeline...")
    run_advanced_modeling_pipeline(
        X=None,  # Will be loaded in the function
        y=None,  # Will be loaded in the function
        feature_selection_method='xgboost',
        model_type='xgboost',
        temporal_cv=True
    )