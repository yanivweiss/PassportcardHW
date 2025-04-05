import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib
import os
from scipy.stats import randint, uniform
import seaborn as sns
import re

def prepare_data_for_xgboost(features_df, target_col='future_6m_claims', test_size=0.2):
    """
    Prepare data specifically for XGBoost modeling
    """
    # Make a copy to avoid modifying the original
    df = features_df.copy()
    
    # Identify and exclude non-feature columns
    exclude_cols = ['Member_ID', 'PolicyID', target_col, 'PolicyStartDate', 'PolicyEndDate', 'DateOfBirth']
    exclude_cols = [col for col in exclude_cols if col in df.columns]
    
    # Remove datetime columns
    datetime_cols = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            datetime_cols.append(col)
            exclude_cols.append(col)
    
    # Get feature columns
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Replace infinity with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Handle missing and infinite values
    for col in feature_cols:
        # Check if the column has NaN values
        if df[col].isna().any():
            # For columns with more than 50% missing values, drop the column
            if df[col].isna().mean() > 0.5:
                feature_cols.remove(col)
            else:
                # For numeric columns, fill NaN with median
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median())
                else:
                    # For non-numeric columns, fill with mode
                    df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0)
    
    # Handle categorical features
    categorical_cols = []
    for col in feature_cols:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            categorical_cols.append(col)
            df[col] = pd.Categorical(df[col]).codes
    
    # Split features and target
    X = df[feature_cols]
    y = df[target_col]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Check for any remaining NaN or inf values (just to be safe)
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    
    return X_train, X_test, y_train, y_test, feature_cols

def basic_xgboost_train(X_train, y_train, X_test, y_test):
    """
    Train a basic XGBoost model with default parameters
    """
    # Create and train the model
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate
    metrics = evaluate_xgboost_model(y_test, y_pred)
    
    return model, y_pred, metrics

def optimize_xgboost_hyperparams(X_train, y_train, X_test, y_test, feature_cols=None, cv=5, n_iter=50):
    """
    Optimize XGBoost hyperparameters using RandomizedSearchCV
    """
    # Define parameter space
    param_space = {
        'n_estimators': randint(100, 1000),
        'learning_rate': uniform(0.01, 0.3),
        'max_depth': randint(3, 10),
        'min_child_weight': randint(1, 10),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'gamma': uniform(0, 1),
        'reg_alpha': uniform(0, 2),
        'reg_lambda': uniform(0, 2),
        'scale_pos_weight': uniform(0.8, 0.4)
    }
    
    # Base model
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    
    # Setup cross-validation
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Randomized search
    search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_space,
        n_iter=n_iter,
        scoring='neg_root_mean_squared_error',
        cv=kf,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    # Fit search
    search.fit(X_train, y_train)
    
    # Get best model
    best_model = search.best_estimator_
    best_params = search.best_params_
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    metrics = evaluate_xgboost_model(y_test, y_pred)
    
    print(f"Best Parameters: {best_params}")
    print(f"Best RMSE: {-search.best_score_:.4f}")
    
    # Print top features
    if feature_cols is not None:
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
    else:
        # Use feature indices if feature names are not provided
        feature_importance = pd.DataFrame({
            'feature': [f'feature_{i}' for i in range(X_train.shape[1])],
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
    
    print("Top 10 Features:")
    print(feature_importance.head(10))
    
    return best_model, y_pred, metrics, best_params, feature_importance

def advanced_xgboost_train(X_train, y_train, X_test, y_test, params=None):
    """
    Train XGBoost model with early stopping and learning rate schedule
    """
    # Create evaluation datasets
    eval_set = [(X_train, y_train), (X_test, y_test)]
    
    # Default parameters if none provided
    if params is None:
        params = {
            'objective': 'reg:squarederror',
            'n_estimators': 1000,
            'learning_rate': 0.05,
            'max_depth': 6,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 0.5,
            'random_state': 42
        }
    
    # Create model
    model = xgb.XGBRegressor(**params)
    
    # Train with early stopping
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        eval_metric=['rmse', 'mae'],
        early_stopping_rounds=50,
        verbose=False
    )
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate
    metrics = evaluate_xgboost_model(y_test, y_pred)
    
    # Get actual best iteration
    best_iteration = model.best_iteration
    print(f"Best iteration: {best_iteration}")
    
    # Extract feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return model, y_pred, metrics, feature_importance

def evaluate_xgboost_model(y_true, y_pred):
    """
    Evaluate XGBoost model with multiple metrics
    """
    # Calculate standard metrics
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
    
    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape
    }
    
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return metrics

def plot_xgboost_results(model, X_test, y_test, feature_importance):
    """
    Create visualization for XGBoost model results
    """
    # Create output directory if it doesn't exist
    os.makedirs('visualizations', exist_ok=True)
    
    # 1. Predictions vs Actual
    y_pred = model.predict(X_test)
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('XGBoost: Predicted vs Actual Values')
    plt.tight_layout()
    plt.savefig('visualizations/xgboost_predictions.png')
    plt.close()
    
    # 2. Feature Importance
    plt.figure(figsize=(12, 10))
    feature_importance.head(20).plot(x='feature', y='importance', kind='barh')
    plt.title('XGBoost: Top 20 Feature Importance')
    plt.tight_layout()
    plt.savefig('visualizations/xgboost_feature_importance.png')
    plt.close()
    
    # 3. Learning Curves (if results available)
    try:
        if hasattr(model, 'evals_result') and callable(model.evals_result):
            results = model.evals_result()
            if results and 'validation_0' in results and 'rmse' in results['validation_0']:
                epochs = len(results['validation_0']['rmse'])
                x_axis = range(0, epochs)
                
                # RMSE
                plt.figure(figsize=(10, 6))
                plt.plot(x_axis, results['validation_0']['rmse'], label='Train')
                plt.plot(x_axis, results['validation_1']['rmse'], label='Test')
                plt.legend()
                plt.xlabel('Boosting Rounds')
                plt.ylabel('RMSE')
                plt.title('XGBoost RMSE')
                plt.tight_layout()
                plt.savefig('visualizations/xgboost_learning_curve_rmse.png')
                plt.close()
                
                # MAE
                if 'mae' in results['validation_0']:
                    plt.figure(figsize=(10, 6))
                    plt.plot(x_axis, results['validation_0']['mae'], label='Train')
                    plt.plot(x_axis, results['validation_1']['mae'], label='Test')
                    plt.legend()
                    plt.xlabel('Boosting Rounds')
                    plt.ylabel('MAE')
                    plt.title('XGBoost MAE')
                    plt.tight_layout()
                    plt.savefig('visualizations/xgboost_learning_curve_mae.png')
                    plt.close()
    except Exception as e:
        print(f"Note: Could not create learning curve plots: {e}")
        print("This is expected when using RandomizedSearchCV which doesn't track eval metrics across rounds.")
        
        # Create a basic learning curve plot using predictions vs actual
        plt.figure(figsize=(10, 6))
        plt.hist(y_pred - y_test, bins=30, alpha=0.7)
        plt.axvline(x=0, color='r', linestyle='--')
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.title('XGBoost Prediction Error Distribution')
        plt.tight_layout()
        plt.savefig('visualizations/xgboost_error_distribution.png')
        plt.close()

def save_xgboost_model(model, feature_cols, file_path='best_xgboost_model.pkl'):
    """
    Save XGBoost model and related information to a file
    """
    model_info = {
        'model': model,
        'feature_cols': feature_cols,
        'date_saved': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    joblib.dump(model_info, file_path)
    print(f"Model saved to {file_path}")

def clean_feature_names(X_df):
    """
    Clean feature names to be valid for XGBoost (no [, ], <, or special characters)
    
    Parameters:
    -----------
    X_df : pandas DataFrame
        Feature matrix with potentially invalid column names
        
    Returns:
    --------
    pandas DataFrame
        DataFrame with clean column names
    """
    X_clean = X_df.copy()
    
    # Function to clean each column name
    def clean_name(name):
        # Replace any special characters with underscore
        clean = re.sub(r'[\[\]<>,\(\)\{\}]', '_', name)
        # Remove any consecutive underscores
        clean = re.sub(r'_+', '_', clean)
        # Remove leading/trailing underscores
        clean = clean.strip('_')
        return clean
    
    # Create a mapping of old to new names
    name_mapping = {col: clean_name(col) for col in X_clean.columns}
    
    # Rename columns
    X_clean = X_clean.rename(columns=name_mapping)
    
    return X_clean, name_mapping

def train_xgboost_model(X, y, test_size=0.2, random_state=42, optimize=False, n_iter=10):
    """
    Train an XGBoost regression model
    
    Parameters:
    -----------
    X : pandas DataFrame
        Feature matrix
    y : pandas Series
        Target variable
    test_size : float
        Proportion of data to use for testing
    random_state : int
        Random seed for reproducibility
    optimize : bool
        Whether to perform hyperparameter optimization
    n_iter : int
        Number of parameter settings to try for optimization
        
    Returns:
    --------
    dict
        Dictionary with model, feature importance, test data, and predictions
    """
    # Clean feature names for XGBoost compatibility
    X_clean, name_mapping = clean_feature_names(X)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_clean, y, test_size=test_size, random_state=random_state)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Define base XGBoost model
    base_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state
    )
    
    if optimize:
        print("Performing hyperparameter optimization...")
        
        # Define parameter space for optimization
        param_space = {
            'n_estimators': [50, 100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 4, 5, 6, 7],
            'min_child_weight': [1, 3, 5, 7],
            'subsample': [0.6, 0.7, 0.8, 0.9],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
            'gamma': [0, 0.1, 0.2, 0.3],
            'reg_alpha': [0, 0.1, 1, 10],
            'reg_lambda': [0, 0.1, 1, 10]
        }
        
        # Set up RandomizedSearchCV
        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_space,
            n_iter=n_iter,
            scoring='neg_mean_squared_error',
            cv=5,
            verbose=1,
            random_state=random_state,
            n_jobs=-1
        )
        
        try:
            # Fit on training data
            search.fit(X_train, y_train)
            
            print(f"Best parameters: {search.best_params_}")
            print(f"Best CV score: {np.sqrt(-search.best_score_):.4f} (RMSE)")
            
            # Use the best model
            model = search.best_estimator_
        except Exception as e:
            print(f"Error during hyperparameter optimization: {e}")
            print("Falling back to default model")
            model = base_model
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10, verbose=False)
    else:
        # Use the base model
        model = base_model
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10, verbose=False)
    
    # Make predictions on test set
    predictions = model.predict(X_test)
    
    # Get feature importance
    importance = model.feature_importances_
    
    # Map back to original feature names for readability
    reverse_mapping = {v: k for k, v in name_mapping.items()}
    
    feature_importance = pd.DataFrame({
        'feature': [reverse_mapping.get(col, col) for col in X_clean.columns],
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    return {
        'model': model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'predictions': predictions,
        'feature_importance': feature_importance,
        'name_mapping': name_mapping
    }

def evaluate_xgboost_model(model, X_test, y_test):
    """
    Evaluate an XGBoost model and return performance metrics
    
    Parameters:
    -----------
    model : XGBRegressor
        Trained XGBoost model
    X_test : pandas DataFrame
        Test features
    y_test : pandas Series
        Test target values
        
    Returns:
    --------
    dict
        Dictionary with evaluation metrics and predictions
    """
    # Make predictions
    predictions = model.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    # Add small epsilon to avoid division by zero
    epsilon = 1e-10
    mape = np.mean(np.abs((y_test - predictions) / (y_test + epsilon))) * 100
    
    metrics = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape
    }
    
    print("Model Evaluation:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R²: {r2:.4f}")
    print(f"  MAPE: {mape:.4f}%")
    
    # Visualize predictions vs actual
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, predictions, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Predicted vs Actual Values')
    
    # Create output directory if it doesn't exist
    os.makedirs('visualizations', exist_ok=True)
    
    plt.savefig('visualizations/xgboost_predictions.png')
    plt.close()
    
    return {
        'metrics': metrics,
        'predictions': predictions
    }

def run_complete_xgboost_pipeline(features_df, target_col, optimize=True, n_iter=10, save_model=True):
    """
    Run a complete XGBoost modeling pipeline
    
    Parameters:
    -----------
    features_df : pandas DataFrame
        DataFrame with features and target variable
    target_col : str
        Name of the target column
    optimize : bool
        Whether to perform hyperparameter optimization
    n_iter : int
        Number of parameter settings to try for optimization
    save_model : bool
        Whether to save the model to disk
        
    Returns:
    --------
    dict
        Dictionary with model information and metrics
    """
    print("Starting XGBoost modeling pipeline...")
    
    # Split features and target
    X = features_df.drop(columns=[target_col])
    if 'Member_ID' in X.columns:
        X = X.drop(columns=['Member_ID'])
    y = features_df[target_col]
    
    print(f"Features: {X.shape[1]}")
    print(f"Samples: {X.shape[0]}")
    
    # Handle missing values
    if X.isna().any().any():
        print(f"Found {X.isna().sum().sum()} missing values. Filling with zeros.")
        X = X.fillna(0)
    
    # Train model
    model_info = train_xgboost_model(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        optimize=optimize,
        n_iter=n_iter
    )
    
    # Evaluate model
    evaluation = evaluate_xgboost_model(
        model_info['model'],
        model_info['X_test'],
        model_info['y_test']
    )
    
    model_info.update(evaluation)
    
    # Save model and feature importance
    if save_model:
        import joblib
        
        joblib.dump(model_info['model'], 'best_xgboost_model.pkl')
        model_info['feature_importance'].to_csv('feature_importance_xgboost.csv', index=False)
        
        print("Model saved to best_xgboost_model.pkl")
        print("Feature importance saved to feature_importance_xgboost.csv")
    
    return model_info 