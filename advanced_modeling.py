import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_regression, RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from datetime import datetime, timedelta

def select_features(X, y, method='xgboost', threshold=0.01, k=None, visualize=True):
    """
    Select important features using different methods
    
    Parameters:
    -----------
    X : pandas DataFrame
        Feature matrix
    y : pandas Series
        Target variable
    method : str
        Method for feature selection: 'xgboost', 'lasso', 'randomforest', 'kbest', or 'rfe'
    threshold : float
        Importance threshold for feature selection (for tree-based methods)
    k : int, optional
        Number of features to select (for kbest and rfe methods)
    visualize : bool
        Whether to create feature importance visualization
        
    Returns:
    --------
    tuple
        (selected_features, feature_importances) where feature_importances is a DataFrame with feature names and importance scores
    """
    print(f"Selecting features using {method} method...")
    feature_names = X.columns.tolist()
    
    if method == 'xgboost':
        # Use XGBoost's feature importance
        model = xgb.XGBRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        importances = model.feature_importances_
        
        # Create feature importance DataFrame
        feature_importances = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Select features above threshold
        selected_features = feature_importances[feature_importances['importance'] > threshold]['feature'].tolist()
    
    elif method == 'lasso':
        # Use Lasso for feature selection
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = Lasso(alpha=0.01, random_state=42)
        model.fit(X_scaled, y)
        importances = np.abs(model.coef_)
        
        # Create feature importance DataFrame
        feature_importances = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Select non-zero coefficient features
        selected_features = [feature_names[i] for i in range(len(feature_names)) if importances[i] > 0]
    
    elif method == 'randomforest':
        # Use Random Forest for feature selection
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        importances = model.feature_importances_
        
        # Create feature importance DataFrame
        feature_importances = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Select features above threshold
        selected_features = feature_importances[feature_importances['importance'] > threshold]['feature'].tolist()
    
    elif method == 'kbest':
        # Use SelectKBest with f_regression
        if k is None:
            k = min(50, X.shape[1] // 2)  # Default to half of features or 50, whichever is smaller
            
        selector = SelectKBest(f_regression, k=k)
        selector.fit(X, y)
        importances = selector.scores_
        
        # Create feature importance DataFrame
        feature_importances = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Get selected feature mask and feature names
        selected_mask = selector.get_support()
        selected_features = [feature_names[i] for i in range(len(feature_names)) if selected_mask[i]]
    
    elif method == 'rfe':
        # Use Recursive Feature Elimination
        if k is None:
            k = min(50, X.shape[1] // 2)  # Default to half of features or 50, whichever is smaller
            
        base_model = RandomForestRegressor(n_estimators=50, random_state=42)
        selector = RFE(base_model, n_features_to_select=k, step=0.1)
        selector.fit(X, y)
        
        # Get selected feature mask and feature names
        selected_mask = selector.support_
        importances = selector.ranking_  # Lower is better
        
        # Inverse ranking to get higher values for more important features
        max_rank = max(importances)
        importances = max_rank - importances + 1
        
        # Create feature importance DataFrame
        feature_importances = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        selected_features = [feature_names[i] for i in range(len(feature_names)) if selected_mask[i]]
    
    else:
        raise ValueError(f"Unknown feature selection method: {method}")
    
    print(f"Selected {len(selected_features)} features out of {len(feature_names)}")
    
    # Create visualizations if requested
    if visualize:
        # Create output directory if it doesn't exist
        os.makedirs('visualizations/feature_selection', exist_ok=True)
        
        # Plot top features
        plt.figure(figsize=(12, 10))
        top_features = feature_importances.head(30)
        sns.barplot(x='importance', y='feature', data=top_features)
        plt.title(f'Top 30 Features Selected by {method.upper()}')
        plt.tight_layout()
        plt.savefig(f'visualizations/feature_selection/{method}_top_features.png')
        plt.close()
        
        # Save full feature importance list to CSV
        feature_importances.to_csv(f'feature_importance_{method}.csv', index=False)
    
    return selected_features, feature_importances

def apply_smote(X, y, categorical_features=None, sampling_strategy='auto', k_neighbors=5):
    """
    Apply SMOTE to handle imbalanced regression data
    
    Parameters:
    -----------
    X : pandas DataFrame
        Feature matrix
    y : pandas Series
        Target variable
    categorical_features : list or None
        List of categorical feature indices or names
    sampling_strategy : float, dict, or str
        Sampling strategy for SMOTE
    k_neighbors : int
        Number of nearest neighbors to use for SMOTE
        
    Returns:
    --------
    tuple
        (X_resampled, y_resampled) as numpy arrays
    """
    print("Applying SMOTE for imbalanced regression data...")
    
    # For regression, create bins to treat as classes
    # This is a workaround to use SMOTE with regression
    y_binned = pd.qcut(y, q=5, labels=False, duplicates='drop')
    
    # Convert categorical indices to column indices if needed
    if categorical_features is not None and isinstance(categorical_features[0], str):
        cat_indices = [X.columns.get_loc(col) for col in categorical_features if col in X.columns]
    else:
        cat_indices = categorical_features
    
    # Apply SMOTE
    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=k_neighbors,
        random_state=42,
        categorical_features=cat_indices
    )
    
    X_resampled, y_binned_resampled = smote.fit_resample(X, y_binned)
    
    # Map back to original continuous values
    # We'll use the mean value of each bin as the synthetic value
    bin_means = {}
    for bin_idx in range(len(np.unique(y_binned))):
        bin_means[bin_idx] = y[y_binned == bin_idx].mean()
    
    # Create synthetic continuous target
    y_resampled = np.array([bin_means[bin_idx] for bin_idx in y_binned_resampled])
    
    print(f"Original data shape: {X.shape}, Resampled data shape: {X_resampled.shape}")
    
    # Plot distribution before and after SMOTE
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    sns.histplot(y, kde=True)
    plt.title('Original Target Distribution')
    
    plt.subplot(1, 2, 2)
    sns.histplot(y_resampled, kde=True)
    plt.title('SMOTE-Resampled Target Distribution')
    
    plt.tight_layout()
    
    # Create visualizations directory if it doesn't exist
    os.makedirs('visualizations/smote', exist_ok=True)
    plt.savefig('visualizations/smote/smote_distribution_comparison.png')
    plt.close()
    
    return X_resampled, y_resampled

def temporal_cross_validation(X, y, date_index, model, n_splits=5, gap=30, visualize=True):
    """
    Perform temporal cross-validation with time series split
    
    Parameters:
    -----------
    X : pandas DataFrame
        Feature matrix
    y : pandas Series
        Target variable
    date_index : pandas Series or array
        Date index for time series split
    model : estimator object
        Scikit-learn compatible model with fit and predict methods
    n_splits : int
        Number of splits for time series CV
    gap : int
        Number of days to use as gap between train and test
    visualize : bool
        Whether to create visualization of CV splits
        
    Returns:
    --------
    dict
        Cross-validation results with metrics
    """
    print(f"Performing temporal cross-validation with {n_splits} splits and {gap} day gap...")
    
    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(date_index):
        date_index = pd.to_datetime(date_index)
    
    # Create TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Prepare data
    X_array = X.values if isinstance(X, pd.DataFrame) else X
    y_array = y.values if isinstance(y, pd.Series) else y
    date_array = date_index.values if isinstance(date_index, pd.Series) else date_index
    
    # Visualization setup
    if visualize:
        plt.figure(figsize=(15, 10))
        
    # Store results for each fold
    cv_results = {
        'rmse': [],
        'mae': [],
        'r2': [],
        'train_size': [],
        'test_size': [],
        'train_start': [],
        'train_end': [],
        'test_start': [],
        'test_end': []
    }
    
    # Perform cross-validation
    for i, (train_idx, test_idx) in enumerate(tscv.split(X_array)):
        # Add gap between train and test
        if gap > 0:
            # Find the latest date in train
            max_train_date = date_array[train_idx].max()
            # Add gap days
            gap_date = max_train_date + np.timedelta64(gap, 'D')
            # Adjust test indices to start after gap
            test_idx = np.array([idx for idx in test_idx if date_array[idx] >= gap_date])
            
            # If there are no test points left after the gap, skip this fold
            if len(test_idx) == 0:
                continue
        
        # Split data
        X_train, X_test = X_array[train_idx], X_array[test_idx]
        y_train, y_test = y_array[train_idx], y_array[test_idx]
        
        # Store fold dates
        cv_results['train_start'].append(date_array[train_idx].min())
        cv_results['train_end'].append(date_array[train_idx].max())
        cv_results['test_start'].append(date_array[test_idx].min())
        cv_results['test_end'].append(date_array[test_idx].max())
        cv_results['train_size'].append(len(train_idx))
        cv_results['test_size'].append(len(test_idx))
        
        # Fit model and predict
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Store metrics
        cv_results['rmse'].append(rmse)
        cv_results['mae'].append(mae)
        cv_results['r2'].append(r2)
        
        # Visualize split
        if visualize:
            plt.subplot(n_splits, 1, i + 1)
            
            # Plot train and test indices
            plt.scatter(date_array[train_idx], [i + 0.1] * len(train_idx), 
                     c='blue', marker='o', s=5, label='Train' if i == 0 else "")
            plt.scatter(date_array[test_idx], [i + 0.2] * len(test_idx), 
                     c='red', marker='o', s=5, label='Test' if i == 0 else "")
            
            # Add metrics text
            plt.text(date_array.min(), i + 0.3, 
                  f"Fold {i+1}: RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.2f}", 
                  fontsize=10)
            
            if i == 0:
                plt.legend(loc='upper right')
            
            plt.title(f"Fold {i+1}")
            if i == n_splits - 1:
                plt.xlabel('Date')
    
    # Save visualization
    if visualize:
        plt.suptitle("Temporal Cross-Validation Splits")
        plt.tight_layout()
        
        # Create directory if it doesn't exist
        os.makedirs('visualizations/cross_validation', exist_ok=True)
        plt.savefig('visualizations/cross_validation/temporal_cv_splits.png')
        plt.close()
        
        # Create a summary plot of metrics across folds
        plt.figure(figsize=(15, 6))
        
        plt.subplot(1, 3, 1)
        plt.plot(range(1, len(cv_results['rmse'])+1), cv_results['rmse'], 'o-')
        plt.title('RMSE by Fold')
        plt.xlabel('Fold')
        plt.ylabel('RMSE')
        
        plt.subplot(1, 3, 2)
        plt.plot(range(1, len(cv_results['mae'])+1), cv_results['mae'], 'o-')
        plt.title('MAE by Fold')
        plt.xlabel('Fold')
        plt.ylabel('MAE')
        
        plt.subplot(1, 3, 3)
        plt.plot(range(1, len(cv_results['r2'])+1), cv_results['r2'], 'o-')
        plt.title('R² by Fold')
        plt.xlabel('Fold')
        plt.ylabel('R²')
        
        plt.tight_layout()
        plt.savefig('visualizations/cross_validation/temporal_cv_metrics.png')
        plt.close()
    
    # Calculate average metrics
    cv_results['avg_rmse'] = np.mean(cv_results['rmse'])
    cv_results['avg_mae'] = np.mean(cv_results['mae'])
    cv_results['avg_r2'] = np.mean(cv_results['r2'])
    
    print(f"Average CV metrics - RMSE: {cv_results['avg_rmse']:.4f}, MAE: {cv_results['avg_mae']:.4f}, R²: {cv_results['avg_r2']:.4f}")
    
    return cv_results

def run_advanced_modeling_pipeline(claims_df, members_df, date_col='ServiceDate', 
                                 target_col='future_6m_claims', feature_selection_method='xgboost',
                                 use_smote=True, temporal_cv=True):
    """
    Run a complete advanced modeling pipeline with feature selection, SMOTE, and temporal CV
    
    Parameters:
    -----------
    claims_df : pandas DataFrame
        The claims dataframe
    members_df : pandas DataFrame
        The members dataframe
    date_col : str
        Column name containing dates for temporal CV
    target_col : str
        Column name containing the target variable
    feature_selection_method : str
        Method for feature selection
    use_smote : bool
        Whether to apply SMOTE for imbalanced regression
    temporal_cv : bool
        Whether to use temporal cross-validation
        
    Returns:
    --------
    dict
        Results of the modeling pipeline
    """
    print("Starting advanced modeling pipeline...")
    
    # Assuming we have a combined features dataframe from previous steps
    # For this example, we'll create a simple combined dataframe
    from data_preparation import prepare_data_for_modeling
    from feature_engineering import prepare_features_for_modeling
    from enhanced_feature_engineering import enhanced_feature_engineering
    
    print("Preparing data...")
    claims, members = prepare_data_for_modeling()
    
    print("Engineering features...")
    cutoff_date = claims[date_col].max() - timedelta(days=180)
    features_df = prepare_features_for_modeling(claims, members, cutoff_date)
    
    print("Applying advanced feature engineering...")
    enhanced_features = enhanced_feature_engineering(claims, members)
    
    # Combine with the regular features
    combined_features = pd.merge(features_df, enhanced_features, on='Member_ID', how='left')
    
    # Prepare data for modeling
    X = combined_features.drop([target_col, 'Member_ID', 'PolicyID'], axis=1, errors='ignore')
    y = combined_features[target_col]
    
    # Drop date columns and any other non-numeric columns
    X = X.select_dtypes(include=['int', 'float'])
    
    # Handle NaN values
    X = X.fillna(0)
    
    # Feature selection
    selected_features, feature_importances = select_features(
        X, y, 
        method=feature_selection_method,
        threshold=0.01,
        visualize=True
    )
    
    # Use selected features
    X_selected = X[selected_features]
    
    # Apply SMOTE if requested
    if use_smote:
        # Identify categorical columns for SMOTE
        categorical_cols = []  # No categorical columns after one-hot encoding
        X_resampled, y_resampled = apply_smote(
            X_selected, y,
            categorical_features=categorical_cols
        )
    else:
        X_resampled, y_resampled = X_selected.values, y.values
    
    # Choose a model
    model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    # Temporal cross-validation if requested
    if temporal_cv:
        # Extract dates for each member (using the latest claim date for each member)
        member_dates = claims.groupby('Member_ID')[date_col].max().reset_index()
        date_index = pd.Series(index=member_dates['Member_ID'], data=member_dates[date_col])
        
        # Match dates with data order
        date_values = np.array([
            date_index.get(member_id, pd.Timestamp('2000-01-01')) 
            for member_id in combined_features['Member_ID']
        ])
        
        # Run temporal CV
        cv_results = temporal_cross_validation(
            X_resampled, y_resampled,
            date_values, model,
            n_splits=5, gap=30,
            visualize=True
        )
    else:
        # Train on full dataset if not using CV
        model.fit(X_resampled, y_resampled)
        cv_results = None
    
    # Train final model on all data
    final_model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    final_model.fit(X_resampled, y_resampled)
    
    # Save final model and feature list
    model_info = {
        'model': final_model,
        'selected_features': selected_features,
        'feature_importances': feature_importances,
        'cv_results': cv_results
    }
    
    # Save model
    joblib.dump(model_info, 'advanced_model.pkl')
    
    print("Advanced modeling pipeline completed and model saved.")
    
    return model_info

if __name__ == "__main__":
    # Example usage
    print("Running advanced modeling pipeline...")
    run_advanced_modeling_pipeline(
        claims_df=None,  # Will be loaded in the function
        members_df=None,  # Will be loaded in the function
        feature_selection_method='xgboost',
        use_smote=True,
        temporal_cv=True
    )