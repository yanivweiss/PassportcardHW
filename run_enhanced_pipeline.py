import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import sys
import time

# Import our modules
from data_preparation import load_data, preprocess_data
from feature_engineering import create_basic_features
from enhanced_features import create_enhanced_features, create_temporal_features
from advanced_modeling import select_features, train_test_temporal_split
from error_analysis import analyze_prediction_errors, create_regression_confusion_matrix, plot_error_heatmap
from run_enhanced_modeling import run_advanced_modeling_pipeline

def run_full_enhanced_pipeline():
    """
    Run the complete enhanced data science pipeline
    """
    print("Starting enhanced data science pipeline...")
    start_time = time.time()
    
    # Step 1: Load and preprocess data
    print("\n--- Step 1: Data Loading and Preprocessing ---\n")
    claims_df, members_df = load_data()
    claims_df, members_df = preprocess_data(claims_df, members_df)
    
    # Step 2: Create basic features
    print("\n--- Step 2: Creating Basic Features ---\n")
    basic_features = create_basic_features(claims_df, members_df)
    
    # Step 3: Create enhanced features
    print("\n--- Step 3: Creating Enhanced Features ---\n")
    enhanced_features = create_enhanced_features(claims_df, members_df)
    
    # Step 4: Create temporal features
    print("\n--- Step 4: Creating Temporal Features ---\n")
    temporal_features = create_temporal_features(claims_df, members_df)
    
    # Merge all features
    all_features = pd.merge(basic_features, enhanced_features, on='Member_ID', how='outer')
    all_features = pd.merge(all_features, temporal_features, on='Member_ID', how='outer')
    
    # Fill NaN values created during merging
    all_features = all_features.fillna(0)
    
    # Step 5: Prepare target variable
    print("\n--- Step 5: Preparing Target Variable ---\n")
    # Target is already in the features DataFrame as 'future_6m_claims'
    target_col = 'future_6m_claims'
    if target_col not in all_features.columns:
        print(f"Error: Target column '{target_col}' not found in features")
        sys.exit(1)
    
    X = all_features.drop(columns=[target_col, 'Member_ID'])
    y = all_features[target_col]
    
    # Step 6: Feature selection
    print("\n--- Step 6: Feature Selection ---\n")
    selected_features = select_features(X, y, method='xgboost', k=30)  # Select top 30 features
    X_selected = X[selected_features].copy()
    
    # Save feature names for later
    final_features = selected_features
    
    # Save to CSV for analysis
    feature_importance_df = pd.DataFrame({
        'feature': X_selected.columns,
        'importance': [1] * len(X_selected.columns)  # Placeholder, will be updated after model training
    })
    feature_importance_df.to_csv('selected_features.csv', index=False)
    print(f"Selected {len(selected_features)} features")
    
    # Step 7: Train-test split with temporal validation
    print("\n--- Step 7: Temporal Train-Test Split ---\n")
    X_train, X_test, y_train, y_test = train_test_temporal_split(
        X_selected, y, test_size=0.2, random_state=42
    )
    
    # Step 8: XGBoost modeling with temporal validation
    print("\n--- Step 8: XGBoost Modeling ---\n")
    import xgboost as xgb
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    xgb_model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    # Train the model
    xgb_model.fit(X_train, y_train, 
                 eval_set=[(X_train, y_train), (X_test, y_test)],
                 eval_metric='rmse',
                 early_stopping_rounds=20,
                 verbose=True)
    
    # Get predictions
    predictions = xgb_model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_test, predictions)),
        'MAE': mean_absolute_error(y_test, predictions),
        'R2': r2_score(y_test, predictions)
    }
    
    print("\nModel Performance:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Step 9: Error analysis and confusion matrix
    print("\n--- Step 9: Error Analysis and Confusion Matrix ---\n")
    
    # Analyze prediction errors
    error_results = analyze_prediction_errors(
        y_test, predictions,
        feature_matrix=X_test,
        feature_names=X_test.columns
    )
    
    # Create regression confusion matrix
    cm, bin_edges = create_regression_confusion_matrix(
        y_test, predictions,
        n_classes=5,
        visualize=True
    )
    
    # Create error heatmap for top features
    top_features = X_selected.columns[:2]  # Use top 2 features
    plot_error_heatmap(
        y_test, predictions,
        X_test[top_features[0]], X_test[top_features[1]],
        top_features[0], top_features[1]
    )
    
    # Step 10: Save final model
    print("\n--- Step 10: Save Final Models and Results ---\n")
    
    # Save the model
    model_info = {
        'model': xgb_model,
        'features': final_features,
        'metrics': metrics,
        'bin_edges': bin_edges,
        'error_analysis': error_results
    }
    
    joblib.dump(model_info, 'best_xgboost_model.pkl')
    print("Saved model to best_xgboost_model.pkl")
    
    # Run the standard advanced pipeline too for comparison
    print("\n--- Step 11: Run Standard Advanced Pipeline for Comparison ---\n")
    advanced_results = run_advanced_modeling_pipeline(
        claims_df=claims_df,
        members_df=members_df,
        feature_selection_method='xgboost',
        use_smote=True,
        temporal_cv=True
    )
    
    # Compare results
    print("\n--- Final Results Comparison ---\n")
    print("XGBoost Model Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nStandard Model CV Metrics:")
    if 'cv_results' in advanced_results and advanced_results['cv_results'] is not None:
        for metric in ['avg_rmse', 'avg_mae', 'avg_r2']:
            print(f"  {metric.upper()}: {advanced_results['cv_results'][metric]:.4f}")
    
    print("\nEnhanced data science pipeline completed!")
    print("Check the 'visualizations' directory for detailed analysis and results.")

if __name__ == "__main__":
    # Check if directories exist, create if not
    os.makedirs('visualizations', exist_ok=True)
    
    # Run the full pipeline
    try:
        run_full_enhanced_pipeline()
    except Exception as e:
        print(f"Error running pipeline: {e}")
        # Fallback to just the advanced pipeline if there are issues
        print("Falling back to standard advanced pipeline...")
        run_advanced_modeling_pipeline(
            claims_df=None,  # Will be loaded in the function
            members_df=None,  # Will be loaded in the function
            feature_selection_method='xgboost',
            use_smote=True,
            temporal_cv=True
        ) 