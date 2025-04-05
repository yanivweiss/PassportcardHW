import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

# Import our modules
from data_preparation import prepare_data_for_modeling
from enhanced_data_preparation import enhanced_data_preparation
from enhanced_feature_engineering import enhanced_feature_engineering
from advanced_modeling import select_features, apply_smote, temporal_cross_validation, run_advanced_modeling_pipeline
from focal_loss import train_xgboost_with_focal_loss
from error_analysis import analyze_prediction_errors, create_regression_confusion_matrix, plot_error_heatmap

def run_full_enhanced_pipeline():
    """
    Run the complete enhanced data science pipeline
    """
    print("\n" + "="*80)
    print("RUNNING ENHANCED DATA SCIENCE PIPELINE")
    print("="*80 + "\n")
    
    # Step 1: Load and prepare data
    print("\n--- Step 1: Data Preparation ---\n")
    claims_df, members_df = prepare_data_for_modeling()
    
    # Step 2: Enhanced data preparation
    print("\n--- Step 2: Enhanced Data Preparation ---\n")
    cleaned_claims, scaler_claims, outlier_info_claims = enhanced_data_preparation(
        claims_df, 
        missing_strategy='knn',
        outlier_method='iqr',
        scaling_method='robust'
    )
    
    cleaned_members, scaler_members, outlier_info_members = enhanced_data_preparation(
        members_df, 
        missing_strategy='knn',
        outlier_method='iqr',
        scaling_method='robust'
    )
    
    # Step 3: Enhanced feature engineering
    print("\n--- Step 3: Enhanced Feature Engineering ---\n")
    date_columns = ['ServiceDate', 'PolicyStartDate', 'PolicyEndDate', 'DateOfBirth']
    enhanced_features_df = enhanced_feature_engineering(
        cleaned_claims, 
        cleaned_members, 
        date_columns=date_columns
    )
    
    # Step 4: Define cutoff date for training/testing
    cutoff_date = claims_df['ServiceDate'].max() - timedelta(days=180)
    print(f"\nUsing cutoff date: {cutoff_date} for training/testing split")
    
    # Step 5: Split data for modeling
    print("\n--- Step 4: Prepare Data for Modeling ---\n")
    
    # Filter claims data to include only data before cutoff date
    training_claims = claims_df[claims_df['ServiceDate'] <= cutoff_date]
    
    # Create the target variable: sum of claims in the 6 months after cutoff date
    test_claims = claims_df[
        (claims_df['ServiceDate'] > cutoff_date) & 
        (claims_df['ServiceDate'] <= cutoff_date + timedelta(days=180))
    ]
    
    # Group by member and sum claim amounts
    future_claims = test_claims.groupby('Member_ID')['TotPaymentUSD'].sum().reset_index()
    future_claims.columns = ['Member_ID', 'future_6m_claims']
    
    # Merge target with features
    modeling_data = pd.merge(
        enhanced_features_df, 
        future_claims, 
        on='Member_ID', 
        how='left'
    )
    
    # Fill missing target values with 0 (assuming no claims)
    modeling_data['future_6m_claims'].fillna(0, inplace=True)
    
    # Save the prepared data
    modeling_data.to_csv('enhanced_modeling_data.csv', index=False)
    print(f"Saved prepared data with {modeling_data.shape[1]} features for {modeling_data.shape[0]} members")
    
    # Step 6: Feature selection
    print("\n--- Step 5: Feature Selection ---\n")
    X = modeling_data.drop(['Member_ID', 'future_6m_claims'], axis=1, errors='ignore')
    y = modeling_data['future_6m_claims']
    
    # Keep only numeric columns
    X = X.select_dtypes(include=['int', 'float'])
    
    # Fill any remaining NaN values
    X = X.fillna(0)
    
    # Select features using multiple methods
    feature_selection_methods = ['xgboost', 'lasso', 'kbest']
    selected_features = {}
    
    for method in feature_selection_methods:
        print(f"\nSelecting features using {method} method...")
        selected_features[method], feature_importances = select_features(
            X, y, method=method, threshold=0.01, k=50, visualize=True
        )
        print(f"Selected {len(selected_features[method])} features using {method}")
    
    # Find common features across methods (more robust selection)
    common_features = list(set.intersection(*map(set, selected_features.values())))
    print(f"\nFound {len(common_features)} common features across all methods")
    
    # Use XGBoost selected features for further modeling
    final_features = selected_features['xgboost']
    X_selected = X[final_features]
    
    # Step 7: Apply SMOTE for imbalanced regression
    print("\n--- Step 6: Apply SMOTE for Imbalanced Regression ---\n")
    X_resampled, y_resampled = apply_smote(
        X_selected, y,
        categorical_features=None,
        sampling_strategy='auto',
        k_neighbors=5
    )
    
    # Step 8: Temporal cross-validation
    print("\n--- Step 7: Temporal Cross-Validation ---\n")
    
    # Get dates for members (latest claim date)
    member_dates = claims_df.groupby('Member_ID')['ServiceDate'].max().reset_index()
    member_dates = pd.Series(
        member_dates['ServiceDate'].values,
        index=member_dates['Member_ID']
    )
    
    # Match dates with our data order
    dates = modeling_data['Member_ID'].map(member_dates).fillna(pd.Timestamp('2000-01-01'))
    
    # Create a basic model for CV
    from xgboost import XGBRegressor
    base_model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    # Run temporal CV
    cv_results = temporal_cross_validation(
        X_selected, y,
        dates.values,
        base_model,
        n_splits=5,
        gap=30,
        visualize=True
    )
    
    # Step 9: Train with focal loss
    print("\n--- Step 8: Train with Focal Loss ---\n")
    
    # Split data for focal loss training
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42
    )
    
    # Train with focal loss
    focal_model, focal_predictions, focal_metrics = train_xgboost_with_focal_loss(
        X_train, y_train, X_test, y_test,
        gamma=2.0,  # Focusing parameter
        alpha=0.25  # More weight to under-predictions
    )
    
    # Step 10: Error analysis and confusion matrix
    print("\n--- Step 9: Error Analysis and Confusion Matrix ---\n")
    
    # Analyze prediction errors
    error_results = analyze_prediction_errors(
        y_test, focal_predictions,
        feature_matrix=X_test,
        feature_names=X_test.columns
    )
    
    # Create regression confusion matrix
    cm, bin_edges = create_regression_confusion_matrix(
        y_test, focal_predictions,
        n_classes=5,
        visualize=True
    )
    
    # Create error heatmap for top features
    top_features = X_selected.columns[:2]  # Use top 2 features
    plot_error_heatmap(
        y_test, focal_predictions,
        X_test[top_features[0]], X_test[top_features[1]],
        top_features[0], top_features[1]
    )
    
    # Step 11: Save final model
    print("\n--- Step 10: Save Final Models and Results ---\n")
    
    # Save the focal loss model
    focal_model_info = {
        'model': focal_model,
        'features': final_features,
        'metrics': focal_metrics,
        'bin_edges': bin_edges,
        'cv_results': cv_results,
        'error_analysis': error_results
    }
    
    joblib.dump(focal_model_info, 'best_focal_model.pkl')
    print("Saved focal loss model to best_focal_model.pkl")
    
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
    print("Focal Loss Model Metrics:")
    for metric, value in focal_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nStandard Model CV Metrics:")
    if advanced_results['cv_results'] is not None:
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