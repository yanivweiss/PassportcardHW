"""
Evaluate the effectiveness of different feature engineering techniques
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime, timedelta

# Import data processing modules
from data_preparation import prepare_data_for_modeling
from feature_engineering import prepare_features_for_modeling

# Import advanced feature modules
from enhanced_feature_engineering import create_date_features, create_cyclical_features, create_customer_behavior_features
from enhanced_data_preparation import handle_missing_values_advanced, detect_and_handle_outliers
from advanced_modeling import select_features

def evaluate_feature_effectiveness():
    """Evaluate which feature engineering techniques improve model performance"""
    print("=" * 80)
    print("EVALUATING FEATURE ENGINEERING EFFECTIVENESS")
    print("=" * 80)
    
    # Create output directory for visualizations
    os.makedirs('visualizations/feature_evaluation', exist_ok=True)
    
    # Step 1: Load data
    print("\n1. Loading data...")
    claims_df, members_df = prepare_data_for_modeling()
    cutoff_date = claims_df['ServiceDate'].max() - timedelta(days=180)
    print(f"Using cutoff date: {cutoff_date}")
    
    # Step 2: Create baseline features (basic features only)
    print("\n2. Creating baseline features...")
    baseline_features = prepare_features_for_modeling(claims_df, members_df, cutoff_date)
    
    # Prepare data for modeling
    X_baseline = baseline_features.drop(['Member_ID', 'PolicyID', 'future_6m_claims'], axis=1, errors='ignore')
    # Keep only numeric columns
    X_baseline = X_baseline.select_dtypes(include=['int', 'float'])
    y = baseline_features['future_6m_claims']
    print(f"Baseline features: {X_baseline.shape[1]} features")
    
    # Step 3: Create enhanced features with different techniques
    print("\n3. Creating enhanced features with various techniques...")
    
    # Enhanced date features
    claims_date = claims_df.copy()
    claims_date['ServiceDate'] = pd.to_datetime(claims_date['ServiceDate'])
    date_features = create_date_features(claims_date, 'ServiceDate')
    
    # Add cyclical encoding
    cyclical_features = create_cyclical_features(date_features, 'ServiceDate_month', 12)
    cyclical_features = create_cyclical_features(cyclical_features, 'ServiceDate_dayofweek', 7)
    print(f"Date features: {cyclical_features.shape[1] - claims_date.shape[1]} new features")
    
    # Customer behavior features
    behavior_features = create_customer_behavior_features(
        claims_df,
        member_id_col='Member_ID',
        date_col='ServiceDate', 
        amount_col='TotPaymentUSD'
    )
    print(f"Customer behavior features: {behavior_features.shape[1]} features")
    
    # Enhanced data preparation with outlier detection
    claims_clean, outlier_info_claims = detect_and_handle_outliers(
        claims_df,
        columns=['TotPaymentUSD'],
        method='iqr',
        visualization=False
    )
    print(f"Outliers detected and handled in TotPaymentUSD: {outlier_info_claims.get('TotPaymentUSD', {}).get('count', 0)} outliers")
    
    members_clean, outlier_info_members = detect_and_handle_outliers(
        members_df,
        columns=['BMI', 'RiskScore'],
        method='iqr',
        visualization=False
    )
    print(f"Outliers detected and handled in BMI: {outlier_info_members.get('BMI', {}).get('count', 0)} outliers")
    print(f"Outliers detected and handled in RiskScore: {outlier_info_members.get('RiskScore', {}).get('count', 0)} outliers")
    
    # Create enhanced features with the cleaned data
    enhanced_features = prepare_features_for_modeling(claims_clean, members_clean, cutoff_date)
    
    # Merge customer behavior features
    behavior_features['Member_ID'] = behavior_features['Member_ID'].astype(str)
    enhanced_features['Member_ID'] = enhanced_features['Member_ID'].astype(str)
    enhanced_features = pd.merge(enhanced_features, behavior_features, on='Member_ID', how='left')
    
    # Prepare enhanced data for modeling
    X_enhanced = enhanced_features.drop(['Member_ID', 'PolicyID', 'future_6m_claims'], axis=1, errors='ignore')
    # Keep only numeric columns
    X_enhanced = X_enhanced.select_dtypes(include=['int', 'float'])
    print(f"Enhanced features: {X_enhanced.shape[1]} features")
    
    # Fill NaN values
    X_baseline = X_baseline.fillna(0)
    X_enhanced = X_enhanced.fillna(0)
    
    # Step 4: Compare model performance
    print("\n4. Comparing model performance...")
    
    # Split data into train/test sets
    X_train_baseline, X_test_baseline, y_train, y_test = train_test_split(
        X_baseline, y, test_size=0.2, random_state=42
    )
    
    X_train_enhanced, X_test_enhanced, _, _ = train_test_split(
        X_enhanced, y, test_size=0.2, random_state=42
    )
    
    # Train baseline model
    print("\nTraining baseline model...")
    baseline_model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    baseline_model.fit(X_train_baseline, y_train)
    
    # Train enhanced model
    print("\nTraining enhanced model...")
    enhanced_model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    enhanced_model.fit(X_train_enhanced, y_train)
    
    # Evaluate models
    y_pred_baseline = baseline_model.predict(X_test_baseline)
    y_pred_enhanced = enhanced_model.predict(X_test_enhanced)
    
    # Calculate metrics
    baseline_rmse = np.sqrt(mean_squared_error(y_test, y_pred_baseline))
    baseline_mae = mean_absolute_error(y_test, y_pred_baseline)
    baseline_r2 = r2_score(y_test, y_pred_baseline)
    
    enhanced_rmse = np.sqrt(mean_squared_error(y_test, y_pred_enhanced))
    enhanced_mae = mean_absolute_error(y_test, y_pred_enhanced)
    enhanced_r2 = r2_score(y_test, y_pred_enhanced)
    
    # Step 5: Analyze feature importance
    print("\n5. Analyzing feature importance...")
    
    # Baseline feature importance
    baseline_importance = pd.DataFrame({
        'feature': X_train_baseline.columns,
        'importance': baseline_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Enhanced feature importance
    enhanced_importance = pd.DataFrame({
        'feature': X_train_enhanced.columns,
        'importance': enhanced_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Find customer behavior features in top features
    top_n = 20
    top_enhanced_features = enhanced_importance.head(top_n)['feature'].tolist()
    customer_behavior_cols = [col for col in behavior_features.columns if col != 'Member_ID']
    important_customer_behavior = [col for col in top_enhanced_features if col in customer_behavior_cols]
    
    # Step 6: Compare and visualize results
    print("\n6. Comparing results...")
    
    # Print metrics comparison
    print("\nModel Performance Comparison:")
    print(f"{'Metric':<10} {'Baseline':<15} {'Enhanced':<15} {'Improvement':<15}")
    print(f"{'-'*50}")
    print(f"{'RMSE':<10} {baseline_rmse:<15.4f} {enhanced_rmse:<15.4f} {((baseline_rmse - enhanced_rmse) / baseline_rmse * 100):<15.2f}%")
    print(f"{'MAE':<10} {baseline_mae:<15.4f} {enhanced_mae:<15.4f} {((baseline_mae - enhanced_mae) / baseline_mae * 100):<15.2f}%")
    print(f"{'R²':<10} {baseline_r2:<15.4f} {enhanced_r2:<15.4f} {((enhanced_r2 - baseline_r2) / abs(baseline_r2) * 100):<15.2f}%")
    
    # Create visualizations
    plt.figure(figsize=(10, 6))
    metrics = ['RMSE', 'MAE', 'R²']
    baseline_values = [baseline_rmse, baseline_mae, baseline_r2]
    enhanced_values = [enhanced_rmse, enhanced_mae, enhanced_r2]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, baseline_values, width, label='Baseline')
    plt.bar(x + width/2, enhanced_values, width, label='Enhanced')
    plt.xticks(x, metrics)
    plt.ylabel('Value')
    plt.title('Model Performance Comparison')
    plt.legend()
    
    # Save visualization
    plt.savefig('visualizations/feature_evaluation/model_comparison.png')
    plt.close()
    
    # Visualize top features from both models
    plt.figure(figsize=(12, 8))
    baseline_top = baseline_importance.head(10)
    plt.subplot(1, 2, 1)
    plt.barh(np.arange(len(baseline_top)), baseline_top['importance'], align='center')
    plt.yticks(np.arange(len(baseline_top)), baseline_top['feature'])
    plt.title('Top 10 Baseline Features')
    
    enhanced_top = enhanced_importance.head(10)
    plt.subplot(1, 2, 2)
    plt.barh(np.arange(len(enhanced_top)), enhanced_top['importance'], align='center')
    plt.yticks(np.arange(len(enhanced_top)), enhanced_top['feature'])
    plt.title('Top 10 Enhanced Features')
    
    plt.tight_layout()
    plt.savefig('visualizations/feature_evaluation/top_features_comparison.png')
    plt.close()
    
    # Conclusions
    print("\nKey Findings:")
    
    if enhanced_rmse < baseline_rmse:
        print("✅ Enhanced features improved prediction accuracy (lower RMSE)")
    else:
        print("❌ Enhanced features did not improve prediction accuracy (RMSE)")
    
    if enhanced_r2 > baseline_r2:
        print("✅ Enhanced features improved explained variance (higher R²)")
    else:
        print("❌ Enhanced features did not improve explained variance (R²)")
    
    if len(important_customer_behavior) > 0:
        print(f"✅ {len(important_customer_behavior)} customer behavior features are in the top {top_n} important features:")
        for feat in important_customer_behavior:
            print(f"   - {feat}")
    else:
        print(f"❌ No customer behavior features found in top {top_n} important features")
    
    # Effectiveness of different techniques
    print("\nEffectiveness of Feature Engineering Techniques:")
    
    # Check if date features are important
    date_cols = [col for col in top_enhanced_features if any(x in col for x in ['_day', '_month', '_year', '_sin', '_cos'])]
    if len(date_cols) > 0:
        print("✅ Date-based features are effective")
    else:
        print("❌ Date-based features are not effective")
    
    # Check if outlier handling improved performance
    if enhanced_rmse < baseline_rmse and (outlier_info_claims or outlier_info_members):
        print("✅ Outlier detection and handling is effective")
    else:
        print("❌ Outlier detection and handling is not effective or inconclusive")
    
    # Final summary
    print("\nOverall Assessment:")
    if enhanced_rmse < baseline_rmse and enhanced_r2 > baseline_r2:
        print("The advanced feature engineering techniques significantly improve model performance.")
    elif enhanced_rmse < baseline_rmse or enhanced_r2 > baseline_r2:
        print("The advanced feature engineering techniques show some improvement, but results are mixed.")
    else:
        print("The advanced feature engineering techniques do not appear to improve model performance.")
    
    return {
        'baseline_metrics': {
            'rmse': baseline_rmse,
            'mae': baseline_mae,
            'r2': baseline_r2
        },
        'enhanced_metrics': {
            'rmse': enhanced_rmse,
            'mae': enhanced_mae,
            'r2': enhanced_r2
        },
        'baseline_importance': baseline_importance,
        'enhanced_importance': enhanced_importance,
        'important_customer_behavior': important_customer_behavior
    }

if __name__ == "__main__":
    start_time = time.time()
    results = evaluate_feature_effectiveness()
    runtime = time.time() - start_time
    print(f"\nEvaluation completed in {runtime:.2f} seconds") 