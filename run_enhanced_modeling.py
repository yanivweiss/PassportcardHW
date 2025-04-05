#!/usr/bin/env python
"""
PassportCard Insurance Claims Prediction - Enhanced Analysis
This script runs the complete pipeline with all enhancements including:
- Advanced temporal features
- Enhanced risk scoring
- XGBoost modeling with hyperparameter tuning
- Comprehensive business reporting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime, timedelta
import time
import logging

# Import our enhanced modules
from data_preparation import prepare_data_for_modeling
from feature_engineering import prepare_features_for_modeling
from enhanced_features import (create_advanced_temporal_features, 
                             create_service_type_profiles, 
                             create_risk_scores,
                             create_interaction_features)
from advanced_temporal_features import create_advanced_temporal_features
from enhanced_risk_scores import create_enhanced_risk_scores, create_risk_interaction_features
from xgboost_modeling import run_complete_xgboost_pipeline
from advanced_business_report import run_complete_business_analysis

# Configure matplotlib
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 12

def integrate_all_features(claims_df, members_df, cutoff_date=None):
    """
    Integrate all feature engineering approaches
    """
    if cutoff_date is None:
        cutoff_date = claims_df['ServiceDate'].max() - timedelta(days=180)
    
    print("\n1. Creating basic features...")
    basic_features_df = prepare_features_for_modeling(claims_df, members_df, cutoff_date)
    print(f"Basic feature count: {len(basic_features_df.columns)}")
    
    print("\n2. Adding enhanced features from existing module...")
    enhanced_features_df = enhance_features(basic_features_df, claims_df, members_df, cutoff_date)
    print(f"Enhanced feature count: {len(enhanced_features_df.columns)}")
    
    print("\n3. Creating advanced temporal features...")
    temporal_features_df = create_advanced_temporal_features(claims_df, cutoff_date)
    print(f"Advanced temporal feature count: {len(temporal_features_df.columns)}")
    
    print("\n4. Creating enhanced risk scores...")
    risk_scores_df = create_enhanced_risk_scores(members_df, claims_df)
    print(f"Risk scores feature count: {len(risk_scores_df.columns)}")
    
    print("\n5. Combining all features...")
    # Merge temporal features
    combined_df = pd.merge(enhanced_features_df, temporal_features_df, on='Member_ID', how='left')
    
    # Create and merge risk interactions
    risk_interactions_df = create_risk_interaction_features(combined_df, risk_scores_df)
    
    # Fill NaN values
    all_features_df = risk_interactions_df.fillna(0)
    
    print(f"Final feature count: {len(all_features_df.columns)}")
    
    # Output features to CSV
    all_features_df.to_csv('integrated_features.csv', index=False)
    print("Integrated features saved to: integrated_features.csv")
    
    # Also save risk scores separately for business analysis
    risk_scores_df.to_csv('risk_scores.csv', index=False)
    print("Risk scores saved to: risk_scores.csv")
    
    return all_features_df, risk_scores_df

def run_enhanced_analysis(use_advanced_features=True):
    """
    Run the enhanced analysis with advanced temporal features, risk scores, and XGBoost modeling
    
    Parameters:
    -----------
    use_advanced_features : bool
        Whether to use the newly added advanced features (default: True)
    """
    
    start_time = time.time()
    
    print("="*80)
    print("PASSPORTCARD INSURANCE CLAIMS PREDICTION - ENHANCED ANALYSIS")
    if not use_advanced_features:
        print("RUNNING WITHOUT ADVANCED FEATURES")
    print("="*80)
    print("Starting enhanced analysis with advanced temporal features, risk scores, and XGBoost modeling...")
    
    # Step 1: Load and prepare data
    print("\nStep 1: Loading and preparing data...")
    claims_df, members_df = prepare_data_for_modeling()
    
    # Use a cutoff date to split into training and test sets
    cutoff_date = claims_df['ServiceDate'].max() - timedelta(days=180)
    print(f"Using cutoff date: {cutoff_date}")
    
    # Step 2: Feature engineering
    print("\nStep 2: Integrating all feature engineering approaches...")
    
    # Create basic features
    print("\n1. Creating basic features...")
    basic_features = prepare_features_for_modeling(claims_df, members_df, cutoff_date)
    # Ensure Member_ID is string type for consistent merging
    basic_features['Member_ID'] = basic_features['Member_ID'].astype(str)
    print(f"Basic feature count: {basic_features.shape[1]}")
    
    # Create enhanced features from existing module
    print("\n2. Adding enhanced features from existing module...")
    
    # We'll create various types of advanced features
    print("Creating advanced temporal features...")
    temporal_features = create_advanced_temporal_features(claims_df, cutoff_date)
    
    print("Creating service type profiles...")
    service_features = create_service_type_profiles(claims_df, cutoff_date)
    
    print("Creating enhanced risk scores...")
    risk_scores = create_enhanced_risk_scores(members_df, claims_df)
    
    print("Merging feature sets...")
    # Ensure consistent data types for Member_ID
    temporal_features['Member_ID'] = temporal_features['Member_ID'].astype(str)
    service_features['Member_ID'] = service_features['Member_ID'].astype(str)
    risk_scores['Member_ID'] = risk_scores['Member_ID'].astype(str)
    
    enhanced_features = pd.merge(temporal_features, service_features, on='Member_ID', how='outer')
    enhanced_features = pd.merge(enhanced_features, risk_scores, on='Member_ID', how='outer')
    
    print("Creating interaction features...")
    enhanced_features = create_interaction_features(enhanced_features)
    print(f"Enhanced feature count: {enhanced_features.shape[1]}")
    
    # Create advanced temporal features
    print("\n3. Creating advanced temporal features...")
    adv_temporal_features = create_advanced_temporal_features(claims_df, cutoff_date)
    print(f"Advanced temporal feature count: {adv_temporal_features.shape[1]}")
    
    # Create enhanced risk scores
    print("\n4. Creating enhanced risk scores...")
    risk_scores = create_enhanced_risk_scores(members_df, claims_df)
    print(f"Risk scores feature count: {risk_scores.shape[1]}")
    
    # Only use the new advanced features if specified
    if use_advanced_features:
        try:
            from enhanced_data_preparation import enhanced_data_preparation
            from enhanced_feature_engineering import enhanced_feature_engineering
            
            # Apply enhanced data preparation
            print("\n5. Adding new advanced features...")
            
            # Enhanced data preparation
            cleaned_claims, _, _ = enhanced_data_preparation(
                claims_df, 
                missing_strategy='knn',
                outlier_method='iqr',
                scaling_method='robust',
                visualization=False
            )
            
            cleaned_members, _, _ = enhanced_data_preparation(
                members_df, 
                missing_strategy='knn',
                outlier_method='iqr',
                scaling_method='robust',
                visualization=False
            )
            
            # Enhanced feature engineering
            date_columns = ['ServiceDate', 'PolicyStartDate', 'PolicyEndDate', 'DateOfBirth']
            advanced_features = enhanced_feature_engineering(
                cleaned_claims, 
                cleaned_members, 
                date_columns=date_columns
            )
            
            print(f"New advanced feature count: {advanced_features.shape[1]}")
            
            # Combine these advanced features with our other features
            advanced_features['Member_ID'] = advanced_features['Member_ID'].astype(str)
            enhanced_features['Member_ID'] = enhanced_features['Member_ID'].astype(str)
            enhanced_features = pd.merge(enhanced_features, advanced_features, on='Member_ID', how='outer')
            
        except Exception as e:
            print(f"Warning: Could not use advanced features due to: {e}")
            print("Continuing with standard enhanced features only.")
    
    # Step 5: Combine all features
    print("\n5. Combining all features...")
    # Ensure consistent data types for all Member_ID columns
    basic_features['Member_ID'] = basic_features['Member_ID'].astype(str)
    enhanced_features['Member_ID'] = enhanced_features['Member_ID'].astype(str)
    risk_scores['Member_ID'] = risk_scores['Member_ID'].astype(str)
    
    if use_advanced_features and 'advanced_features' in locals():
        advanced_features['Member_ID'] = advanced_features['Member_ID'].astype(str)
        
    features_df = pd.merge(basic_features, enhanced_features, on='Member_ID', how='outer')
    features_df = pd.merge(features_df, risk_scores, on='Member_ID', how='left')
    
    # Save integrated features
    features_df.to_csv('integrated_features.csv', index=False)
    risk_scores.to_csv('risk_scores.csv', index=False)
    
    print(f"Final feature count: {features_df.shape[1]}")
    print(f"Integrated features saved to: integrated_features.csv")
    print(f"Risk scores saved to: risk_scores.csv")
    
    # Step 6: Run XGBoost modeling
    print("\nStep 3: Running XGBoost modeling pipeline...")
    model_info = run_complete_xgboost_pipeline(
        features_df,
        target_col='future_6m_claims',
        optimize=True,
        n_iter=50,  # More iterations for better hyperparameter tuning
        save_model=True
    )
    
    # Step 7: Create business report
    print("\nStep 4: Generating advanced business report...")
    business_analysis = run_complete_business_analysis(model_info, risk_scores)
    
    # Move the business report to reports directory - Fix file exists error
    report_source = 'advanced_business_report.md'
    report_dest = 'reports/advanced_business_report.md'
    if os.path.exists(report_source):
        # Remove destination file if it exists to avoid the error
        if os.path.exists(report_dest):
            os.remove(report_dest)
        os.rename(report_source, report_dest)
        business_analysis['report_path'] = report_dest
    else:
        business_analysis['report_path'] = 'Report file not found'
    
    print(f"Advanced business report created: {business_analysis['report_path']}")
    print(f"Business insights visualizations created in: {business_analysis['visualizations_path']}")
    
    # Calculate runtime
    runtime = time.time() - start_time
    runtime_minutes = runtime / 60
    
    # Print summary
    print("\nStep 5: Summarizing results...\n")
    print("Model Performance:")
    for metric, value in model_info['metrics'].items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nTop 10 Features:")
    for i, (feature, importance) in enumerate(zip(model_info['feature_importance']['feature'].head(10), 
                                               model_info['feature_importance']['importance'].head(10))):
        print(f"  {i+1}. {feature}: {importance:.4f}")
    
    print("\nOutputs:")
    print(f"  Integrated Features: integrated_features.csv ({features_df.shape[0]} rows, {features_df.shape[1]} columns)")
    print(f"  Risk Scores: risk_scores.csv ({risk_scores.shape[0]} rows, {risk_scores.shape[1]} columns)")
    print(f"  Model: best_xgboost_model.pkl")
    print(f"  Business Report: {business_analysis['report_path']}")
    print(f"  Visualizations: {business_analysis['visualizations_path']}")
    print(f"\nTotal runtime: {runtime:.2f} seconds ({runtime_minutes:.2f} minutes)")
    
    print("\nAnalysis complete! All results have been saved.")
    
    print("\nTo use the model for predictions, load it with:")
    print("  import joblib")
    print("  model_info = joblib.load('best_xgboost_model.pkl')")
    print("  model = model_info['model']")
    print("  feature_cols = model_info['feature_cols']")
    
    return model_info

if __name__ == "__main__":
    # Set up logging for the script
    logging.basicConfig(level=logging.INFO, 
                      format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Run with or without advanced features based on command line argument
    import sys
    use_advanced = True
    if len(sys.argv) > 1 and sys.argv[1].lower() == 'basic':
        use_advanced = False
    
    # Run the enhanced analysis
    model_info = run_enhanced_analysis(use_advanced_features=use_advanced) 