import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime, timedelta

# Import our enhanced modules
from data_preparation import prepare_data_for_modeling
from feature_engineering import prepare_features_for_modeling
from enhanced_features import enhance_features
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

def main():
    """Run the full advanced analysis pipeline with all enhancements"""
    print("Starting enhanced analysis with advanced temporal features, risk scores, and XGBoost modeling...")
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    print("\nStep 1: Loading and preparing data...")
    claims_df, members_df = prepare_data_for_modeling()
    
    # Use a cutoff date 6 months before the last date
    cutoff_date = claims_df['ServiceDate'].max() - timedelta(days=180)
    print(f"Using cutoff date: {cutoff_date.strftime('%Y-%m-%d')}")
    
    print("\nStep 2: Integrating all feature engineering approaches...")
    integrated_features, risk_scores = integrate_all_features(claims_df, members_df, cutoff_date)
    
    print("\nStep 3: Running XGBoost modeling pipeline...")
    # Run the XGBoost pipeline with hyperparameter optimization
    xgb_results = run_complete_xgboost_pipeline(
        integrated_features,
        target_col='future_6m_claims',
        optimize=True,
        n_iter=20,
        save_model=True
    )
    
    print("\nStep 4: Generating advanced business report...")
    # Create advanced business report with visualizations
    business_analysis = run_complete_business_analysis(xgb_results, risk_scores)
    
    print("\nStep 5: Summarizing results...")
    # Extract metrics for summary
    metrics = xgb_results.get('metrics', {})
    
    print("\nModel Performance:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nTop 10 Features:")
    top_features = xgb_results.get('feature_importance', pd.DataFrame()).head(10)
    for i, (feature, importance) in enumerate(zip(top_features['feature'], top_features['importance'])):
        print(f"  {i+1}. {feature}: {importance:.4f}")
    
    print("\nOutputs:")
    print(f"  Integrated Features: integrated_features.csv ({len(integrated_features)} rows, {len(integrated_features.columns)} columns)")
    print(f"  Risk Scores: risk_scores.csv ({len(risk_scores)} rows, {len(risk_scores.columns)} columns)")
    print(f"  Model: best_xgboost_model.pkl")
    print(f"  Business Report: {business_analysis['report_path']}")
    print(f"  Visualizations: {business_analysis['visualizations_path']}")
    
    print("\nAnalysis complete! All results have been saved.")

if __name__ == "__main__":
    main() 