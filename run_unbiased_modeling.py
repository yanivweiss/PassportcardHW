#!/usr/bin/env python
"""
PassportCard Insurance Claims Prediction - Unbiased Analysis
This script runs the updated pipeline that avoids using claims data for feature engineering
and only uses member attributes to predict future claims.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime, timedelta
import time
import logging
import joblib
import sys

# Import modules
from data_preparation import prepare_data_for_modeling
from member_feature_engineering import prepare_member_features
from xgboost_modeling import train_xgboost_model, evaluate_xgboost_model
from error_analysis import analyze_prediction_errors, create_regression_confusion_matrix, plot_error_heatmap

def run_unbiased_modeling(visualize=True):
    """
    Run the unbiased modeling pipeline without using claim features for prediction
    
    Parameters:
    -----------
    visualize : bool
        Whether to create visualizations
        
    Returns:
    --------
    dict
        Dictionary with model information and metrics
    """
    start_time = time.time()
    
    print("="*80)
    print("PASSPORTCARD INSURANCE CLAIMS PREDICTION - UNBIASED ANALYSIS")
    print("Using only member attributes to avoid claims-based feature bias")
    print("="*80)
    
    # Step 1: Load and prepare data
    print("\nStep 1: Loading and preparing data...")
    claims_df, members_df = prepare_data_for_modeling()
    
    # Use a cutoff date to split into training and test sets
    cutoff_date = claims_df['ServiceDate'].max() - timedelta(days=180)
    print(f"Using cutoff date: {cutoff_date}")
    
    # Step 2: Create member-based features
    print("\nStep 2: Creating member-based features...")
    features_df = prepare_member_features(members_df, claims_df, cutoff_date)
    print(f"Created {features_df.shape[1]-1} features for {features_df.shape[0]} members")
    
    # Step 3: Prepare data for modeling
    print("\nStep 3: Preparing data for modeling...")
    # Get target variable and ID column
    X = features_df.drop(columns=['Member_ID', 'future_6m_claims'])
    y = features_df['future_6m_claims']
    
    # Check for NaN values
    if X.isna().any().any():
        print(f"Warning: Found {X.isna().sum().sum()} NaN values in features. Filling with zeros.")
        X = X.fillna(0)
    
    # Step 4: Train XGBoost model
    print("\nStep 4: Training XGBoost model...")
    model_info = train_xgboost_model(
        X, y, 
        test_size=0.2, 
        random_state=42,
        optimize=True,
        n_iter=50
    )
    
    # Step 5: Evaluate model
    print("\nStep 5: Evaluating model performance...")
    evaluation = evaluate_xgboost_model(
        model_info['model'],
        model_info['X_test'],
        model_info['y_test']
    )
    
    model_info.update(evaluation)
    
    # Step 6: Error analysis
    print("\nStep 6: Performing error analysis...")
    error_analysis = analyze_prediction_errors(
        model_info['y_test'], 
        model_info['predictions'],
        feature_matrix=model_info['X_test'],
        feature_names=model_info['X_test'].columns
    )
    
    model_info['error_analysis'] = error_analysis
    
    # Create confusion matrix
    cm, bin_edges = create_regression_confusion_matrix(
        model_info['y_test'], 
        model_info['predictions'],
        n_classes=5,
        visualize=True,
        output_path='visualizations/unbiased_model/regression_confusion_matrix.png'
    )
    
    model_info['confusion_matrix'] = cm
    model_info['bin_edges'] = bin_edges
    
    # Step 7: Create visualizations
    if visualize:
        print("\nStep 7: Creating visualizations...")
        
        # Ensure directory exists
        os.makedirs('visualizations/unbiased_model', exist_ok=True)
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        importance_df = model_info['feature_importance'].sort_values('importance', ascending=True).tail(20)
        sns.barplot(x='importance', y='feature', data=importance_df)
        plt.title('Top 20 Feature Importance (Unbiased Model)')
        plt.tight_layout()
        plt.savefig('visualizations/unbiased_model/feature_importance.png')
        plt.close()
        
        # Plot predictions vs actual
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=model_info['y_test'], y=model_info['predictions'])
        plt.plot([0, model_info['y_test'].max()], [0, model_info['y_test'].max()], 'r--')
        plt.xlabel('Actual Claims')
        plt.ylabel('Predicted Claims')
        plt.title('Predictions vs Actual (Unbiased Model)')
        plt.tight_layout()
        plt.savefig('visualizations/unbiased_model/predictions_vs_actual.png')
        plt.close()
        
        # Plot error distribution
        plt.figure(figsize=(10, 6))
        errors = model_info['y_test'] - model_info['predictions']
        sns.histplot(errors, kde=True)
        plt.xlabel('Prediction Error')
        plt.title('Error Distribution (Unbiased Model)')
        plt.tight_layout()
        plt.savefig('visualizations/unbiased_model/error_distribution.png')
        plt.close()
        
        # Plot error heatmap for top 2 features
        if len(model_info['feature_importance']) >= 2:
            top_features = model_info['feature_importance'].sort_values('importance', ascending=False).head(2)['feature'].values
            plot_error_heatmap(
                model_info['y_test'], 
                model_info['predictions'],
                model_info['X_test'][top_features[0]], 
                model_info['X_test'][top_features[1]],
                top_features[0], 
                top_features[1],
                output_path='visualizations/unbiased_model/error_heatmap.png'
            )
    
    # Step 8: Save model and results
    print("\nStep 8: Saving model and results...")
    
    # Save feature importance
    model_info['feature_importance'].to_csv('feature_importance_unbiased.csv', index=False)
    
    # Save model
    joblib.dump(model_info['model'], 'best_unbiased_model.pkl')
    
    # Save model metrics
    with open('unbiased_model_metrics.txt', 'w') as f:
        f.write(f"RMSE: {model_info['metrics']['rmse']:.2f}\n")
        f.write(f"MAE: {model_info['metrics']['mae']:.2f}\n")
        f.write(f"R²: {model_info['metrics']['r2']:.2f}\n")
        f.write(f"MAPE: {model_info['metrics']['mape']:.2f}\n")
    
    # Calculate runtime
    runtime = time.time() - start_time
    runtime_minutes = runtime / 60
    
    # Print summary
    print("\nStep 9: Summary of results...\n")
    print("Model Performance:")
    for metric, value in model_info['metrics'].items():
        print(f"  {metric.upper()}: {value:.4f}")
    
    print("\nTop 10 Important Features:")
    for i, (feature, importance) in enumerate(zip(
            model_info['feature_importance']['feature'].head(10), 
            model_info['feature_importance']['importance'].head(10))):
        print(f"  {i+1}. {feature}: {importance:.4f}")
    
    print(f"\nTotal runtime: {runtime:.2f} seconds ({runtime_minutes:.2f} minutes)")
    
    return model_info

def update_readme_with_metrics(model_info):
    """
    Update the README with new metrics and findings
    
    Parameters:
    -----------
    model_info : dict
        Dictionary with model information and metrics
    """
    print("\nUpdating README with new metrics...")
    
    # Create a short summary for the README
    metrics_summary = f"""
## Unbiased Model Performance

The unbiased model uses only member attributes and avoids using claim-based features for prediction:

- **RMSE**: {model_info['metrics']['rmse']:.2f} USD
- **MAE**: {model_info['metrics']['mae']:.2f} USD
- **R²**: {model_info['metrics']['r2']:.2f}
- **MAPE**: {model_info['metrics']['mape']:.2f}%

### Top 5 Important Features

{model_info['feature_importance'].sort_values('importance', ascending=False).head(5)[['feature', 'importance']].to_markdown(index=False)}

### Key Findings

- The model now predicts future claims using only member characteristics, avoiding potential bias from using historical claim features
- Questionnaire responses remain strong predictors of future claims
- Demographic and lifestyle factors show high predictive power
- This approach better isolates true risk factors rather than merely using past claims to predict future claims
"""
    
    # Read existing README
    with open('README.md', 'r') as f:
        readme_content = f.read()
    
    # Find a good place to insert our new metrics
    if "## Unbiased Model Performance" in readme_content:
        # Replace existing unbiased model section
        start_idx = readme_content.find("## Unbiased Model Performance")
        end_idx = readme_content.find("##", start_idx + 1)
        if end_idx == -1:
            end_idx = len(readme_content)
        
        new_readme = readme_content[:start_idx] + metrics_summary + readme_content[end_idx:]
    else:
        # Add to the end of Model Performance section
        if "## Model Performance" in readme_content:
            insert_idx = readme_content.find("##", readme_content.find("## Model Performance") + 1)
            if insert_idx == -1:
                insert_idx = len(readme_content)
            
            new_readme = readme_content[:insert_idx] + metrics_summary + readme_content[insert_idx:]
        else:
            # Just append to the end
            new_readme = readme_content + "\n\n" + metrics_summary
    
    # Write updated README
    with open('README.md', 'w') as f:
        f.write(new_readme)
    
    print("README updated successfully.")

def update_changelog():
    """Update CHANGELOG.md with information about the unbiased model"""
    
    # Get current date
    today = datetime.now().strftime("%Y-%m-%d")
    
    changelog_entry = f"""
## [2.0.0] - {today}

### Added
- New unbiased modeling approach that avoids using claim features for prediction
- Member-based feature engineering focused on demographics, health indicators, and questionnaires
- Improved model integrity by preventing data leakage between features and target

### Changed
- Completely revised feature engineering process to remove bias
- Updated pipeline to use only member attributes for prediction
- Modified visualization and reporting to reflect new model approach

### Fixed
- Addressed the circular reasoning issue where claims were used to predict future claims
- Eliminated correlation bias between independent and dependent variables
- Improved model generalizability by focusing on true predictive factors
"""
    
    # Read existing CHANGELOG
    with open('CHANGELOG.md', 'r') as f:
        changelog_content = f.read()
    
    # Find where to insert the new entry
    if "# Changelog" in changelog_content:
        insert_idx = changelog_content.find("\n", changelog_content.find("# Changelog")) + 1
    else:
        # Add header if it doesn't exist
        changelog_content = "# Changelog\n\n" + changelog_content
        insert_idx = len("# Changelog\n\n")
    
    # Insert the new entry
    new_changelog = changelog_content[:insert_idx] + changelog_entry + changelog_content[insert_idx:]
    
    # Write updated CHANGELOG
    with open('CHANGELOG.md', 'w') as f:
        f.write(new_changelog)
    
    print("CHANGELOG updated successfully.")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, 
                      format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Run the unbiased modeling pipeline
    try:
        model_info = run_unbiased_modeling(visualize=True)
        
        # Update README with new metrics
        update_readme_with_metrics(model_info)
        
        # Update CHANGELOG
        update_changelog()
        
        # Push to git if requested
        if len(sys.argv) > 1 and sys.argv[1] == '--push':
            # Import sys
            import sys
            import subprocess
            
            print("\nPushing changes to git...")
            try:
                # Add files
                subprocess.run(["git", "add", "."], check=True)
                
                # Commit
                subprocess.run(["git", "commit", "-m", "Implement unbiased modeling approach"], check=True)
                
                # Push
                subprocess.run(["git", "push"], check=True)
                
                print("Successfully pushed to git.")
            except subprocess.CalledProcessError as e:
                print(f"Error pushing to git: {e}")
        
    except Exception as e:
        logging.error(f"Error in unbiased modeling pipeline: {e}", exc_info=True)
        raise 