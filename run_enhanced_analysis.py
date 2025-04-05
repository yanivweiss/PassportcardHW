import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os

# Import our modules
from data_preparation import prepare_data_for_modeling
from feature_engineering import prepare_features_for_modeling
from enhanced_features import enhance_features
from advanced_modeling import run_advanced_modeling, prepare_model_data

# Configure matplotlib
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 12

# Check if shap is available (optional dependency)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP is not available. SHAP-based visualizations will be skipped.")

def create_visualizations(best_feature_importance, feature_cols, model, X_test):
    """Create advanced visualizations for model interpretation"""
    # Create output directory if it doesn't exist
    os.makedirs('visualizations', exist_ok=True)
    
    # 1. Feature Importance Plot
    plt.figure(figsize=(12, 8))
    best_feature_importance.head(20).plot(x='feature', y='importance', kind='barh')
    plt.title('Top 20 Most Important Features')
    plt.tight_layout()
    plt.savefig('visualizations/feature_importance.png')
    plt.close()
    
    # 2. SHAP plots (only if available)
    if SHAP_AVAILABLE:
        try:
            plt.figure(figsize=(12, 10))
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            
            shap.summary_plot(shap_values, X_test, plot_type="bar", feature_names=feature_cols)
            plt.tight_layout()
            plt.savefig('visualizations/shap_summary.png')
            plt.close()
            
            # 3. SHAP Dependence Plots for top features
            top_features = best_feature_importance.head(5)['feature'].tolist()
            for feature in top_features:
                if feature in feature_cols:
                    plt.figure(figsize=(10, 7))
                    feature_idx = feature_cols.index(feature)
                    shap.dependence_plot(feature_idx, shap_values, X_test, feature_names=feature_cols)
                    plt.tight_layout()
                    plt.savefig(f'visualizations/dependence_plot_{feature.replace(" ", "_")}.png')
                    plt.close()
            
            # 4. SHAP Decision Plot
            plt.figure(figsize=(15, 10))
            sample_indices = np.random.choice(range(len(X_test)), size=min(50, len(X_test)), replace=False)
            shap.decision_plot(explainer.expected_value, shap_values[sample_indices], 
                             feature_names=feature_cols)
            plt.tight_layout()
            plt.savefig('visualizations/decision_plot.png')
            plt.close()
        except Exception as e:
            print(f"Could not create SHAP plots: {e}")
    
    # 5. Feature Correlation Heatmap
    if isinstance(X_test, np.ndarray):
        X_test_df = pd.DataFrame(X_test, columns=feature_cols)
    else:
        X_test_df = X_test
    
    # Select top features for correlation plot (too many make it unreadable)
    top_n_features = min(20, len(feature_cols))
    top_features = best_feature_importance['feature'].head(top_n_features).tolist()
    
    plt.figure(figsize=(14, 12))
    corr_matrix = X_test_df[top_features].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", 
              center=0, square=True, linewidths=.5)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('visualizations/correlation_heatmap.png')
    plt.close()
    
    # 6. Feature Distributions
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(top_features[:5]):  # Top 5 features
        plt.subplot(2, 3, i+1)
        sns.histplot(X_test_df[feature], kde=True)
        plt.title(f'Distribution of {feature}')
    plt.tight_layout()
    plt.savefig('visualizations/feature_distributions.png')
    plt.close()

def create_business_report(best_metrics, best_feature_importance, model_type):
    """Create a business-focused report with insights and recommendations"""
    with open('business_report.md', 'w') as f:
        f.write("# PassportCard Insurance Claims Prediction - Business Report\n\n")
        
        # Executive Summary
        f.write("## Executive Summary\n\n")
        f.write("This report presents the results of our predictive modeling for insurance claims at PassportCard. ")
        f.write("We've developed a model that predicts the total claim amount per customer for the next six months, ")
        f.write("which can help optimize financial planning, risk assessment, and premium setting.\n\n")
        
        # Model Performance
        f.write("## Model Performance\n\n")
        f.write(f"Our {model_type} model achieved the following performance metrics:\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        for metric, value in best_metrics.items():
            f.write(f"| {metric} | {value:.4f} |\n")
        f.write("\n")
        
        # Key Drivers
        f.write("## Key Drivers of Claims\n\n")
        f.write("The following factors were identified as the most significant predictors of future claims:\n\n")
        for i, (feature, importance) in enumerate(best_feature_importance.head(10).values):
            f.write(f"{i+1}. **{feature}** (Importance: {importance:.4f})\n")
        f.write("\n")
        
        # Business Recommendations
        f.write("## Business Recommendations\n\n")
        
        f.write("### Risk Assessment\n")
        f.write("- Implement the predictive model in the underwriting process to better assess customer risk profiles\n")
        f.write("- Develop a risk scoring system based on the identified key drivers\n")
        f.write("- Create a monitoring dashboard to track predictions versus actuals\n\n")
        
        f.write("### Premium Optimization\n")
        f.write("- Adjust premium calculations to account for the key risk factors identified by the model\n")
        f.write("- Consider implementing tier-based pricing based on predicted claim amounts\n")
        f.write("- Develop targeted discount programs for lower-risk customers\n\n")
        
        f.write("### Customer Segmentation\n")
        f.write("- Use model predictions to segment customers into risk categories\n")
        f.write("- Develop tailored communication strategies for different risk segments\n")
        f.write("- Create specialized customer service workflows for high-risk customers\n\n")
        
        f.write("### Financial Planning\n")
        f.write("- Use aggregate predictions to improve reserves forecasting\n")
        f.write("- Implement monthly model updates to refine financial projections\n")
        f.write("- Develop scenario planning based on model predictions\n\n")
        
        # Next Steps
        f.write("## Next Steps\n\n")
        f.write("1. Implement the model in a production environment with API access\n")
        f.write("2. Develop a user-friendly interface for underwriters and actuaries\n")
        f.write("3. Establish regular model retraining and performance monitoring\n")
        f.write("4. Expand the model to include additional data sources\n")
        f.write("5. Consider developing separate models for different claim categories\n")
        
    print("Business report created: business_report.md")

def main():
    """Run the full enhanced analysis pipeline"""
    print("1. Preparing data...")
    claims_df, members_df = prepare_data_for_modeling()
    
    print("\n2. Engineering basic features...")
    # Use a cutoff date 6 months before the last date to create validation set
    cutoff_date = claims_df['ServiceDate'].max() - timedelta(days=180)
    basic_features_df = prepare_features_for_modeling(claims_df, members_df, cutoff_date)
    
    print(f"\n3. Enhancing features (basic feature count: {len(basic_features_df.columns)})...")
    enhanced_features_df = enhance_features(basic_features_df, claims_df, members_df, cutoff_date)
    print(f"Enhanced feature count: {len(enhanced_features_df.columns)}")
    
    print("\n4. Running advanced modeling...")
    # Run the advanced modeling pipeline with feature selection
    best_model, best_metrics, best_feature_importance, feature_cols, scaler = run_advanced_modeling(
        enhanced_features_df, 
        perform_feature_selection=True
    )
    
    # Save enhanced features DataFrame to CSV
    enhanced_features_df.to_csv('enhanced_features.csv', index=False)
    print("Enhanced features saved to: enhanced_features.csv")
    
    # Create data for visualization
    print("\n5. Creating visualizations...")
    # We need to recreate test data for visualizations
    X_train, X_test, y_train, y_test, _, _ = prepare_model_data(enhanced_features_df)
    
    create_visualizations(best_feature_importance, feature_cols, best_model, X_test)
    
    print("\n6. Creating business report...")
    create_business_report(best_metrics, best_feature_importance, 
                         "Ensemble" if hasattr(best_model, "estimators_") else type(best_model).__name__)
    
    print("\nAnalysis complete! All results have been saved.")

if __name__ == "__main__":
    main() 