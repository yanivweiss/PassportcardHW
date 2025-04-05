import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

from data_preparation import prepare_data_for_modeling
from feature_engineering import prepare_features_for_modeling
from model_development import train_and_evaluate_model

def plot_feature_importance(feature_importance, title='Top 20 Most Important Features'):
    """Plot feature importance"""
    plt.figure(figsize=(12, 6))
    feature_importance.head(20).plot(x='feature', y='importance', kind='bar')
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

def plot_predictions_vs_actual(y_test, y_pred, title='Predicted vs Actual Claims'):
    """Plot predicted vs actual values"""
    plt.figure(figsize=(10, 10))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Claims')
    plt.ylabel('Predicted Claims')
    plt.title(title)
    plt.tight_layout()
    plt.savefig('predictions_vs_actual.png')
    plt.close()

def main():
    """Main analysis pipeline"""
    print("1. Preparing data...")
    claims_df, members_df = prepare_data_for_modeling()
    
    print("\n2. Engineering features...")
    # Use a cutoff date 6 months before the last date to create validation set
    cutoff_date = claims_df['ServiceDate'].max() - timedelta(days=180)
    features_df = prepare_features_for_modeling(claims_df, members_df, cutoff_date)
    
    print("\n3. Training and evaluating model...")
    model, metrics, feature_importance, shap_values, scaler = train_and_evaluate_model(features_df)
    
    print("\nModel Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\n4. Creating visualizations...")
    plot_feature_importance(feature_importance)
    
    # Get test predictions for visualization
    exclude_cols = ['Member_ID', 'PolicyID', 'future_6m_claims', 'PolicyStartDate', 'PolicyEndDate', 'DateOfBirth']
    feature_cols = [col for col in features_df.columns if col not in exclude_cols]
    
    X_test = features_df.sample(frac=0.2, random_state=42)
    y_test = X_test['future_6m_claims']
    X_test_scaled = scaler.transform(X_test[feature_cols])
    y_pred = model.predict(X_test_scaled)
    plot_predictions_vs_actual(y_test, y_pred)
    
    print("\n5. Saving results...")
    # Save feature importance
    feature_importance.to_csv('feature_importance.csv', index=False)
    
    # Save model metrics
    with open('model_metrics.txt', 'w') as f:
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
    
    print("\nAnalysis complete! Results have been saved to files.")

if __name__ == "__main__":
    main() 