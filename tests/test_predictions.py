"""
Test loading and using the saved XGBoost model for predictions
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os
import sys
import warnings
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_model_predictions():
    """Test loading and using the saved model for predictions"""
    print("=" * 80)
    print("TESTING MODEL PREDICTIONS")
    print("=" * 80)
    
    # Create output directory for visualizations
    Path('outputs/figures/predictions').mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load the saved model
    print("\n1. Loading saved model...")
    try:
        model_info = joblib.load('models/best_xgboost_model.pkl')
        model = model_info['model']
        feature_cols = model_info.get('feature_cols', [])
        print(f"Model loaded successfully with {len(feature_cols)} features")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Step 2: Load the integrated features
    print("\n2. Loading features data...")
    try:
        features_df = pd.read_csv('data/processed/integrated_features.csv')
        print(f"Loaded features with shape: {features_df.shape}")
    except Exception as e:
        print(f"Error loading features: {e}")
        return
    
    # Step 3: Prepare data for prediction
    print("\n3. Preparing data for prediction...")
    
    # More efficient approach to creating the feature DataFrame
    # Filter for required columns that exist in features_df
    available_features = [f for f in feature_cols if f in features_df.columns]
    missing_features = [f for f in feature_cols if f not in features_df.columns]
    
    # Create DataFrame with available features
    X = features_df[available_features].copy()
    
    # Add missing features as zeros
    if missing_features:
        for feature in missing_features:
            X[feature] = 0
    
    # Convert object types to numeric and handle missing values
    for col in X.columns:
        if X[col].dtype == 'object':
            try:
                X[col] = pd.to_numeric(X[col], errors='coerce')
            except:
                X[col] = 0
    
    # Fill any remaining missing values with 0
    X = X.fillna(0)
    
    # Get the actual target values for comparison
    if 'future_6m_claims' in features_df.columns:
        y_true = features_df['future_6m_claims']
    else:
        print("Warning: 'future_6m_claims' column not found. Using zeros for y_true.")
        y_true = np.zeros(len(features_df))
    
    print(f"Data prepared with {len(feature_cols)} features")
    if missing_features:
        print(f"Created dummy values for {len(missing_features)} missing features")
    
    # Step 4: Make predictions
    print("\n4. Making predictions...")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_pred = model.predict(X)
        print(f"Made predictions for {len(y_pred)} instances")
    except Exception as e:
        print(f"Error making predictions: {e}")
        return
    
    # Step 5: Calculate metrics
    print("\n5. Calculating metrics...")
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Improved MAPE calculation to avoid division issues
    # Filter out zero or very small values in y_true to avoid extreme MAPE values
    y_true_filtered = y_true.copy()
    y_pred_filtered = y_pred.copy()
    
    # Only calculate MAPE for values above a threshold
    threshold = 10.0
    mask = y_true_filtered > threshold
    
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true_filtered[mask] - y_pred_filtered[mask]) / y_true_filtered[mask])) * 100
    else:
        mape = np.mean(np.abs(y_true_filtered - y_pred_filtered)) / (np.mean(y_true_filtered) + 1e-10) * 100
    
    print("\nPrediction Metrics:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R²: {r2:.4f}")
    print(f"  MAPE: {mape:.4f}%")
    
    # Step 6: Create visualization
    print("\n6. Creating visualization...")
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    
    # Add perfect prediction line
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.title('Actual vs Predicted Claims')
    plt.xlabel('Actual Claims (USD)')
    plt.ylabel('Predicted Claims (USD)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('outputs/figures/predictions/actual_vs_predicted_test.png')
    
    # Add residual plot
    plt.figure(figsize=(10, 6))
    residuals = y_true - y_pred
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Residual Plot')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('outputs/figures/predictions/residual_plot.png')
    
    # Step 7: Print statistics about predictions
    print("\n7. Prediction statistics:")
    print(f"  Min predicted value: ${np.min(y_pred):.2f}")
    print(f"  Max predicted value: ${np.max(y_pred):.2f}")
    print(f"  Average predicted value: ${np.mean(y_pred):.2f}")
    print(f"  Median predicted value: ${np.median(y_pred):.2f}")
    
    # Calculate negative predictions (potential issue)
    neg_preds = (y_pred < 0).sum()
    if neg_preds > 0:
        print(f"  Warning: {neg_preds} negative predictions ({neg_preds/len(y_pred)*100:.2f}% of total)")
    
    print("\nPrediction test completed successfully!")
    
    return {
        'metrics': {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        },
        'predictions': y_pred,
        'actual': y_true
    }

if __name__ == "__main__":
    results = test_model_predictions() 