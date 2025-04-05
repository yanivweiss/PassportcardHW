"""
Test loading and using the saved XGBoost model for predictions
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os

def test_model_predictions():
    """Test loading and using the saved model for predictions"""
    print("=" * 80)
    print("TESTING MODEL PREDICTIONS")
    print("=" * 80)
    
    # Create output directory for visualizations
    os.makedirs('visualizations/predictions', exist_ok=True)
    
    # Step 1: Load the saved model
    print("\n1. Loading saved model...")
    try:
        model_info = joblib.load('best_xgboost_model.pkl')
        model = model_info['model']
        feature_cols = model_info.get('feature_cols', [])
        print(f"Model loaded successfully with {len(feature_cols)} features")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Step 2: Load the integrated features
    print("\n2. Loading features data...")
    try:
        features_df = pd.read_csv('integrated_features.csv')
        print(f"Loaded features with shape: {features_df.shape}")
    except Exception as e:
        print(f"Error loading features: {e}")
        return
    
    # Step 3: Prepare data for prediction
    print("\n3. Preparing data for prediction...")
    
    # Create a new dataframe with the exact features needed by the model
    X = pd.DataFrame(index=features_df.index)
    
    # For each feature needed by the model, either get it from the dataframe or create a dummy column
    missing_features = []
    for feature in feature_cols:
        if feature in features_df.columns:
            # If the feature exists in our data, use it
            X[feature] = features_df[feature]
            # Convert object types to numeric if possible
            if X[feature].dtype == 'object':
                try:
                    X[feature] = pd.to_numeric(X[feature], errors='coerce')
                except:
                    # If conversion fails, set to 0
                    X[feature] = 0
        else:
            # If the feature doesn't exist, create a dummy column of zeros
            X[feature] = 0
            missing_features.append(feature)
    
    # Fill any missing values with 0
    X = X.fillna(0)
    
    # Get the actual target values for comparison
    y_true = features_df['future_6m_claims']
    
    print(f"Data prepared with {len(feature_cols)} features")
    if missing_features:
        print(f"Created dummy values for {len(missing_features)} missing features")
    
    # Step 4: Make predictions
    print("\n4. Making predictions...")
    y_pred = model.predict(X)
    print(f"Made predictions for {len(y_pred)} instances")
    
    # Step 5: Calculate metrics
    print("\n5. Calculating metrics...")
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100  # Adding small constant to avoid division by zero
    
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
    plt.savefig('visualizations/predictions/actual_vs_predicted_test.png')
    
    # Step 7: Print statistics about predictions
    print("\n7. Prediction statistics:")
    print(f"  Min predicted value: ${np.min(y_pred):.2f}")
    print(f"  Max predicted value: ${np.max(y_pred):.2f}")
    print(f"  Average predicted value: ${np.mean(y_pred):.2f}")
    print(f"  Median predicted value: ${np.median(y_pred):.2f}")
    
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