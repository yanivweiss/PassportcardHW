import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import shap

def prepare_model_data(features_df, target_col='future_6m_claims', test_size=0.2):
    """Prepare data for modeling"""
    # Remove any non-feature columns
    exclude_cols = ['Member_ID', 'PolicyID', target_col, 'PolicyStartDate', 'PolicyEndDate', 'DateOfBirth']
    feature_cols = [col for col in features_df.columns if col not in exclude_cols]
    
    # Convert categorical columns to numeric
    categorical_cols = ['CountryOfOrigin', 'PayerType', 'CountryOfDestination', 'Sex']
    for col in categorical_cols:
        if col in features_df.columns:
            features_df[col] = pd.Categorical(features_df[col]).codes
    
    # Split features and target
    X = features_df[feature_cols]
    y = features_df[target_col]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols, scaler

def train_lightgbm_model(X_train, y_train, X_test, y_test):
    """Train a LightGBM model"""
    # Define model parameters
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }
    
    # Create dataset
    train_data = lgb.Dataset(X_train, label=y_train)
    
    # Train model
    model = lgb.train(params, train_data, num_boost_round=100)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    return model, y_pred

def evaluate_model(y_true, y_pred):
    """Evaluate model performance"""
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred)
    }
    
    return metrics

def analyze_feature_importance(model, feature_cols, X_test):
    """Analyze feature importance using SHAP values"""
    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # Create feature importance DataFrame
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': np.abs(shap_values).mean(0)
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    return feature_importance, shap_values

def train_and_evaluate_model(features_df):
    """Main function to train and evaluate the model"""
    # Prepare data
    X_train, X_test, y_train, y_test, feature_cols, scaler = prepare_model_data(features_df)
    
    # Train model
    model, y_pred = train_lightgbm_model(X_train, y_train, X_test, y_test)
    
    # Evaluate model
    metrics = evaluate_model(y_test, y_pred)
    
    # Analyze feature importance
    feature_importance, shap_values = analyze_feature_importance(model, feature_cols, X_test)
    
    return model, metrics, feature_importance, shap_values, scaler 