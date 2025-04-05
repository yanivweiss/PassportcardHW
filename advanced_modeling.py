import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.neural_network import MLPRegressor
import joblib
from datetime import datetime

# Check if shap is available (optional dependency)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP is not available. Feature importance will use built-in methods instead.")

def prepare_model_data(features_df, target_col='future_6m_claims', test_size=0.2):
    """Prepare data for modeling with optional feature selection"""
    # Make a copy to avoid modifying the original
    features_df = features_df.copy()
    
    # Remove any non-feature columns
    exclude_cols = ['Member_ID', 'PolicyID', target_col, 'PolicyStartDate', 'PolicyEndDate', 'DateOfBirth']
    exclude_cols = [col for col in exclude_cols if col in features_df.columns]
    
    # Identify and remove datetime columns
    datetime_cols = []
    for col in features_df.columns:
        if pd.api.types.is_datetime64_any_dtype(features_df[col]):
            if col not in exclude_cols:
                datetime_cols.append(col)
                exclude_cols.append(col)
    
    if datetime_cols:
        print(f"Removed datetime columns from feature set: {datetime_cols}")
    
    feature_cols = [col for col in features_df.columns if col not in exclude_cols]
    
    # Replace infinity values with NaN
    features_df = features_df.replace([np.inf, -np.inf], np.nan)
    
    # Check for NaN values and fill them
    nan_cols = features_df[feature_cols].columns[features_df[feature_cols].isna().any()].tolist()
    if nan_cols:
        print(f"Filling NaN values in {len(nan_cols)} columns")
        # Fill NaN with 0 for simplicity
        features_df[feature_cols] = features_df[feature_cols].fillna(0)
    
    # Convert categorical columns to numeric
    categorical_cols = ['CountryOfOrigin', 'PayerType', 'CountryOfDestination', 'Sex']
    categorical_cols = [col for col in categorical_cols if col in features_df.columns]
    for col in categorical_cols:
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

def train_lightgbm_model(X_train, y_train, X_test, y_test, params=None, tuning='basic'):
    """Train a LightGBM model with hyperparameter tuning"""
    if params is None:
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbosity': -1
        }
    
    if tuning == 'basic':
        # Basic hyperparameter tuning
        param_grid = {
            'num_leaves': [15, 31, 63],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [100, 200, 300]
        }
        
        # Create model
        lgb_model = lgb.LGBMRegressor(**params)
        
        # Grid search
        grid_search = GridSearchCV(
            lgb_model,
            param_grid,
            cv=3,
            scoring='neg_root_mean_squared_error',
            verbose=0
        )
        
        # Train model
        grid_search.fit(X_train, y_train)
        
        # Get best model
        model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
    elif tuning == 'advanced':
        # Advanced hyperparameter tuning
        param_grid = {
            'num_leaves': [15, 31, 63, 127],
            'learning_rate': [0.01, 0.03, 0.05, 0.1],
            'n_estimators': [100, 200, 300, 400],
            'max_depth': [-1, 5, 10, 15],
            'min_child_samples': [5, 10, 20, 30],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0]
        }
        
        # Create model
        lgb_model = lgb.LGBMRegressor(**params)
        
        # Random search (more efficient for large param space)
        random_search = RandomizedSearchCV(
            lgb_model,
            param_distributions=param_grid,
            n_iter=20,
            cv=3,
            scoring='neg_root_mean_squared_error',
            verbose=0,
            random_state=42
        )
        
        # Train model
        random_search.fit(X_train, y_train)
        
        # Get best model
        model = random_search.best_estimator_
        best_params = random_search.best_params_
        
    else:
        # No tuning
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train)
        best_params = params
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    return model, y_pred, best_params

def train_xgboost_model(X_train, y_train, X_test, y_test, params=None, tuning='basic'):
    """Train an XGBoost model with hyperparameter tuning"""
    if params is None:
        params = {
            'objective': 'reg:squarederror',
            'learning_rate': 0.05,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0
        }
    
    if tuning == 'basic':
        # Basic hyperparameter tuning
        param_grid = {
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [100, 200, 300]
        }
        
        # Create model
        xgb_model = xgb.XGBRegressor(**params)
        
        # Grid search
        grid_search = GridSearchCV(
            xgb_model,
            param_grid,
            cv=3,
            scoring='neg_root_mean_squared_error',
            verbose=0
        )
        
        # Train model
        grid_search.fit(X_train, y_train)
        
        # Get best model
        model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
    elif tuning == 'advanced':
        # Advanced hyperparameter tuning
        param_grid = {
            'max_depth': [3, 6, 9, 12],
            'learning_rate': [0.01, 0.03, 0.05, 0.1],
            'n_estimators': [100, 200, 300, 400],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'gamma': [0, 0.1, 0.2, 0.5],
            'min_child_weight': [1, 3, 5, 7]
        }
        
        # Create model
        xgb_model = xgb.XGBRegressor(**params)
        
        # Random search
        random_search = RandomizedSearchCV(
            xgb_model,
            param_distributions=param_grid,
            n_iter=20,
            cv=3,
            scoring='neg_root_mean_squared_error',
            verbose=0,
            random_state=42
        )
        
        # Train model
        random_search.fit(X_train, y_train)
        
        # Get best model
        model = random_search.best_estimator_
        best_params = random_search.best_params_
        
    else:
        # No tuning
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        best_params = params
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    return model, y_pred, best_params

def train_neural_network(X_train, y_train, X_test, y_test, params=None):
    """Train a neural network model"""
    if params is None:
        params = {
            'hidden_layer_sizes': (100, 50),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.0001,
            'learning_rate': 'adaptive',
            'max_iter': 500,
            'early_stopping': True,
            'random_state': 42
        }
    
    # Create and train model
    model = MLPRegressor(**params)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    return model, y_pred

def create_ensemble_model(X_train, y_train, X_test, y_test):
    """Create an ensemble model combining multiple algorithms"""
    # Create base models
    lgb_model = lgb.LGBMRegressor(
        objective='regression',
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31
    )
    
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6
    )
    
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )
    
    gbr_model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )
    
    # Try different ensemble strategies
    
    # 1. Voting Regressor
    voting_model = VotingRegressor([
        ('lgb', lgb_model),
        ('xgb', xgb_model),
        ('rf', rf_model),
        ('gbr', gbr_model)
    ])
    
    voting_model.fit(X_train, y_train)
    voting_pred = voting_model.predict(X_test)
    
    # 2. Stacking Regressor
    stacking_model = StackingRegressor(
        estimators=[
            ('lgb', lgb_model),
            ('xgb', xgb_model),
            ('rf', rf_model),
            ('gbr', gbr_model)
        ],
        final_estimator=Ridge(),
        cv=5
    )
    
    stacking_model.fit(X_train, y_train)
    stacking_pred = stacking_model.predict(X_test)
    
    # Evaluate both models
    voting_rmse = np.sqrt(mean_squared_error(y_test, voting_pred))
    stacking_rmse = np.sqrt(mean_squared_error(y_test, stacking_pred))
    
    print(f"Voting Ensemble RMSE: {voting_rmse:.2f}")
    print(f"Stacking Ensemble RMSE: {stacking_rmse:.2f}")
    
    # Return the better model
    if voting_rmse <= stacking_rmse:
        return voting_model, voting_pred, "Voting"
    else:
        return stacking_model, stacking_pred, "Stacking"

def feature_selection(X_train, y_train, X_test, feature_cols, method='lgb'):
    """Perform feature selection to reduce dimensionality"""
    if method == 'lgb':
        # Use LightGBM for feature selection
        model = lgb.LGBMRegressor(
            objective='regression',
            n_estimators=100,
            learning_rate=0.05,
            importance_type='gain'
        )
        model.fit(X_train, y_train)
        
        # Get feature importances
        importances = model.feature_importances_
        
        # Create DataFrame of feature importances
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Select top features (e.g., top 50%)
        top_features = feature_importance.nlargest(int(len(feature_cols) * 0.5), 'importance')['feature'].tolist()
        
        # Get indices of top features
        top_indices = [feature_cols.index(f) for f in top_features]
        
        # Filter X_train and X_test
        X_train_selected = X_train[:, top_indices]
        X_test_selected = X_test[:, top_indices]
        
        return X_train_selected, X_test_selected, top_features
    
    elif method == 'auto':
        # Automatic feature selection using SelectFromModel
        model = xgb.XGBRegressor(n_estimators=100, random_state=42)
        selector = SelectFromModel(model, threshold='median')
        
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
        
        # Get names of selected features
        mask = selector.get_support()
        top_features = [feature_cols[i] for i in range(len(feature_cols)) if mask[i]]
        
        return X_train_selected, X_test_selected, top_features
    
    else:
        # No feature selection
        return X_train, X_test, feature_cols

def evaluate_model(y_true, y_pred):
    """Evaluate model performance with multiple metrics"""
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'MAPE': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100  # Add small value to avoid division by zero
    }
    
    return metrics

def analyze_feature_importance(model, feature_cols, X_test, y_test=None, model_type='lgb'):
    """Analyze feature importance using SHAP values if available, otherwise use built-in methods"""
    if SHAP_AVAILABLE and model_type in ['lgb', 'xgb', 'rf', 'gbr', 'ensemble']:
        try:
            # Use SHAP for tree-based models
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            
            # Create feature importance DataFrame
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': np.abs(shap_values).mean(0)
            })
            feature_importance = feature_importance.sort_values('importance', ascending=False)
            
            return feature_importance, shap_values
        except Exception as e:
            print(f"SHAP analysis failed: {e}. Falling back to built-in feature importance.")
            # Fall back to built-in methods
            return get_built_in_feature_importance(model, feature_cols, X_test, y_test, model_type), None
    else:
        # Use built-in methods for feature importance
        return get_built_in_feature_importance(model, feature_cols, X_test, y_test, model_type), None

def get_built_in_feature_importance(model, feature_cols, X_test, y_test, model_type):
    """Get feature importance using built-in methods of the model"""
    if model_type in ['lgb', 'xgb', 'rf', 'gbr']:
        # For tree-based models, use feature_importances_
        importances = model.feature_importances_
    elif model_type == 'ensemble':
        # For ensemble models, get importance from the first base estimator if available
        if hasattr(model, 'estimators_'):
            try:
                importances = model.estimators_[0].feature_importances_
            except:
                # If not available, use permutation importance
                from sklearn.inspection import permutation_importance
                result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
                importances = result.importances_mean
        else:
            # Fallback to equal importance
            importances = np.ones(len(feature_cols)) / len(feature_cols)
    else:
        # For other models, use permutation importance
        from sklearn.inspection import permutation_importance
        result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
        importances = result.importances_mean
    
    # Create DataFrame of feature importances
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    return feature_importance

def train_models(X_train, X_test, y_train, y_test, feature_cols, model_types=None):
    """Train multiple models and return the best one"""
    if model_types is None:
        model_types = ['lgb', 'xgb', 'ensemble']
    
    results = {}
    
    for model_type in model_types:
        print(f"Training {model_type} model...")
        
        if model_type == 'lgb':
            model, y_pred, params = train_lightgbm_model(X_train, y_train, X_test, y_test, tuning='basic')
            
        elif model_type == 'xgb':
            model, y_pred, params = train_xgboost_model(X_train, y_train, X_test, y_test, tuning='basic')
            
        elif model_type == 'nn':
            model, y_pred = train_neural_network(X_train, y_train, X_test, y_test)
            params = None
            
        elif model_type == 'ensemble':
            model, y_pred, ensemble_type = create_ensemble_model(X_train, y_train, X_test, y_test)
            params = {'ensemble_type': ensemble_type}
            
        else:
            continue
        
        # Evaluate model
        metrics = evaluate_model(y_test, y_pred)
        
        # Get feature importance
        feature_importance, shap_values = analyze_feature_importance(
            model, feature_cols, X_test, y_test, model_type=model_type
        )
        
        # Store results
        results[model_type] = {
            'model': model,
            'predictions': y_pred,
            'metrics': metrics,
            'feature_importance': feature_importance,
            'shap_values': shap_values,
            'params': params
        }
        
        print(f"{model_type} model metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    # Find best model based on RMSE
    best_model_type = min(results, key=lambda x: results[x]['metrics']['RMSE'])
    print(f"\nBest model: {best_model_type} (RMSE: {results[best_model_type]['metrics']['RMSE']:.4f})")
    
    return results, best_model_type

def save_model(model, feature_cols, scaler, file_path='best_model.pkl'):
    """Save the trained model and associated metadata"""
    model_data = {
        'model': model,
        'feature_cols': feature_cols,
        'scaler': scaler
    }
    
    joblib.dump(model_data, file_path)
    print(f"Model saved to {file_path}")

def run_advanced_modeling(features_df, perform_feature_selection=True):
    """Run the advanced modeling pipeline"""
    # Prepare data
    X_train, X_test, y_train, y_test, feature_cols, scaler = prepare_model_data(features_df)
    
    # Feature selection (if requested)
    if perform_feature_selection and len(feature_cols) > 20:  # Only do feature selection if we have many features
        print(f"Performing feature selection (original features: {len(feature_cols)})...")
        X_train, X_test, selected_features = feature_selection(X_train, y_train, X_test, feature_cols, method='auto')
        print(f"Selected {len(selected_features)} features")
        feature_cols = selected_features
    
    # Train models
    results, best_model_type = train_models(
        X_train, X_test, y_train, y_test, feature_cols,
        model_types=['lgb', 'xgb', 'ensemble']
    )
    
    # Get best model
    best_model = results[best_model_type]['model']
    best_metrics = results[best_model_type]['metrics']
    best_feature_importance = results[best_model_type]['feature_importance']
    
    # Save the best model
    save_model(best_model, feature_cols, scaler, file_path='best_model.pkl')
    
    return best_model, best_metrics, best_feature_importance, feature_cols, scaler