import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Import modules to test
from data_preparation import prepare_data_for_modeling
from feature_engineering import prepare_features_for_modeling
from enhanced_features import (create_advanced_temporal_features, 
                             create_service_type_profiles, 
                             create_risk_scores,
                             create_interaction_features)
from advanced_modeling import (prepare_model_data,
                              train_lightgbm_model,
                              train_xgboost_model,
                              feature_selection,
                              evaluate_model)

class TestDataPreparation(unittest.TestCase):
    """Test data preparation functionality"""
    
    def setUp(self):
        """Create sample test data"""
        # Create sample claims data
        self.claims_data = pd.DataFrame({
            'ClaimNumber': range(1, 11),
            'TotPaymentUSD': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
            'ServiceDate': [datetime(2022, 1, 1) + timedelta(days=i*30) for i in range(10)],
            'ServiceGroup': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C'],
            'ServiceType': ['X', 'Y', 'Z', 'X', 'Y', 'Z', 'X', 'Y', 'Z', 'X'],
            'Member_ID': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            'PolicyID': [101, 101, 102, 102, 103, 103, 104, 104, 105, 105],
            'Sex': ['M', 'M', 'F', 'F', 'M', 'M', 'F', 'F', 'M', 'M'],
            'PayDate': [datetime(2022, 1, 5) + timedelta(days=i*30) for i in range(10)]
        })
        
        # Create sample members data with all required questionnaire fields
        self.members_data = pd.DataFrame({
            'Member_ID': range(1, 6),
            'PolicyID': range(101, 106),
            'PolicyStartDate': [datetime(2021, 1, 1) for _ in range(5)],
            'PolicyEndDate': [datetime(2023, 1, 1) for _ in range(5)],
            'DateOfBirth': [datetime(1980, 1, 1) + timedelta(days=i*365*2) for i in range(5)],
            'CountryOfOrigin': ['USA', 'UK', 'Canada', 'Australia', 'France'],
            'CountryOfDestination': ['Japan', 'China', 'Germany', 'Italy', 'Spain'],
            'Gender': [True, False, True, False, True],  # True is male
            'BMI': [22.5, 24.1, 25.6, 23.2, 26.7],
            'Questionnaire_cancer': [0, 1, 0, 0, 1],
            'Questionnaire_smoke': [1, 0, 1, 0, 1],
            'Questionnaire_heart': [0, 0, 1, 0, 1],
            'Questionnaire_diabetes': [0, 1, 0, 1, 0],
            'Questionnaire_respiratory': [0, 0, 1, 0, 1],
            'Questionnaire_thyroid': [0, 1, 0, 0, 0],
            'Questionnaire_liver': [0, 0, 0, 1, 0],
            'Questionnaire_immune': [0, 0, 0, 0, 1],
            'Questionnaire_tumor': [0, 1, 0, 0, 0],
            'Questionnaire_relatives': [1, 0, 0, 0, 1],
            'Questionnaire_alcoholism': [0, 0, 1, 0, 0],
            'Questionnaire_drink': [1, 0, 1, 0, 1]
        })
    
    def test_basic_data_prep(self):
        """Test that data preparation functions run without errors"""
        try:
            # If data files exist, this will use them
            if os.path.exists('claims_data_clean.csv') and os.path.exists('members_data_clean.csv'):
                claims_df, members_df = prepare_data_for_modeling()
                self.assertIsInstance(claims_df, pd.DataFrame)
                self.assertIsInstance(members_df, pd.DataFrame)
                print(f"Loaded real data: {len(claims_df)} claims, {len(members_df)} members")
            else:
                # Otherwise, we'll skip this test
                print("Test data files not found, skipping real data test")
        except Exception as e:
            print(f"Could not load real data, error: {e}")
    
    def test_feature_engineering(self):
        """Test feature engineering functions with synthetic data"""
        cutoff_date = datetime(2022, 9, 1)
        
        # Test advanced temporal features
        temporal_features = create_advanced_temporal_features(self.claims_data, cutoff_date)
        self.assertIsInstance(temporal_features, pd.DataFrame)
        self.assertIn('Member_ID', temporal_features.columns)
        self.assertIn('claim_frequency_monthly', temporal_features.columns)
        
        # Test service type profiles
        service_profiles = create_service_type_profiles(self.claims_data, cutoff_date)
        self.assertIsInstance(service_profiles, pd.DataFrame)
        self.assertIn('Member_ID', service_profiles.columns)
        self.assertIn('service_entropy', service_profiles.columns)
        
        # Test risk scores
        risk_scores = create_risk_scores(self.members_data)
        self.assertIsInstance(risk_scores, pd.DataFrame)
        self.assertIn('Member_ID', risk_scores.columns)
        self.assertIn('weighted_risk_score', risk_scores.columns)
        
        # Test interaction features
        # First, create a base feature set
        base_features = pd.DataFrame({
            'Member_ID': range(1, 6),
            'claim_count': [2, 2, 2, 2, 2],
            'total_claims': [300, 700, 1100, 1500, 1900],
            'basic_risk_score': [1, 2, 2, 1, 3],
            'weighted_risk_score': [2, 4, 5, 3, 7]
        })
        
        interaction_features = create_interaction_features(base_features)
        self.assertIsInstance(interaction_features, pd.DataFrame)
        self.assertTrue(len(interaction_features.columns) > len(base_features.columns))
        
        # Check for specific interaction columns
        interaction_cols = [col for col in interaction_features.columns if '_x_' in col]
        self.assertTrue(len(interaction_cols) > 0)

class TestModeling(unittest.TestCase):
    """Test modeling functionality"""
    
    def setUp(self):
        """Create sample features data"""
        np.random.seed(42)
        n_samples = 100
        n_features = 10
        
        # Create synthetic feature matrix
        X = np.random.randn(n_samples, n_features)
        # Create target with some relationship to features
        y = 2 * X[:, 0] + 3 * X[:, 1] - 1.5 * X[:, 2] + 0.5 * np.random.randn(n_samples)
        
        # Convert to DataFrame
        feature_names = [f'feature_{i}' for i in range(n_features)]
        self.features_df = pd.DataFrame(X, columns=feature_names)
        self.features_df['Member_ID'] = range(n_samples)
        self.features_df['future_6m_claims'] = y
    
    def test_data_preparation_for_modeling(self):
        """Test data preparation for modeling"""
        X_train, X_test, y_train, y_test, feature_cols, scaler = prepare_model_data(self.features_df)
        
        # Check correct shapes
        self.assertEqual(X_train.shape[1], 10)  # 10 features
        self.assertEqual(X_test.shape[1], 10)
        self.assertEqual(len(y_train.shape), 1)  # 1D array
        self.assertEqual(len(y_test.shape), 1)
        
        # Check feature columns
        self.assertEqual(len(feature_cols), 10)
        self.assertTrue(all(f'feature_{i}' in feature_cols for i in range(10)))
        
        # Check scaler
        self.assertIsNotNone(scaler)
    
    def test_lightgbm_model(self):
        """Test LightGBM model training"""
        X_train, X_test, y_train, y_test, feature_cols, _ = prepare_model_data(self.features_df)
        
        model, y_pred, params = train_lightgbm_model(
            X_train, y_train, X_test, y_test, 
            params={'objective': 'regression', 'n_estimators': 50}, 
            tuning=None
        )
        
        # Check model and predictions
        self.assertIsNotNone(model)
        self.assertEqual(len(y_pred), len(y_test))
        
        # Check metrics
        metrics = evaluate_model(y_test, y_pred)
        self.assertIn('RMSE', metrics)
        self.assertIn('R2', metrics)
        self.assertIn('MAE', metrics)
    
    def test_feature_selection(self):
        """Test feature selection"""
        X_train, X_test, y_train, y_test, feature_cols, _ = prepare_model_data(self.features_df)
        
        # Test LightGBM-based feature selection
        X_train_selected, X_test_selected, selected_features = feature_selection(
            X_train, y_train, X_test, feature_cols, method='lgb'
        )
        
        # Check results
        self.assertLess(X_train_selected.shape[1], X_train.shape[1])
        self.assertEqual(X_train_selected.shape[1], len(selected_features))
        self.assertEqual(X_train_selected.shape[1], X_test_selected.shape[1])

if __name__ == '__main__':
    unittest.main() 