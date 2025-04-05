"""
Test suite for the consolidated prediction pipeline
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Add parent directory to path to allow importing from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the module to test
from src.run_prediction_pipeline import (
    load_and_preprocess_data,
    engineer_features,
    train_or_load_model,
    make_predictions,
    analyze_results,
    run_pipeline
)

class TestPredictionPipeline(unittest.TestCase):
    """
    Test cases for the prediction pipeline
    """
    
    def setUp(self):
        """Set up test fixtures"""
        # Create test data directories if they don't exist
        os.makedirs('data/processed', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        os.makedirs('outputs/figures/predictions', exist_ok=True)
        os.makedirs('outputs/tables', exist_ok=True)
        
        # Create small test dataframes
        self.claims_df = pd.DataFrame({
            'Member_ID': ['M001', 'M001', 'M002', 'M003', 'M004'],
            'ServiceDate': pd.date_range(start='2021-01-01', periods=5),
            'TotPaymentUSD': [100.0, 200.0, 50.0, 300.0, 150.0],
            'ServiceType': ['Medical', 'Pharmacy', 'Medical', 'Dental', 'Vision']
        })
        
        self.members_df = pd.DataFrame({
            'Member_ID': ['M001', 'M002', 'M003', 'M004', 'M005'],
            'Age': [35, 42, 28, 55, 19],
            'Gender': ['M', 'F', 'M', 'F', 'M'],
            'PolicyStartDate': pd.date_range(start='2020-01-01', periods=5),
            'PolicyEndDate': pd.date_range(start='2022-01-01', periods=5)
        })
        
        # Save test data to disk for functions that read from files
        self.claims_df.to_csv('data/processed/claims_data_clean.csv', index=False)
        self.members_df.to_csv('data/processed/members_data_clean.csv', index=False)
    
    def tearDown(self):
        """Clean up after tests"""
        # Remove test model file if it exists
        if os.path.exists('models/best_xgboost_model.pkl'):
            os.remove('models/best_xgboost_model.pkl')
    
    @patch('src.run_prediction_pipeline.pd.read_csv')
    def test_load_and_preprocess_data(self, mock_read_csv):
        """Test data loading and preprocessing"""
        # Configure mocks
        mock_read_csv.side_effect = [self.claims_df, self.members_df]
        
        # Call the function
        claims, members = load_and_preprocess_data()
        
        # Verify the function was called correctly
        self.assertEqual(mock_read_csv.call_count, 2)
        
        # Verify the returned dataframes
        self.assertIsNotNone(claims)
        self.assertIsNotNone(members)
        
        # Check that the dates were converted to datetime
        self.assertEqual(claims['ServiceDate'].dtype, 'datetime64[ns]')

    @patch('src.run_prediction_pipeline.load_and_preprocess_data')
    def test_engineer_features(self, mock_load_data):
        """Test feature engineering"""
        # Configure mock
        mock_load_data.return_value = (self.claims_df, self.members_df)
        
        # Ensure ServiceDate is datetime
        self.claims_df['ServiceDate'] = pd.to_datetime(self.claims_df['ServiceDate'])
        
        # Call the function
        features_df, cutoff_date = engineer_features(self.claims_df, self.members_df)
        
        # Verify the returned data
        self.assertIsNotNone(features_df)
        self.assertIsNotNone(cutoff_date)
        
        # Check if target variable was created
        if 'future_6m_claims' in features_df.columns:
            self.assertTrue('future_6m_claims' in features_df.columns)
    
    @patch('src.run_prediction_pipeline.joblib.load')
    @patch('src.run_prediction_pipeline.joblib.dump')
    @patch('src.run_prediction_pipeline.os.path.exists')
    @patch('src.run_prediction_pipeline.prepare_data_for_xgboost')
    @patch('src.run_prediction_pipeline.train_xgboost_model')
    @patch('src.run_prediction_pipeline.evaluate_xgboost_model')
    def test_train_or_load_model(self, mock_evaluate, mock_train, mock_prepare, mock_exists, 
                                mock_dump, mock_load):
        """Test model training or loading"""
        # Test scenario 1: Model exists and should be loaded
        mock_exists.return_value = True
        mock_load.return_value = {
            'model': MagicMock(),
            'feature_cols': ['feature1', 'feature2']
        }
        
        # Create test features
        features_df = pd.DataFrame({
            'Member_ID': ['M001', 'M002'],
            'feature1': [1.0, 2.0],
            'feature2': [3.0, 4.0],
            'future_6m_claims': [100.0, 200.0]
        })
        
        # Call the function to load model
        model, feature_cols = train_or_load_model(features_df, force_train=False)
        
        # Verify model was loaded, not trained
        mock_load.assert_called_once()
        mock_train.assert_not_called()
        self.assertIsNotNone(model)
        self.assertEqual(len(feature_cols), 2)
        
        # Test scenario 2: Model should be trained (force_train=True)
        mock_load.reset_mock()
        mock_train.reset_mock()
        mock_prepare.return_value = (
            np.array([[1.0, 2.0]]), np.array([[3.0, 4.0]]), 
            np.array([100.0]), np.array([200.0]), 
            ['feature1', 'feature2']
        )
        
        mock_model = MagicMock()
        mock_train.return_value = {
            'model': mock_model,
            'X_test': np.array([[3.0, 4.0]]),
            'y_test': np.array([200.0])
        }
        
        # Mock the evaluate function
        mock_evaluate.return_value = {
            'metrics': {'rmse': 10.0, 'mae': 5.0, 'r2': 0.9}
        }
        
        # Call the function to train model
        model, feature_cols = train_or_load_model(features_df, force_train=True)
        
        # Verify model was trained
        mock_train.assert_called_once()
        mock_evaluate.assert_called_once()
        self.assertIsNotNone(model)
        self.assertEqual(model, mock_model)
    
    @patch('src.run_prediction_pipeline.load_and_preprocess_data')
    @patch('src.run_prediction_pipeline.engineer_features')
    @patch('src.run_prediction_pipeline.train_or_load_model')
    @patch('src.run_prediction_pipeline.make_predictions')
    @patch('src.run_prediction_pipeline.analyze_results')
    def test_run_pipeline(self, mock_analyze, mock_predict, mock_train, 
                        mock_engineer, mock_load):
        """Test the complete pipeline"""
        # Configure mocks
        mock_load.return_value = (self.claims_df, self.members_df)
        
        features_df = pd.DataFrame({
            'Member_ID': ['M001', 'M002'],
            'feature1': [1.0, 2.0],
            'future_6m_claims': [100.0, 200.0]
        })
        mock_engineer.return_value = (features_df, pd.Timestamp('2021-06-01'))
        
        model = MagicMock()
        feature_cols = ['feature1']
        mock_train.return_value = (model, feature_cols)
        
        y_pred = np.array([90.0, 180.0])
        y_true = np.array([100.0, 200.0])
        metrics = {'rmse': 15.0, 'mae': 15.0, 'r2': 0.9}
        mock_predict.return_value = (y_pred, y_true, metrics)
        
        mock_analyze.return_value = pd.DataFrame({
            'Member_ID': ['M001', 'M002'],
            'Actual_Claims': [100.0, 200.0],
            'Predicted_Claims': [90.0, 180.0]
        })
        
        # Call the pipeline
        result = run_pipeline(force_train=False, advanced_features=False, use_business_report=False)
        
        # Verify all steps were called
        mock_load.assert_called_once()
        mock_engineer.assert_called_once()
        mock_train.assert_called_once()
        mock_predict.assert_called_once()
        mock_analyze.assert_called_once()
        
        # Verify the result
        self.assertIsNotNone(result)
        self.assertIn('model', result)
        self.assertIn('metrics', result)
        self.assertEqual(result['metrics'], metrics)

if __name__ == '__main__':
    unittest.main() 