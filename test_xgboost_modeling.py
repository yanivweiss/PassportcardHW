import unittest
import pandas as pd
import numpy as np
import os
import shutil
from datetime import datetime, timedelta

# Import module to test
from xgboost_modeling import (
    prepare_data_for_xgboost,
    basic_xgboost_train,
    evaluate_xgboost_model,
    save_xgboost_model
)

class TestXGBoostModeling(unittest.TestCase):
    """Test the XGBoost modeling functionality"""
    
    def setUp(self):
        """Create sample test data"""
        # Create synthetic data
        np.random.seed(42)
        n_samples = 100
        n_features = 10
        
        # Create synthetic feature matrix
        X = np.random.randn(n_samples, n_features)
        # Create target with some relationship to features
        y = 2 * X[:, 0] + 3 * X[:, 1] - 1.5 * X[:, 2] + 0.5 * np.random.randn(n_samples)
        
        # Add some datetime columns 
        dates = [datetime(2022, 1, 1) + timedelta(days=i) for i in range(n_samples)]
        
        # Add some categorical columns
        categories = ['A', 'B', 'C']
        cat_col = [categories[i % len(categories)] for i in range(n_samples)]
        
        # Convert to DataFrame
        feature_names = [f'feature_{i}' for i in range(n_features)]
        self.features_df = pd.DataFrame(X, columns=feature_names)
        self.features_df['Member_ID'] = range(n_samples)
        self.features_df['PolicyID'] = range(1000, 1000 + n_samples)
        self.features_df['PolicyStartDate'] = dates
        self.features_df['category'] = cat_col
        self.features_df['future_6m_claims'] = y
        
        # Create visualizations directory if it doesn't exist
        os.makedirs('visualizations', exist_ok=True)
    
    def tearDown(self):
        """Clean up test artifacts"""
        # Remove test model files
        test_files = ['test_model.pkl', 'xgboost_feature_importance.csv']
        for file in test_files:
            if os.path.exists(file):
                os.remove(file)
    
    def test_prepare_data_for_xgboost(self):
        """Test data preparation for XGBoost"""
        X_train, X_test, y_train, y_test, feature_cols = prepare_data_for_xgboost(self.features_df)
        
        # Check correct shapes
        self.assertGreater(len(X_train), 0)
        self.assertGreater(len(X_test), 0)
        self.assertEqual(X_train.shape[1], 11)  # 10 numeric features + 1 categorical
        self.assertEqual(X_test.shape[1], 11)
        
        # Check feature columns
        self.assertEqual(len(feature_cols), 11)
        for i in range(10):
            self.assertIn(f'feature_{i}', feature_cols)
        self.assertIn('category', feature_cols)
        
        # Check Member_ID and dates were excluded
        self.assertNotIn('Member_ID', feature_cols)
        self.assertNotIn('PolicyStartDate', feature_cols)
    
    def test_basic_xgboost_train(self):
        """Test basic XGBoost model training"""
        X_train, X_test, y_train, y_test, _ = prepare_data_for_xgboost(self.features_df)
        
        model, y_pred, metrics = basic_xgboost_train(X_train, y_train, X_test, y_test)
        
        # Check model was created
        self.assertIsNotNone(model)
        
        # Check predictions were made
        self.assertEqual(len(y_pred), len(y_test))
        
        # Check metrics were calculated
        self.assertIn('RMSE', metrics)
        self.assertIn('MAE', metrics)
        self.assertIn('R2', metrics)
        
        # Check metrics are reasonable
        self.assertGreater(metrics['R2'], 0)  # R2 should be positive (better than mean)
    
    def test_save_xgboost_model(self):
        """Test model saving functionality"""
        X_train, X_test, y_train, y_test, feature_cols = prepare_data_for_xgboost(self.features_df)
        
        model, _, _ = basic_xgboost_train(X_train, y_train, X_test, y_test)
        
        # Save model
        test_file = 'test_model.pkl'
        save_xgboost_model(model, feature_cols, test_file)
        
        # Check file was created
        self.assertTrue(os.path.exists(test_file))
        
        # Check file has non-zero size
        self.assertGreater(os.path.getsize(test_file), 0)

if __name__ == '__main__':
    unittest.main() 