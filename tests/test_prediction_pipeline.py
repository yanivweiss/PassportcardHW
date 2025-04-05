"""
Test the prediction pipeline components and full end-to-end flow
"""
import os
import sys
import unittest
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from pathlib import Path
import shutil
import tempfile

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import pipeline components
from src.run_prediction_pipeline import (
    load_and_preprocess_data,
    engineer_features,
    create_target_variable,
    train_or_load_model,
    make_predictions,
    analyze_results,
    run_pipeline
)

class TestPredictionPipeline(unittest.TestCase):
    """Test the prediction pipeline components and integration"""
    
    @classmethod
    def setUpClass(cls):
        """Create temp directory for test outputs"""
        cls.test_output_dir = Path(tempfile.mkdtemp())
        
        # Create subdirectories
        (cls.test_output_dir / 'figures' / 'predictions').mkdir(parents=True, exist_ok=True)
        (cls.test_output_dir / 'tables').mkdir(parents=True, exist_ok=True)
        
    @classmethod
    def tearDownClass(cls):
        """Clean up temp directory"""
        if os.path.exists(cls.test_output_dir):
            shutil.rmtree(cls.test_output_dir)
    
    def test_1_load_and_preprocess_data(self):
        """Test data loading and preprocessing"""
        # Skip if data files don't exist
        if not (Path('data/processed/claims_data_clean.csv').exists() and 
                Path('data/processed/members_data_clean.csv').exists()):
            self.skipTest("Required data files not found")
        
        claims_df, members_df = load_and_preprocess_data()
        
        # Check if data was loaded
        self.assertIsNotNone(claims_df)
        self.assertIsNotNone(members_df)
        
        # Check that data has expected columns
        if claims_df is not None:
            self.assertIn('Member_ID', claims_df.columns)
            if 'ServiceDate' in claims_df.columns:
                self.assertTrue(pd.api.types.is_datetime64_dtype(claims_df['ServiceDate']))
        
        if members_df is not None:
            self.assertIn('Member_ID', members_df.columns)
    
    def test_2_engineer_features(self):
        """Test feature engineering"""
        # Skip if data files don't exist
        if not (Path('data/processed/claims_data_clean.csv').exists() and 
                Path('data/processed/members_data_clean.csv').exists()):
            self.skipTest("Required data files not found")
        
        claims_df, members_df = load_and_preprocess_data()
        if claims_df is None or members_df is None:
            self.skipTest("Could not load data for feature engineering test")
        
        features_df, cutoff_date = engineer_features(claims_df, members_df)
        
        # Check if features were created
        self.assertIsNotNone(features_df)
        self.assertIsNotNone(cutoff_date)
        
        # Check that expected features were created
        if features_df is not None:
            self.assertIn('Member_ID', features_df.columns)
            if 'future_6m_claims' in features_df.columns:
                self.assertTrue(pd.api.types.is_numeric_dtype(features_df['future_6m_claims']))
    
    def test_3_target_variable_creation(self):
        """Test target variable creation"""
        # Skip if claims data doesn't exist
        if not Path('data/processed/claims_data_clean.csv').exists():
            self.skipTest("Claims data not found for target variable test")
        
        claims_df, _ = load_and_preprocess_data()
        if claims_df is None:
            self.skipTest("Could not load claims data for target variable test")
        
        # Check if ServiceDate column exists
        if 'ServiceDate' not in claims_df.columns:
            self.skipTest("ServiceDate column not found in claims data")
        
        # Set a cutoff date
        cutoff_date = claims_df['ServiceDate'].max() - pd.Timedelta(days=180)
        
        future_claims = create_target_variable(claims_df, cutoff_date)
        
        # Check if target variable was created
        self.assertIsNotNone(future_claims)
        
        # Check that expected columns are present
        if future_claims is not None:
            self.assertIn('Member_ID', future_claims.columns)
            self.assertIn('future_6m_claims', future_claims.columns)
    
    def test_4_train_or_load_model(self):
        """Test model training or loading"""
        # Try loading the integrated features
        try:
            features_df = pd.read_csv('data/processed/integrated_features.csv')
        except FileNotFoundError:
            # If integrated features don't exist, try creating them
            if not (Path('data/processed/claims_data_clean.csv').exists() and 
                    Path('data/processed/members_data_clean.csv').exists()):
                self.skipTest("Required data files not found for model test")
            
            claims_df, members_df = load_and_preprocess_data()
            if claims_df is None or members_df is None:
                self.skipTest("Could not load data for model test")
            
            features_df, _ = engineer_features(claims_df, members_df)
        
        if features_df is None:
            self.skipTest("Could not create or load features for model test")
        
        # Test model loading (if it exists)
        model, feature_cols = train_or_load_model(features_df, force_train=False)
        
        # If model doesn't exist, try training with minimal data
        if model is None:
            # Create a minimal dataset for testing
            if 'future_6m_claims' not in features_df.columns:
                self.skipTest("Target variable not found in features")
                
            # Subsample for faster training
            if len(features_df) > 20:
                sampled_df = features_df.sample(20, random_state=42)
            else:
                sampled_df = features_df
                
            model, feature_cols = train_or_load_model(sampled_df, force_train=True)
        
        # Check if model was loaded or trained
        self.assertIsNotNone(model)
        self.assertIsNotNone(feature_cols)
        self.assertTrue(len(feature_cols) > 0)
    
    def test_5_make_predictions(self):
        """Test making predictions"""
        # Try loading the integrated features
        try:
            features_df = pd.read_csv('data/processed/integrated_features.csv')
        except FileNotFoundError:
            self.skipTest("Integrated features not found for prediction test")
        
        # Load or train model
        model, feature_cols = train_or_load_model(features_df, force_train=False)
        if model is None:
            self.skipTest("Could not load or train model for prediction test")
        
        # Make predictions
        y_pred, y_true, metrics = make_predictions(model, feature_cols, features_df)
        
        # Check predictions
        self.assertIsNotNone(y_pred)
        self.assertIsNotNone(y_true)
        
        # Check that predictions have expected length
        if y_pred is not None:
            self.assertEqual(len(y_pred), len(features_df))
        
        # Check metrics if available
        if metrics and 'rmse' in metrics:
            self.assertTrue(metrics['rmse'] > 0)
    
    def test_6_analyze_results(self):
        """Test result analysis"""
        # Create synthetic data for testing
        np.random.seed(42)
        n_samples = 50
        
        # Create synthetic features, predictions, and actual values
        features_df = pd.DataFrame({
            'Member_ID': range(1, n_samples+1),
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples),
            'future_6m_claims': np.abs(np.random.exponential(1000, n_samples))
        })
        
        y_true = features_df['future_6m_claims'].values
        y_pred = y_true * (0.8 + 0.4 * np.random.random(n_samples))  # Add noise
        
        # Test analyze_results with synthetic data
        results_df = analyze_results(y_true, y_pred, features_df, output_dir=str(self.test_output_dir))
        
        # Check if results were created
        self.assertIsNotNone(results_df)
        
        # Check that output files were created
        self.assertTrue((self.test_output_dir / 'figures' / 'predictions' / 'actual_vs_predicted.png').exists())
        self.assertTrue((self.test_output_dir / 'figures' / 'predictions' / 'residual_plot.png').exists())
        self.assertTrue((self.test_output_dir / 'tables' / 'prediction_results.csv').exists())
    
    def test_7_full_pipeline_integration(self):
        """Test the full pipeline integration, using limited data for speed"""
        # Skip this test if it's running in CI environment (too slow)
        if os.environ.get('CI') == 'true':
            self.skipTest("Skipping full pipeline test in CI environment")
        
        # Check if we have the necessary data files
        if not (Path('data/processed/claims_data_clean.csv').exists() and 
                Path('data/processed/members_data_clean.csv').exists()):
            self.skipTest("Required data files not found for full pipeline test")
        
        # Run pipeline with minimal training (or load existing model)
        result = run_pipeline(force_train=False)
        
        # Check for pipeline success
        self.assertIsNotNone(result)
        
        if result:
            self.assertIn('model', result)
            self.assertIn('predictions', result)
            self.assertIn('metrics', result)
            self.assertTrue(len(result['predictions']) > 0)

class TestPredictionPipelineMocking(unittest.TestCase):
    """
    Test the prediction pipeline with mocked data 
    (useful when real data is not available)
    """
    
    def setUp(self):
        """Create mock data for testing"""
        # Create synthetic claims data
        n_claims = 100
        self.claims_df = pd.DataFrame({
            'ClaimNumber': range(1, n_claims+1),
            'Member_ID': np.random.choice(range(1, 21), n_claims),
            'ServiceDate': pd.date_range(start='2022-01-01', periods=n_claims),
            'TotPaymentUSD': np.abs(np.random.exponential(1000, n_claims)),
            'ServiceGroup': np.random.choice(['A', 'B', 'C'], n_claims),
            'ServiceType': np.random.choice(['X', 'Y', 'Z'], n_claims)
        })
        
        # Create synthetic members data
        n_members = 50
        self.members_df = pd.DataFrame({
            'Member_ID': range(1, n_members+1),
            'Age': np.random.randint(18, 80, n_members),
            'Gender': np.random.choice(['M', 'F'], n_members),
            'PolicyStartDate': pd.date_range(start='2021-01-01', periods=n_members),
            'PolicyEndDate': pd.date_range(start='2023-01-01', periods=n_members),
            'CountryOfResidence': np.random.choice(['USA', 'UK', 'Canada'], n_members)
        })
        
        # Create temporary output directory
        self.test_output_dir = Path(tempfile.mkdtemp())
        (self.test_output_dir / 'figures' / 'predictions').mkdir(parents=True, exist_ok=True)
        (self.test_output_dir / 'tables').mkdir(parents=True, exist_ok=True)
    
    def tearDown(self):
        """Clean up temp directory"""
        if os.path.exists(self.test_output_dir):
            shutil.rmtree(self.test_output_dir)
    
    def test_mock_create_target_variable(self):
        """Test target variable creation with mock data"""
        # Set a cutoff date
        cutoff_date = self.claims_df['ServiceDate'].max() - pd.Timedelta(days=180)
        
        # Create target variable
        target_df = create_target_variable(self.claims_df, cutoff_date)
        
        # Check result
        self.assertIsNotNone(target_df)
        self.assertIn('Member_ID', target_df.columns)
        self.assertIn('future_6m_claims', target_df.columns)
    
    def test_mock_analyze_results(self):
        """Test result analysis with mock data"""
        # Create mock predictions and actuals
        n_samples = 30
        mock_features = pd.DataFrame({
            'Member_ID': range(1, n_samples+1),
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples)
        })
        
        y_true = np.abs(np.random.exponential(1000, n_samples))
        y_pred = y_true * (0.7 + 0.6 * np.random.random(n_samples))  # Add noise
        
        # Test analyze_results
        results_df = analyze_results(y_true, y_pred, mock_features, output_dir=str(self.test_output_dir))
        
        # Check results
        self.assertIsNotNone(results_df)
        self.assertEqual(len(results_df), n_samples)
        self.assertIn('Member_ID', results_df.columns)
        self.assertIn('Actual_Claims', results_df.columns)
        self.assertIn('Predicted_Claims', results_df.columns)

if __name__ == '__main__':
    unittest.main() 