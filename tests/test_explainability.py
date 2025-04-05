import unittest
import pandas as pd
import numpy as np
import os
import sys
import shutil
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import warnings
from io import StringIO

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import explainability as ex

class TestExplainability(unittest.TestCase):
    """Test cases for explainability module"""

    @classmethod
    def setUpClass(cls):
        """Create test data and model that will be used across tests"""
        # Suppress matplotlib warnings
        plt.ioff()  # Turn off interactive mode
        warnings.filterwarnings("ignore", category=UserWarning)
        
        # Create output directories if they don't exist
        os.makedirs('visualizations/explainability', exist_ok=True)
        os.makedirs('visualizations/explainability/comparison', exist_ok=True)
        os.makedirs('reports', exist_ok=True)
        
        # Create synthetic data
        X, y = make_regression(n_samples=500, n_features=8, n_informative=4, random_state=42)
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        cls.X = pd.DataFrame(X, columns=feature_names)
        cls.y = y
        
        # Split for training
        X_train, X_test, y_train, y_test = train_test_split(cls.X, cls.y, test_size=0.2, random_state=42)
        cls.X_train = X_train
        cls.X_test = X_test
        cls.y_train = y_train
        cls.y_test = y_test
        
        # Train a simple XGBoost model
        cls.model = xgb.XGBRegressor(n_estimators=50, max_depth=3, random_state=42)
        cls.model.fit(X_train, y_train)
        
        # Get predictions
        cls.y_pred = cls.model.predict(X_test)

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests are run"""
        # Remove test visualization files if they exist
        for path in [
            'visualizations/explainability/shap_summary.png',
            'visualizations/explainability/shap_bar.png',
            'visualizations/explainability/single_prediction_force.png',
            'visualizations/explainability/single_prediction_waterfall.png',
            'visualizations/explainability/feature_interactions.png',
            'visualizations/explainability/partial_dependence.png',
            'visualizations/explainability/permutation_importance.png',
            'visualizations/explainability/comparison/global_vs_local.png',
            'visualizations/explainability/comparison/instance_0_force.png',
            'reports/explainability_report.md'
        ]:
            if os.path.exists(path):
                os.remove(path)

    def test_explain_model_predictions(self):
        """Test SHAP-based model explanation"""
        # Test with summary plot
        explanation = ex.explain_model_predictions(self.model, self.X_test, plot_type='summary')
        
        # Check if result is a dictionary with the expected keys
        self.assertIsInstance(explanation, dict)
        self.assertIn('shap_values', explanation)
        self.assertIn('shap_data', explanation)
        self.assertIn('explainer', explanation)
        
        # Check if SHAP values are calculated correctly
        self.assertEqual(len(explanation['shap_values']), len(self.X_test))
        
        # Verify that the plot was created
        self.assertTrue(os.path.exists('visualizations/explainability/shap_summary.png'))
        
        # Test with bar plot
        explanation = ex.explain_model_predictions(self.model, self.X_test, plot_type='bar')
        self.assertTrue(os.path.exists('visualizations/explainability/shap_bar.png'))

    def test_explain_single_prediction(self):
        """Test explanation of a single prediction"""
        # Test explanation for the first instance
        explanation = ex.explain_single_prediction(self.model, self.X_test, 0)
        
        # Check if result is a dictionary with the expected keys
        self.assertIsInstance(explanation, dict)
        self.assertIn('shap_values', explanation)
        self.assertIn('instance', explanation)
        self.assertIn('prediction', explanation)
        self.assertIn('base_value', explanation)
        
        # Check if prediction matches model's prediction
        self.assertAlmostEqual(explanation['prediction'], self.y_pred[0], places=5)
        
        # Verify that the plots were created
        self.assertTrue(os.path.exists('visualizations/explainability/single_prediction_force.png'))
        self.assertTrue(os.path.exists('visualizations/explainability/single_prediction_waterfall.png'))

    def test_analyze_feature_interactions(self):
        """Test analysis of feature interactions"""
        # Analyze feature interactions
        interactions = ex.analyze_feature_interactions(self.model, self.X_test)
        
        # Check if result is a DataFrame
        self.assertIsInstance(interactions, pd.DataFrame)
        
        # Check if required columns are present
        self.assertIn('feature1', interactions.columns)
        self.assertIn('feature2', interactions.columns)
        self.assertIn('strength', interactions.columns)
        
        # Verify that the plot was created
        self.assertTrue(os.path.exists('visualizations/explainability/feature_interactions.png'))

    def test_plot_partial_dependence(self):
        """Test creation of partial dependence plots"""
        # Plot partial dependence for the top 3 features
        features = ['feature_0', 'feature_1', 'feature_2']
        ex.plot_partial_dependence(self.model, self.X_test, features)
        
        # Verify that the plot was created
        self.assertTrue(os.path.exists('visualizations/explainability/partial_dependence.png'))

    def test_calculate_permutation_importance(self):
        """Test calculation of permutation feature importance"""
        # Calculate permutation importance
        importance = ex.calculate_permutation_importance(self.model, self.X_test, self.y_test)
        
        # Check if result is a DataFrame
        self.assertIsInstance(importance, pd.DataFrame)
        
        # Check if required columns are present
        self.assertIn('feature', importance.columns)
        self.assertIn('importance_mean', importance.columns)
        self.assertIn('importance_std', importance.columns)
        
        # Check if all features are included
        self.assertEqual(len(importance), len(self.X_test.columns))
        
        # Verify that the plot was created
        self.assertTrue(os.path.exists('visualizations/explainability/permutation_importance.png'))

    def test_compare_global_local_explanations(self):
        """Test comparison of global and local explanations"""
        # Compare global and local explanations
        comparison = ex.compare_global_local_explanations(self.model, self.X_test, self.y_test, n_instances=2)
        
        # Check if result is a dictionary with the expected keys
        self.assertIsInstance(comparison, dict)
        self.assertIn('global_importance', comparison)
        self.assertIn('local_explanations', comparison)
        self.assertIn('comparison', comparison)
        
        # Check local explanations
        self.assertIsInstance(comparison['local_explanations'], dict)
        self.assertEqual(len(comparison['local_explanations']), 2)  # 2 instances
        
        # Check comparison DataFrame
        self.assertIsInstance(comparison['comparison'], pd.DataFrame)
        self.assertIn('global_importance', comparison['comparison'].columns)
        self.assertIn('local_importance', comparison['comparison'].columns)
        
        # Verify that the plots were created
        self.assertTrue(os.path.exists('visualizations/explainability/comparison/global_vs_local.png'))
        self.assertTrue(os.path.exists('visualizations/explainability/comparison/instance_0_force.png'))

    def test_generate_explainability_report(self):
        """Test generation of explainability report"""
        # Generate report with a smaller dataset for faster testing
        X_small = self.X_test.iloc[:50]
        y_small = self.y_test[:50]
        
        # Redirect stdout to suppress progress prints
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        try:
            # Generate report
            report_path = ex.generate_explainability_report(self.model, X_small, y_small)
            
            # Check if report was created
            self.assertTrue(os.path.exists(report_path))
            
            # Check report content
            with open(report_path, 'r') as f:
                report_content = f.read()
                
            # Check for expected sections
            self.assertIn('# Model Explainability Report', report_content)
            self.assertIn('## Global Feature Importance', report_content)
            self.assertIn('### Permutation Importance', report_content)
            self.assertIn('### SHAP Feature Importance', report_content)
            self.assertIn('## Feature Interactions', report_content)
            self.assertIn('## Recommendations', report_content)
        finally:
            # Restore stdout
            sys.stdout = old_stdout

    def test_main_function(self):
        """Test the main demo function"""
        # Capture stdout to suppress output
        saved_stdout = sys.stdout
        try:
            out = StringIO()
            sys.stdout = out
            
            # Run the main function
            ex.main()
            
            # Check output
            output = out.getvalue()
            self.assertIn('Report generated', output)
            
        finally:
            # Restore stdout
            sys.stdout = saved_stdout

if __name__ == '__main__':
    unittest.main() 