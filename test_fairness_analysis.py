import unittest
import pandas as pd
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
import sys
from io import StringIO
import warnings

# Add parent directory to path to import modules
sys.path.append('.')
import fairness_analysis as fa

class TestFairnessAnalysis(unittest.TestCase):
    """Test cases for fairness_analysis module"""

    @classmethod
    def setUpClass(cls):
        """Create test data that will be used across tests"""
        # Suppress matplotlib warnings
        plt.ioff()  # Turn off interactive mode
        warnings.filterwarnings("ignore", category=UserWarning)
        
        # Create output directories if they don't exist
        os.makedirs('visualizations/fairness', exist_ok=True)
        os.makedirs('reports', exist_ok=True)
        
        # Create synthetic data with two groups
        np.random.seed(42)
        n_samples = 500
        
        # Create group membership
        cls.group_membership = np.random.choice(['A', 'B'], n_samples)
        
        # Create biased data - Group B has higher targets on average
        cls.y_true = np.zeros(n_samples)
        cls.y_true[cls.group_membership == 'A'] = np.random.normal(100, 30, sum(cls.group_membership == 'A'))
        cls.y_true[cls.group_membership == 'B'] = np.random.normal(150, 40, sum(cls.group_membership == 'B'))
        
        # Create biased predictions - under-predict for group B
        cls.y_pred = np.zeros(n_samples)
        cls.y_pred[cls.group_membership == 'A'] = cls.y_true[cls.group_membership == 'A'] + np.random.normal(0, 20, sum(cls.group_membership == 'A'))
        cls.y_pred[cls.group_membership == 'B'] = cls.y_true[cls.group_membership == 'B'] * 0.8 + np.random.normal(0, 30, sum(cls.group_membership == 'B'))
        
        # Create a continuous feature correlated with the target
        cls.feature = np.zeros(n_samples)
        cls.feature[cls.group_membership == 'A'] = cls.y_true[cls.group_membership == 'A'] / 10 + np.random.normal(0, 2, sum(cls.group_membership == 'A'))
        cls.feature[cls.group_membership == 'B'] = cls.y_true[cls.group_membership == 'B'] / 12 + np.random.normal(0, 3, sum(cls.group_membership == 'B'))
        
        # Create DataFrame for testing
        cls.X = pd.DataFrame({
            'feature': cls.feature,
            'group': cls.group_membership,
            'Member_ID': range(n_samples)
        })
        
        # Create a dummy model
        cls.dummy_model = type('DummyModel', (), {'predict': lambda self, X: cls.y_pred})()

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests are run"""
        # Remove test visualization files if they exist
        if os.path.exists('visualizations/fairness/fairness_metrics_group.png'):
            os.remove('visualizations/fairness/fairness_metrics_group.png')
        if os.path.exists('visualizations/fairness/disparate_impact_group.png'):
            os.remove('visualizations/fairness/disparate_impact_group.png')
        if os.path.exists('visualizations/fairness/calibration_by_group.png'):
            os.remove('visualizations/fairness/calibration_by_group.png')
        if os.path.exists('visualizations/fairness/performance_by_feature.png'):
            os.remove('visualizations/fairness/performance_by_feature.png')
        if os.path.exists('reports/fairness_report.md'):
            os.remove('reports/fairness_report.md')

    def test_calculate_fairness_metrics_regression(self):
        """Test calculation of fairness metrics for regression"""
        # Calculate fairness metrics
        result = fa.calculate_fairness_metrics_regression(
            self.y_true, self.y_pred, self.group_membership, group_name='Group'
        )
        
        # Check if result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)
        
        # Check if both groups are included
        self.assertIn('A', result.index)
        self.assertIn('B', result.index)
        
        # Check if metrics were calculated correctly
        self.assertIn('rmse', result.columns)
        self.assertIn('demographic_parity_diff', result.columns)
        self.assertIn('prediction_proportionality', result.columns)
        
        # Check if bias is detected correctly (B should be underpredicted)
        self.assertTrue(result.loc['B', 'prediction_proportionality'] < 1.0)
        
        # Verify that the plot was created
        self.assertTrue(os.path.exists('visualizations/fairness/fairness_metrics_group.png'))

    def test_calculate_disparate_impact(self):
        """Test calculation of disparate impact"""
        # Create DataFrame for disparate impact analysis
        df = pd.DataFrame({
            'group': self.group_membership,
            'y_true': self.y_true,
            'y_pred': self.y_pred
        })
        
        # Calculate disparate impact
        threshold = np.percentile(self.y_pred, 75)  # 75th percentile as threshold
        result = fa.calculate_disparate_impact(df, 'group', 'y_pred', threshold)
        
        # Check if result is a dictionary with the expected keys
        self.assertIsInstance(result, dict)
        self.assertIn('group_rates', result)
        self.assertIn('reference_group', result)
        self.assertIn('disparate_impact', result)
        self.assertIn('overall_rate', result)
        
        # Check if rates add up to the expected total rate
        group_rates = result['group_rates']
        overall_rate = result['overall_rate']
        expected_overall = ((group_rates['A'] * (df['group'] == 'A').sum()) + 
                            (group_rates['B'] * (df['group'] == 'B').sum())) / len(df)
        self.assertAlmostEqual(overall_rate, expected_overall, places=5)
        
        # Verify that the plot was created
        self.assertTrue(os.path.exists('visualizations/fairness/disparate_impact_group.png'))

    def test_calculate_calibration_by_group(self):
        """Test calculation of calibration curves by group"""
        # Calculate calibration curves
        result = fa.calculate_calibration_by_group(
            self.y_true, self.y_pred, self.group_membership, n_bins=5
        )
        
        # Check if result is a dictionary
        self.assertIsInstance(result, dict)
        self.assertIn('A', result)
        self.assertIn('B', result)
        
        # Check calibration curve properties
        for group in ['A', 'B']:
            curve = result[group]
            self.assertIsInstance(curve, pd.DataFrame)
            self.assertIn('bin', curve.columns)
            self.assertIn('y_pred', curve.columns)
            self.assertIn('y_true', curve.columns)
        
        # Verify that the plot was created
        self.assertTrue(os.path.exists('visualizations/fairness/calibration_by_group.png'))

    def test_audit_model_performance(self):
        """Test comprehensive model audit"""
        # Run model audit
        audit_results = fa.audit_model_performance(
            self.dummy_model, 
            self.X, 
            self.y_true, 
            'group', 
            continuous_cols=['feature']
        )
        
        # Check if result is a dictionary with the expected keys
        self.assertIsInstance(audit_results, dict)
        self.assertIn('overall_metrics', audit_results)
        self.assertIn('fairness_metrics', audit_results)
        self.assertIn('disparate_impact', audit_results)
        self.assertIn('calibration', audit_results)
        self.assertIn('continuous_performance', audit_results)
        
        # Check overall metrics
        self.assertIn('rmse', audit_results['overall_metrics'])
        self.assertIn('mae', audit_results['overall_metrics'])
        self.assertIn('r2', audit_results['overall_metrics'])
        
        # Check continuous performance
        self.assertIn('feature', audit_results['continuous_performance'])
        
        # Verify that the plot was created
        self.assertTrue(os.path.exists('visualizations/fairness/performance_by_feature.png'))

    def test_generate_fairness_report(self):
        """Test generation of fairness report"""
        # Run model audit first
        audit_results = fa.audit_model_performance(
            self.dummy_model, 
            self.X, 
            self.y_true, 
            'group', 
            continuous_cols=['feature']
        )
        
        # Generate report
        report_path = fa.generate_fairness_report(audit_results)
        
        # Check if report was created
        self.assertTrue(os.path.exists(report_path))
        
        # Check report content
        with open(report_path, 'r') as f:
            report_content = f.read()
            
        # Check for expected sections
        self.assertIn('# Model Fairness and Performance Audit Report', report_content)
        self.assertIn('## Overall Model Performance', report_content)
        self.assertIn('## Fairness Metrics by Group', report_content)
        self.assertIn('## Disparate Impact Analysis', report_content)
        self.assertIn('## Summary and Recommendations', report_content)

    def test_main_function(self):
        """Test the main demo function"""
        # Capture stdout
        saved_stdout = sys.stdout
        try:
            out = StringIO()
            sys.stdout = out
            
            # Run the main function
            fa.main()
            
            # Check output
            output = out.getvalue()
            self.assertIn('Report generated', output)
            
        finally:
            # Restore stdout
            sys.stdout = saved_stdout

if __name__ == '__main__':
    unittest.main() 