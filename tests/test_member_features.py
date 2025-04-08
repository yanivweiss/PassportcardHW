import unittest
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the module to test
from src.member_feature_engineering import (
    extract_member_features,
    prepare_member_features
)

class TestMemberFeatureEngineering(unittest.TestCase):
    """Test cases for member feature engineering functionality"""
    
    def setUp(self):
        """Create sample test data"""
        # Create synthetic member data
        np.random.seed(42)
        n_samples = 100
        
        # Create member IDs
        member_ids = range(1000, 1000 + n_samples)
        
        # Create dates
        current_date = datetime.now()
        policy_start_dates = [current_date - timedelta(days=np.random.randint(30, 1000)) for _ in range(n_samples)]
        policy_end_dates = [start_date + timedelta(days=np.random.randint(365, 1095)) for start_date in policy_start_dates]
        birth_dates = [current_date - timedelta(days=np.random.randint(6570, 29200)) for _ in range(n_samples)]
        
        # Create other attributes
        genders = np.random.choice(['M', 'F'], size=n_samples)
        countries = np.random.choice(['USA', 'UK', 'Canada', 'Germany', 'France', 'Other'], 
                                     size=n_samples, p=[0.3, 0.2, 0.15, 0.1, 0.1, 0.15])
        bmis = np.random.normal(25, 5, n_samples)
        
        # Create questionnaire data
        questionnaire_cols = {}
        for i in range(1, 6):
            questionnaire_cols[f'Questionnaire_{i}'] = np.random.choice([0, 1], size=n_samples)
            
        # Combine into DataFrame
        self.members_df = pd.DataFrame({
            'Member_ID': member_ids,
            'PolicyStartDate': policy_start_dates,
            'PolicyEndDate': policy_end_dates,
            'DateOfBirth': birth_dates,
            'Gender': genders,
            'CountryOfOrigin': countries,
            'BMI': bmis,
            **questionnaire_cols
        })
        
        # Create synthetic claims data for testing prepare_member_features
        n_claims = n_samples * 3  # Average 3 claims per member
        
        # Random subset of members for claims
        claim_member_ids = np.random.choice(member_ids, size=n_claims)
        
        # Random service dates
        service_dates = [current_date - timedelta(days=np.random.randint(1, 730)) for _ in range(n_claims)]
        
        # Random payment amounts
        payments = np.random.exponential(scale=500, size=n_claims)
        
        # Combine into DataFrame
        self.claims_df = pd.DataFrame({
            'Member_ID': claim_member_ids,
            'ServiceDate': service_dates,
            'TotPaymentUSD': payments
        })
    
    def test_extract_member_features(self):
        """Test extraction of features from member data"""
        # Extract features
        features = extract_member_features(self.members_df)
        
        # Check output structure
        self.assertIsInstance(features, pd.DataFrame)
        self.assertGreater(len(features), 0)
        
        # Check Member_ID is preserved
        self.assertIn('Member_ID', features.columns)
        self.assertEqual(len(features), len(self.members_df))
        
        # Check age features were created
        self.assertIn('Age', features.columns)
        
        # Check age groups were created
        age_group_cols = [col for col in features.columns if col.startswith('age_group_')]
        self.assertGreater(len(age_group_cols), 0)
        
        # Check policy duration was calculated
        self.assertIn('PolicyDuration', features.columns)
        
        # Check BMI categories were created
        bmi_cols = [col for col in features.columns if col.startswith('bmi_')]
        self.assertGreater(len(bmi_cols), 0)
        
        # Check country features were created
        country_cols = [col for col in features.columns if col.startswith('country_')]
        self.assertGreater(len(country_cols), 0)
        
        # Check gender features were created
        gender_cols = [col for col in features.columns if col.startswith('gender_')]
        self.assertGreater(len(gender_cols), 0)
    
    def test_prepare_member_features(self):
        """Test preparation of member features with target variable"""
        # Set a cutoff date
        cutoff_date = datetime.now() - timedelta(days=180)
        
        # Prepare features with claims data for target
        features = prepare_member_features(self.members_df, self.claims_df, cutoff_date)
        
        # Check output structure
        self.assertIsInstance(features, pd.DataFrame)
        self.assertEqual(len(features), len(self.members_df))
        
        # Check target variable was created
        self.assertIn('future_6m_claims', features.columns)
        
        # Check all rows have a target value (default 0 for no claims)
        self.assertEqual(features['future_6m_claims'].isna().sum(), 0)
    
    def test_prepare_member_features_without_claims(self):
        """Test preparation of member features without claims data"""
        # Prepare features without claims data
        features = prepare_member_features(self.members_df)
        
        # Check output structure
        self.assertIsInstance(features, pd.DataFrame)
        self.assertEqual(len(features), len(self.members_df))
        
        # Check no target variable was created
        self.assertNotIn('future_6m_claims', features.columns)
    
    def test_interactions(self):
        """Test interaction features"""
        # Extract features
        features = extract_member_features(self.members_df)
        
        # Check interaction features
        if 'Age' in features.columns and 'BMI' in features.columns:
            self.assertIn('Age_BMI_interaction', features.columns)

if __name__ == '__main__':
    unittest.main() 