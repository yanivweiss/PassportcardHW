#!/usr/bin/env python
"""
PassportCard Claims Analysis Pipeline

This script orchestrates the entire analysis pipeline, including:
1. Data preparation and cleaning
2. Feature engineering (basic and enhanced)
3. Model training and tuning
4. Visualization generation
5. Business report creation

Usage:
    python run_full_analysis.py [--test]

Arguments:
    --test    Run in test mode with a small sample
"""

import sys
import os
import time
import argparse
from datetime import datetime

# Create output directories if they don't exist
os.makedirs('visualizations', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

def main(test_mode=False):
    """Run the full analysis pipeline"""
    start_time = time.time()
    print(f"Starting PassportCard Claims Analysis Pipeline at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if test_mode:
        print("Running in TEST MODE with small sample data")
        # Import and run test data generation
        from create_test_data import create_test_claims_data, create_test_members_data
        create_test_claims_data(n_claims=500, n_members=50)
        create_test_members_data(n_members=50)
    
    # Step 1: Run the enhanced analysis
    print("\nRunning enhanced analysis...")
    import run_enhanced_analysis
    run_enhanced_analysis.main()
    
    # Step 2: Run unit tests to verify functionality
    print("\nRunning unit tests...")
    import unittest
    from test_models import TestDataPreparation, TestModeling
    
    # Create a test suite
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(TestDataPreparation))
    test_suite.addTest(unittest.makeSuite(TestModeling))
    
    # Run the tests
    test_runner = unittest.TextTestRunner(verbosity=2)
    test_results = test_runner.run(test_suite)
    
    # Step 3: Print summary
    end_time = time.time()
    duration = end_time - start_time
    print(f"\nAnalysis completed in {duration:.2f} seconds")
    print(f"Results saved to the following locations:")
    print(f"- Visualizations: ./visualizations/")
    print(f"- Business Report: ./business_report.md")
    print(f"- Model: ./best_model.pkl")
    print(f"- Enhanced Features: ./enhanced_features.csv")
    
    # Return success if all tests passed
    return 0 if test_results.wasSuccessful() else 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the PassportCard Claims Analysis Pipeline")
    parser.add_argument('--test', action='store_true', help='Run in test mode with small sample data')
    args = parser.parse_args()
    
    sys.exit(main(test_mode=args.test)) 