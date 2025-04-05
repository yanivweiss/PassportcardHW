#!/usr/bin/env python
"""
Run all tests for the PassportCard insurance claims prediction system.
"""
import os
import sys
import unittest
import argparse
import time
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

def run_all_tests(verbose=True, specific_test=None):
    """
    Run all test cases
    
    Parameters:
    -----------
    verbose : bool
        Whether to run tests in verbose mode
    specific_test : str
        Name of a specific test to run (e.g., 'TestPredictionPipeline')
    """
    start_time = time.time()
    print("="*80)
    print("RUNNING TESTS FOR PASSPORTCARD INSURANCE CLAIMS PREDICTION SYSTEM")
    print("="*80)
    
    # Define test directory
    test_dir = Path('tests')
    if not test_dir.exists():
        print(f"Error: Test directory '{test_dir}' not found!")
        return False
    
    # Discover and run tests
    loader = unittest.TestLoader()
    
    # Initialize test count
    total_tests = 0
    successful_tests = 0
    failed_tests = []
    
    if specific_test:
        print(f"\nRunning specific test: {specific_test}\n")
        # Try to load the specific test
        try:
            # Discover all test modules
            test_suite = loader.discover(str(test_dir))
            
            # Find the specific test in the suite
            selected_suite = unittest.TestSuite()
            
            def add_specific_tests(suite, test_name):
                for test in suite:
                    if isinstance(test, unittest.TestSuite):
                        add_specific_tests(test, test_name)
                    else:
                        if test_name in str(test):
                            selected_suite.addTest(test)
            
            add_specific_tests(test_suite, specific_test)
            
            if selected_suite.countTestCases() == 0:
                print(f"No tests found matching '{specific_test}'")
                return False
            
            # Run the selected tests
            runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
            result = runner.run(selected_suite)
            
            total_tests = result.testsRun
            successful_tests = total_tests - len(result.failures) - len(result.errors)
            failed_tests = [str(t[0]) for t in result.failures + result.errors]
        except Exception as e:
            print(f"Error running specific test: {e}")
            return False
    else:
        print("\nRunning all tests\n")
        
        # Discover all test modules
        test_suite = loader.discover(str(test_dir))
        
        # Run all tests
        runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
        result = runner.run(test_suite)
        
        total_tests = result.testsRun
        successful_tests = total_tests - len(result.failures) - len(result.errors)
        failed_tests = [str(t[0]) for t in result.failures + result.errors]
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Total tests: {total_tests}")
    print(f"Successful tests: {successful_tests}")
    print(f"Failed tests: {len(failed_tests)}")
    
    if failed_tests:
        print("\nFailed tests:")
        for test in failed_tests:
            print(f"  - {test}")
    
    print(f"\nTotal time elapsed: {elapsed_time:.2f} seconds")
    
    # Return success status
    return len(failed_tests) == 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tests for PassportCard insurance claims prediction")
    parser.add_argument('-v', '--verbose', action='store_true', help='Run tests in verbose mode')
    parser.add_argument('-t', '--test', type=str, help='Run a specific test class or method')
    args = parser.parse_args()
    
    success = run_all_tests(verbose=args.verbose, specific_test=args.test)
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1) 