#!/usr/bin/env python
"""
PassportCard Claims Analysis - Main Entry Point

This script serves as the main entry point for the PassportCard insurance claims analysis pipeline.
It provides a simplified interface to run the full prediction pipeline with various options.

Usage:
    python main.py [options]
    
Options:
    --force-train       Force training a new model even if one exists
    --basic-features    Use only basic features without advanced ones
    --no-report         Skip generating business report
"""

import os
import sys
import argparse
from datetime import datetime

def main():
    """Main entry point for the PassportCard analysis pipeline"""
    parser = argparse.ArgumentParser(description="Run the PassportCard Claims Analysis Pipeline")
    parser.add_argument('--force-train', action='store_true', help='Force training a new model even if one exists')
    parser.add_argument('--basic-features', action='store_true', help='Use only basic features without advanced ones')
    parser.add_argument('--no-report', action='store_true', help='Skip generating business report')
    args = parser.parse_args()
    
    start_time = datetime.now()
    
    print(f"Starting PassportCard Claims Analysis Pipeline at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)
    os.makedirs('outputs/figures/predictions', exist_ok=True)
    os.makedirs('outputs/tables', exist_ok=True)
    
    # Import and run the main prediction pipeline
    try:
        from src.run_prediction_pipeline import run_pipeline
        
        result = run_pipeline(
            force_train=args.force_train,
            advanced_features=not args.basic_features,
            use_business_report=not args.no_report
        )
        
        # Calculate and print elapsed time
        end_time = datetime.now()
        elapsed_time = (end_time - start_time).total_seconds() / 60.0
        
        print("\nAnalysis Pipeline Summary:")
        print(f"- Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"- Completed: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"- Total runtime: {elapsed_time:.2f} minutes")
        
        if result and 'metrics' in result:
            print("\nModel Performance:")
            for metric, value in result['metrics'].items():
                print(f"  {metric.upper()}: {value:.4f}")
        
        print("\nOutputs:")
        print("- Model: models/best_xgboost_model.pkl")
        print("- Business Report: reports/advanced_business_report.md")
        print("- Visualizations: visualizations/ and outputs/figures/")
        print("- Prediction Results: outputs/tables/prediction_results.csv")
        
        return 0
    
    except Exception as e:
        print(f"Error running prediction pipeline: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 