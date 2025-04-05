"""
End-to-End Prediction Pipeline for Insurance Claims

This script runs the full prediction pipeline from data loading to prediction:
1. Load and preprocess claims and member data
2. Engineer features (basic, enhanced, and advanced temporal features) 
3. Train model (if needed) or load existing model
4. Make predictions and evaluate
5. Generate visualizations and reports

Note: This is a consolidated version that incorporates functionality from:
- run_enhanced_analysis.py
- run_enhanced_modeling.py
- run_enhanced_pipeline.py
- run_full_analysis.py
- run_advanced_analysis_2.py
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import custom modules
try:
    from enhanced_data_preparation import handle_missing_values_advanced, detect_and_handle_outliers, scale_features
    from enhanced_feature_engineering import create_date_features, create_cyclical_features, create_customer_behavior_features
    from xgboost_modeling import prepare_data_for_xgboost, train_xgboost_model, evaluate_xgboost_model
    from error_analysis import analyze_prediction_errors, create_regression_confusion_matrix
    from fairness_analysis import audit_model_performance
except ImportError:
    # Fall back to direct imports when running as a script
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.enhanced_data_preparation import handle_missing_values_advanced, detect_and_handle_outliers, scale_features
    from src.enhanced_feature_engineering import create_date_features, create_cyclical_features, create_customer_behavior_features
    from src.xgboost_modeling import prepare_data_for_xgboost, train_xgboost_model, evaluate_xgboost_model
    from src.error_analysis import analyze_prediction_errors, create_regression_confusion_matrix
    from src.fairness_analysis import audit_model_performance

def load_and_preprocess_data():
    """
    Load and preprocess claims and member data
    """
    print("\n1. Loading and preprocessing data...")
    
    try:
        # Load data
        claims_df = pd.read_csv('data/processed/claims_data_clean.csv')
        members_df = pd.read_csv('data/processed/members_data_clean.csv')
        
        print(f"Loaded claims data: {claims_df.shape[0]} rows, {claims_df.shape[1]} columns")
        print(f"Loaded members data: {members_df.shape[0]} rows, {members_df.shape[1]} columns")
        
        # Convert date columns to datetime
        if 'ServiceDate' in claims_df.columns:
            claims_df['ServiceDate'] = pd.to_datetime(claims_df['ServiceDate'])
        
        date_columns = ['PolicyStartDate', 'PolicyEndDate', 'DateOfBirth']
        for col in date_columns:
            if col in members_df.columns:
                members_df[col] = pd.to_datetime(members_df[col])
        
        # Handle missing values
        claims_df = handle_missing_values_advanced(claims_df)
        members_df = handle_missing_values_advanced(members_df)
        
        # Detect and handle outliers in claims amount
        if 'TotPaymentUSD' in claims_df.columns:
            claims_df, outliers_info = detect_and_handle_outliers(
                claims_df, 
                columns=['TotPaymentUSD'], 
                method='iqr',
                visualization=False
            )
            print(f"Detected and handled {len(outliers_info.get('TotPaymentUSD', []))} outliers in claims amount")
        
        return claims_df, members_df
    
    except Exception as e:
        print(f"Error loading or preprocessing data: {e}")
        return None, None

def engineer_features(claims_df, members_df):
    """
    Engineer features from claims and member data
    """
    print("\n2. Engineering features...")
    
    try:
        # Define cutoff date for train/test split and prediction window
        if 'ServiceDate' in claims_df.columns:
            max_date = claims_df['ServiceDate'].max()
            cutoff_date = max_date - timedelta(days=180)  # 6 months
        else:
            # Use current date minus 6 months if ServiceDate not available
            max_date = datetime.now()
            cutoff_date = max_date - timedelta(days=180)
        
        print(f"Using cutoff date: {cutoff_date}")
        
        # Create date features from claims
        if 'ServiceDate' in claims_df.columns:
            claims_with_dates = create_date_features(claims_df, 'ServiceDate')
            
            # Create cyclical features for month and day of week
            claims_with_dates = create_cyclical_features(claims_with_dates, 'ServiceDate_month', 12)
            claims_with_dates = create_cyclical_features(claims_with_dates, 'ServiceDate_dayofweek', 7)
        else:
            claims_with_dates = claims_df.copy()
        
        # Create customer behavior features
        if all(col in claims_df.columns for col in ['Member_ID', 'ServiceDate', 'TotPaymentUSD']):
            customer_features = create_customer_behavior_features(
                claims_df,
                member_id_col='Member_ID',
                date_col='ServiceDate',
                amount_col='TotPaymentUSD'
            )
            print(f"Created {customer_features.shape[1]} customer behavior features")
        else:
            # Create empty DataFrame if required columns are missing
            customer_features = pd.DataFrame({'Member_ID': members_df['Member_ID'].unique()})
            print("Warning: Required columns for customer behavior features are missing")
        
        # Create target variable (future claims in the next 6 months)
        future_claims = create_target_variable(claims_df, cutoff_date)
        
        # Merge all features
        features_df = members_df.merge(customer_features, on='Member_ID', how='left')
        features_df = features_df.merge(future_claims, on='Member_ID', how='left')
        
        # Fill missing values and apply final preprocessing
        features_df = features_df.fillna(0)
        
        # Save integrated features
        Path('data/processed').mkdir(parents=True, exist_ok=True)
        features_df.to_csv('data/processed/integrated_features.csv', index=False)
        
        print(f"Successfully engineered features: {features_df.shape[1]} total features")
        print(f"Features saved to: data/processed/integrated_features.csv")
        
        return features_df, cutoff_date
        
    except Exception as e:
        print(f"Error engineering features: {e}")
        return None, None

def create_target_variable(claims_df, cutoff_date, prediction_window=180):
    """
    Create target variable by calculating total claims per member in the future window
    """
    try:
        end_date = cutoff_date + timedelta(days=prediction_window)
        
        # Filter claims in prediction window
        future_claims = claims_df[
            (claims_df['ServiceDate'] > cutoff_date) & 
            (claims_df['ServiceDate'] <= end_date)
        ]
        
        # Calculate total claims per member in prediction window
        future_agg = future_claims.groupby('Member_ID')['TotPaymentUSD'].sum().reset_index()
        future_agg.rename(columns={'TotPaymentUSD': 'future_6m_claims'}, inplace=True)
        
        print(f"Created target variable with {len(future_agg)} members having future claims")
        return future_agg
        
    except Exception as e:
        print(f"Error creating target variable: {e}")
        return pd.DataFrame({'Member_ID': claims_df['Member_ID'].unique(), 'future_6m_claims': 0})

def train_or_load_model(features_df, force_train=False):
    """
    Train a new model or load an existing one
    """
    print("\n3. Training/loading model...")
    
    model_path = 'models/best_xgboost_model.pkl'
    
    # Check if model exists and we're not forcing a retrain
    if os.path.exists(model_path) and not force_train:
        try:
            print(f"Loading existing model from {model_path}")
            model_info = joblib.load(model_path)
            model = model_info['model']
            feature_cols = model_info.get('feature_cols', [])
            print(f"Model loaded successfully with {len(feature_cols)} features")
            return model, feature_cols
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Proceeding to train a new model...")
    
    # Train a new model
    try:
        print("Training new XGBoost model...")
        
        # Prepare data for XGBoost
        X_train, X_test, y_train, y_test, feature_cols = prepare_data_for_xgboost(features_df)
        
        # Check if X_train is a numpy array and convert to DataFrame if needed
        if isinstance(X_train, np.ndarray):
            X_train_df = pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(X_train.shape[1])])
            # Update feature_cols to match the new column names
            feature_cols = X_train_df.columns.tolist()
        else:
            X_train_df = X_train
        
        # Train the model
        model_info = train_xgboost_model(
            X_train_df, y_train, 
            test_size=0.2, 
            random_state=42, 
            optimize=True,
            n_iter=10
        )
        
        model = model_info['model']
        
        # Evaluate the model
        evaluation = evaluate_xgboost_model(
            model,
            model_info['X_test'],
            model_info['y_test']
        )
        
        # Create directory if it doesn't exist
        Path('models').mkdir(exist_ok=True)
        
        # Save the model
        joblib.dump({
            'model': model,
            'feature_cols': feature_cols,
            'date_saved': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'metrics': evaluation['metrics']
        }, model_path)
        
        print(f"Model trained and saved to {model_path}")
        print(f"Training metrics: RMSE={evaluation['metrics']['rmse']:.4f}, R²={evaluation['metrics']['r2']:.4f}")
        
        return model, feature_cols
    
    except Exception as e:
        print(f"Error training model: {e}")
        return None, []

def make_predictions(model, feature_cols, features_df):
    """
    Make predictions using the trained model
    """
    print("\n4. Making predictions...")
    
    try:
        # Prepare features in the right format - more efficient approach
        # Filter for existing columns and create those dataframe at once
        available_features = [f for f in feature_cols if f in features_df.columns]
        missing_features = [f for f in feature_cols if f not in features_df.columns]
        
        # Create DataFrame with available features
        X = features_df[available_features].copy()
        
        # Add missing features as zeros - all at once efficiently
        if missing_features:
            # Create a dictionary with zeros for each missing feature
            missing_data = {feature: [0] * len(features_df) for feature in missing_features}
            # Create DataFrame and join with existing features
            missing_df = pd.DataFrame(missing_data, index=features_df.index)
            X = pd.concat([X, missing_df], axis=1)
        
        # Convert object types to numeric and fill missing values
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = pd.to_numeric(X[col], errors='coerce')
        
        X = X.fillna(0)
        
        # Get actual values if available
        if 'future_6m_claims' in features_df.columns:
            y_true = features_df['future_6m_claims']
        else:
            y_true = pd.Series(np.zeros(len(features_df)))
            print("Warning: No actual values available for comparison")
        
        # Make predictions
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_pred = model.predict(X)
        
        print(f"Made predictions for {len(y_pred)} instances")
        
        # Calculate metrics if we have actual values
        if not (y_true == 0).all():
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            # Improved MAPE calculation
            threshold = 10.0
            mask = y_true > threshold
            
            if mask.sum() > 0:
                mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            else:
                mape = np.mean(np.abs(y_true - y_pred)) / (np.mean(y_true) + 1e-10) * 100
            
            print("\nPrediction Metrics:")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE: {mae:.4f}")
            print(f"  R²: {r2:.4f}")
            print(f"  MAPE: {mape:.4f}%")
            
            metrics = {
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'mape': mape
            }
        else:
            print("No actual values available for calculating metrics")
            metrics = {}
        
        return y_pred, y_true, metrics
    
    except Exception as e:
        print(f"Error making predictions: {e}")
        return None, None, {}

def analyze_results(y_true, y_pred, features_df, output_dir='outputs'):
    """
    Analyze prediction results and create visualizations
    """
    print("\n5. Analyzing results...")
    
    try:
        # Create output directories
        Path(f'{output_dir}/figures/predictions').mkdir(parents=True, exist_ok=True)
        Path(f'{output_dir}/tables').mkdir(parents=True, exist_ok=True)
        
        # Create visualizations
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        
        # Add perfect prediction line
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.title('Actual vs Predicted Claims')
        plt.xlabel('Actual Claims (USD)')
        plt.ylabel('Predicted Claims (USD)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(f'{output_dir}/figures/predictions/actual_vs_predicted.png')
        plt.close()
        
        # Create residual plot
        plt.figure(figsize=(10, 6))
        residuals = y_true - y_pred
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('Residual Plot')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(f'{output_dir}/figures/predictions/residual_plot.png')
        plt.close()
        
        # Create distribution of predictions
        plt.figure(figsize=(10, 6))
        sns.histplot(y_pred, kde=True)
        plt.axvline(x=np.mean(y_pred), color='r', linestyle='--', label=f'Mean: {np.mean(y_pred):.2f}')
        plt.axvline(x=np.median(y_pred), color='g', linestyle=':', label=f'Median: {np.median(y_pred):.2f}')
        plt.title('Distribution of Predicted Claims')
        plt.xlabel('Predicted Claims (USD)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig(f'{output_dir}/figures/predictions/prediction_distribution.png')
        plt.close()
        
        # Calculate and print statistics
        print("\nPrediction Statistics:")
        print(f"  Min predicted value: ${np.min(y_pred):.2f}")
        print(f"  Max predicted value: ${np.max(y_pred):.2f}")
        print(f"  Average predicted value: ${np.mean(y_pred):.2f}")
        print(f"  Median predicted value: ${np.median(y_pred):.2f}")
        
        # Check for negative predictions
        neg_preds = (y_pred < 0).sum()
        if neg_preds > 0:
            print(f"  Warning: {neg_preds} negative predictions ({neg_preds/len(y_pred)*100:.2f}% of total)")
        
        # Save predictions to CSV
        results_df = pd.DataFrame({
            'Member_ID': features_df['Member_ID'] if 'Member_ID' in features_df.columns else range(len(y_pred)),
            'Actual_Claims': y_true,
            'Predicted_Claims': y_pred,
            'Residual': y_true - y_pred
        })
        
        results_df.to_csv(f'{output_dir}/tables/prediction_results.csv', index=False)
        print(f"Results saved to {output_dir}/tables/prediction_results.csv")
        
        # Run error analysis
        if 'future_6m_claims' in features_df.columns:
            try:
                print("\nRunning detailed error analysis...")
                error_results = analyze_prediction_errors(
                    y_true, y_pred,
                    feature_matrix=features_df.drop(columns=['future_6m_claims']),
                    feature_names=features_df.drop(columns=['future_6m_claims']).columns
                )
                
                # Create regression confusion matrix
                cm, bin_edges = create_regression_confusion_matrix(
                    y_true, y_pred, n_classes=5,
                    output_path=f'{output_dir}/figures/predictions/regression_confusion_matrix.png'
                )
                
                print("Error analysis completed and visualizations saved")
            except Exception as e:
                print(f"Error in detailed error analysis: {e}")
        
        return results_df
    
    except Exception as e:
        print(f"Error analyzing results: {e}")
        return None

def run_pipeline(force_train=False, advanced_features=True, use_business_report=True):
    """
    Run the complete end-to-end prediction pipeline
    
    Parameters:
    -----------
    force_train : bool
        Whether to force training a new model even if one exists
    advanced_features : bool
        Whether to use advanced features from enhanced_feature_engineering
    use_business_report : bool
        Whether to generate a business-focused report
    """
    print("=" * 80)
    print("RUNNING END-TO-END PREDICTION PIPELINE")
    print("=" * 80)
    
    start_time = datetime.now()
    
    # Step 1: Load and preprocess data
    claims_df, members_df = load_and_preprocess_data()
    if claims_df is None or members_df is None:
        print("Error: Could not load or preprocess data. Exiting pipeline.")
        return
    
    # Step 2: Engineer features
    features_df, cutoff_date = engineer_features(claims_df, members_df)
    if features_df is None:
        print("Error: Could not engineer features. Exiting pipeline.")
        return
        
    # Step 3: Add advanced features if requested
    # This section consolidated from run_enhanced_modeling.py
    if advanced_features:
        try:
            print("\nAdding advanced features...")
            # Import modules that might not be available
            try:
                from enhanced_data_preparation import enhanced_data_preparation
                from enhanced_feature_engineering import enhanced_feature_engineering
                from advanced_temporal_features import create_advanced_temporal_features
                from enhanced_risk_scores import create_enhanced_risk_scores
                
                # Create advanced temporal features
                temporal_features = create_advanced_temporal_features(claims_df, cutoff_date)
                print(f"Created {temporal_features.shape[1]} advanced temporal features")
                
                # Create enhanced risk scores
                risk_scores = create_enhanced_risk_scores(members_df, claims_df)
                print(f"Created {risk_scores.shape[1]} risk score features")
                
                # Ensure consistent ID formats for merging
                features_df['Member_ID'] = features_df['Member_ID'].astype(str)
                temporal_features['Member_ID'] = temporal_features['Member_ID'].astype(str)
                risk_scores['Member_ID'] = risk_scores['Member_ID'].astype(str)
                
                # Merge with existing features
                features_df = pd.merge(features_df, temporal_features, on='Member_ID', how='left')
                features_df = pd.merge(features_df, risk_scores, on='Member_ID', how='left')
                
                # Apply enhanced feature engineering if available
                date_columns = ['ServiceDate', 'PolicyStartDate', 'PolicyEndDate', 'DateOfBirth']
                cleaned_claims, _, _ = enhanced_data_preparation(
                    claims_df, 
                    missing_strategy='knn',
                    outlier_method='iqr',
                    scaling_method='robust',
                    visualization=False
                )
                
                cleaned_members, _, _ = enhanced_data_preparation(
                    members_df, 
                    missing_strategy='knn',
                    outlier_method='iqr',
                    scaling_method='robust',
                    visualization=False
                )
                
                # Enhanced feature engineering
                advanced_features_df = enhanced_feature_engineering(
                    cleaned_claims, 
                    cleaned_members, 
                    date_columns=date_columns
                )
                
                # Merge these advanced features
                advanced_features_df['Member_ID'] = advanced_features_df['Member_ID'].astype(str)
                features_df = pd.merge(features_df, advanced_features_df, on='Member_ID', how='left')
                
                print(f"Total features after adding advanced features: {features_df.shape[1]}")
                
                # Save integrated features for reference
                features_df.to_csv('data/processed/integrated_features.csv', index=False)
                risk_scores.to_csv('data/processed/risk_scores.csv', index=False)
                
            except Exception as e:
                print(f"Warning: Could not add all advanced features: {e}")
                print("Continuing with standard features only.")
        except Exception as e:
            print(f"Error adding advanced features: {e}")
            print("Continuing with basic features only.")
    
    # Step 4: Train or load model
    model, feature_cols = train_or_load_model(features_df, force_train)
    if model is None:
        print("Error: Could not train or load model. Exiting pipeline.")
        return
    
    # Step 5: Make predictions
    y_pred, y_true, metrics = make_predictions(model, feature_cols, features_df)
    if y_pred is None:
        print("Error: Could not make predictions. Exiting pipeline.")
        return
    
    # Step 6: Analyze results
    results_df = analyze_results(y_true, y_pred, features_df)
    
    # Step 7: Generate business report if requested
    # This section consolidated from run_enhanced_analysis.py
    if use_business_report:
        try:
            print("\nGenerating business report...")
            
            try:
                # Import the business report module
                from advanced_business_report import run_complete_business_analysis
                
                # Load risk scores if they exist
                risk_scores_path = 'data/processed/risk_scores.csv'
                if os.path.exists(risk_scores_path):
                    risk_scores = pd.read_csv(risk_scores_path)
                else:
                    risk_scores = None
                
                # Prepare model info in the format expected by the business report
                model_info = {
                    'model': model,
                    'metrics': metrics,
                    'feature_importance': pd.DataFrame({
                        'feature': feature_cols,
                        'importance': getattr(model, 'feature_importances_', np.ones(len(feature_cols)))
                    }).sort_values('importance', ascending=False)
                }
                
                # Generate the business report
                business_analysis = run_complete_business_analysis(model_info, risk_scores)
                
                # Ensure the reports directory exists
                os.makedirs('reports', exist_ok=True)
                
                # Move the business report to reports directory
                report_source = 'advanced_business_report.md'
                report_dest = 'reports/advanced_business_report.md'
                if os.path.exists(report_source):
                    # Remove destination file if it exists to avoid errors
                    if os.path.exists(report_dest):
                        os.remove(report_dest)
                    os.rename(report_source, report_dest)
                    print(f"Business report created: {report_dest}")
                else:
                    print("Business report could not be generated")
                
            except Exception as e:
                print(f"Error generating advanced business report: {e}")
                print("Falling back to simple business report...")
                
                # Create a simple business report
                report_path = 'reports/business_report.md'
                os.makedirs('reports', exist_ok=True)
                
                with open(report_path, 'w') as f:
                    f.write("# PassportCard Insurance Claims Prediction - Business Report\n\n")
                    
                    # Executive Summary
                    f.write("## Executive Summary\n\n")
                    f.write("This report presents the results of our predictive modeling for insurance claims at PassportCard. ")
                    f.write("We've developed a model that predicts the total claim amount per customer for the next six months.\n\n")
                    
                    # Model Performance
                    f.write("## Model Performance\n\n")
                    f.write("Our model achieved the following performance metrics:\n\n")
                    f.write("| Metric | Value |\n")
                    f.write("|--------|-------|\n")
                    for metric, value in metrics.items():
                        f.write(f"| {metric} | {value:.4f} |\n")
                    
                print(f"Simple business report created: {report_path}")
        
        except Exception as e:
            print(f"Error generating business report: {e}")
    
    # Calculate and print elapsed time
    elapsed_time = datetime.now() - start_time
    print(f"\nPipeline completed in {elapsed_time.total_seconds():.2f} seconds")
    
    return {
        'model': model,
        'feature_cols': feature_cols,
        'predictions': y_pred,
        'actual': y_true,
        'metrics': metrics,
        'results_df': results_df
    }

if __name__ == "__main__":
    # Get command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description="Run the end-to-end prediction pipeline")
    parser.add_argument('--force-train', action='store_true', help='Force training a new model even if one exists')
    parser.add_argument('--basic-features', action='store_true', help='Use only basic features without advanced ones')
    parser.add_argument('--no-report', action='store_true', help='Skip generating business report')
    args = parser.parse_args()
    
    # Run the pipeline
    result = run_pipeline(
        force_train=args.force_train,
        advanced_features=not args.basic_features,
        use_business_report=not args.no_report
    ) 