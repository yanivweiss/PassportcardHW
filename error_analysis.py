import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import confusion_matrix, classification_report
import os

def analyze_prediction_errors(y_true, y_pred, feature_matrix=None, feature_names=None):
    """
    Comprehensive analysis of prediction errors
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    feature_matrix : array-like or DataFrame, optional
        Feature matrix for additional analysis
    feature_names : list, optional
        Names of features in feature_matrix
        
    Returns:
    --------
    dict
        Dictionary with error analysis results
    """
    # Create output directory
    os.makedirs('visualizations/error_analysis', exist_ok=True)
    
    # Basic error metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Calculate errors
    errors = y_pred - y_true
    abs_errors = np.abs(errors)
    
    # Error statistics
    error_stats = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'Mean Error': np.mean(errors),
        'Median Error': np.median(errors),
        'Min Error': np.min(errors),
        'Max Error': np.max(errors),
        'Error Std Dev': np.std(errors)
    }
    
    # Create error distribution plot
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, kde=True)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title('Distribution of Prediction Errors')
    plt.xlabel('Prediction Error (Predicted - Actual)')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('visualizations/error_analysis/error_distribution.png')
    plt.close()
    
    # Create scatter plot of actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    
    # Add perfect prediction line
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.title('Actual vs Predicted Values')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('visualizations/error_analysis/actual_vs_predicted.png')
    plt.close()
    
    # Create residual plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, errors, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Residual Plot')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residual (Predicted - Actual)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('visualizations/error_analysis/residual_plot.png')
    plt.close()
    
    # Create Q-Q plot
    plt.figure(figsize=(10, 6))
    stats = pd.Series(errors).sort_values()
    # Calculate theoretical quantiles
    n = len(stats)
    qs = np.linspace(0, 1, n+1)[1:-1]  # quantiles, excluding 0 and 1
    theoretical_quantiles = stats.mean() + stats.std() * np.sqrt(2) * np.array([stats.ppf(q) for q in qs])
    
    plt.scatter(theoretical_quantiles, stats, alpha=0.5)
    plt.plot([stats.min(), stats.max()], [stats.min(), stats.max()], 'r--')
    plt.title('Q-Q Plot of Residuals')
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Sample Quantiles')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('visualizations/error_analysis/qq_plot.png')
    plt.close()
    
    # Analyze errors by prediction magnitude
    if np.min(y_true) >= 0:  # Only for non-negative targets
        # Create bins for prediction magnitude
        bins = 5
        y_true_bins = pd.qcut(y_true, bins, labels=False, duplicates='drop')
        bin_errors = pd.DataFrame({
            'bin': y_true_bins,
            'actual': y_true,
            'predicted': y_pred,
            'error': errors,
            'abs_error': abs_errors
        })
        
        # Calculate average error by bin
        bin_stats = bin_errors.groupby('bin').agg({
            'actual': ['mean', 'count'],
            'error': ['mean', 'std'],
            'abs_error': 'mean'
        })
        
        # Flatten column names
        bin_stats.columns = ['_'.join(col).strip() for col in bin_stats.columns.values]
        
        # Create plot of error by prediction magnitude
        plt.figure(figsize=(12, 6))
        
        # Plot mean absolute error by bin
        plt.subplot(1, 2, 1)
        plt.bar(bin_stats.index, bin_stats['abs_error_mean'])
        plt.title('Mean Absolute Error by Actual Value Quantile')
        plt.xlabel('Actual Value Quantile')
        plt.ylabel('Mean Absolute Error')
        plt.xticks(bin_stats.index)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Plot mean error by bin (to show bias)
        plt.subplot(1, 2, 2)
        plt.bar(bin_stats.index, bin_stats['error_mean'])
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('Mean Error by Actual Value Quantile')
        plt.xlabel('Actual Value Quantile')
        plt.ylabel('Mean Error')
        plt.xticks(bin_stats.index)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('visualizations/error_analysis/error_by_magnitude.png')
        plt.close()
    
    # Analyze feature-error relationships if features are provided
    if feature_matrix is not None and feature_names is not None:
        # Convert feature matrix to DataFrame if it's not already
        if not isinstance(feature_matrix, pd.DataFrame):
            feature_matrix = pd.DataFrame(feature_matrix, columns=feature_names)
        
        # Add errors to feature matrix
        feature_matrix = feature_matrix.copy()
        feature_matrix['error'] = errors
        feature_matrix['abs_error'] = abs_errors
        
        # Calculate correlation between features and errors
        error_correlations = feature_matrix.corr()['error'].sort_values(ascending=False)
        abs_error_correlations = feature_matrix.corr()['abs_error'].sort_values(ascending=False)
        
        # Plot top correlated features with error
        plt.figure(figsize=(12, 10))
        
        # Error correlations
        plt.subplot(2, 1, 1)
        top_error_corr = error_correlations.drop(['error', 'abs_error']).abs().nlargest(10)
        sns.barplot(x=top_error_corr.values, y=top_error_corr.index)
        plt.title('Top 10 Features Correlated with Error')
        plt.xlabel('Absolute Correlation')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Absolute error correlations
        plt.subplot(2, 1, 2)
        top_abs_error_corr = abs_error_correlations.drop(['error', 'abs_error']).abs().nlargest(10)
        sns.barplot(x=top_abs_error_corr.values, y=top_abs_error_corr.index)
        plt.title('Top 10 Features Correlated with Absolute Error')
        plt.xlabel('Absolute Correlation')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('visualizations/error_analysis/feature_error_correlations.png')
        plt.close()
        
        # Scatter plots for top correlated features with error
        top_feature = top_error_corr.index[0]
        plt.figure(figsize=(10, 6))
        plt.scatter(feature_matrix[top_feature], errors, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title(f'Error vs {top_feature}')
        plt.xlabel(top_feature)
        plt.ylabel('Error')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(f'visualizations/error_analysis/error_vs_{top_feature}.png')
        plt.close()
    
    # Return error statistics and other results
    return {
        'error_stats': error_stats,
        'errors': errors,
        'bin_stats': bin_stats if np.min(y_true) >= 0 else None,
        'error_correlations': error_correlations if feature_matrix is not None else None,
        'abs_error_correlations': abs_error_correlations if feature_matrix is not None else None
    }

def create_regression_confusion_matrix(y_true, y_pred, n_classes=5, visualize=True):
    """
    Create a confusion matrix for regression by binning the data
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    n_classes : int
        Number of bins/classes to create
    visualize : bool
        Whether to create visualizations
        
    Returns:
    --------
    tuple
        (confusion_matrix, bin_edges)
    """
    # Create output directory if visualization is requested
    if visualize:
        os.makedirs('visualizations/confusion_matrix', exist_ok=True)
    
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Create bins
    # Use the range of true values to create bin edges
    bin_edges = np.percentile(y_true, np.linspace(0, 100, n_classes+1))
    
    # If there are duplicate bin edges (possible with discrete or skewed data), adjust
    if len(np.unique(bin_edges)) < len(bin_edges):
        bin_edges = np.linspace(np.min(y_true), np.max(y_true), n_classes+1)
    
    # Create bin labels
    bin_labels = [f'Bin {i+1}' for i in range(n_classes)]
    
    # Assign true and predicted values to bins
    y_true_binned = np.digitize(y_true, bin_edges[1:])  # returns bin indices (0 to n_classes-1)
    y_pred_binned = np.digitize(y_pred, bin_edges[1:])
    
    # Create confusion matrix
    cm = confusion_matrix(y_true_binned, y_pred_binned)
    
    # Visualize confusion matrix
    if visualize:
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=bin_labels, yticklabels=bin_labels)
        plt.title('Regression Confusion Matrix')
        plt.xlabel('Predicted Bin')
        plt.ylabel('Actual Bin')
        plt.tight_layout()
        plt.savefig('visualizations/confusion_matrix/regression_confusion_matrix.png')
        plt.close()
        
        # Create bin edge descriptions
        bin_descriptions = []
        for i in range(n_classes):
            bin_descriptions.append(f"Bin {i+1}: [{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f})")
        
        # Plot bin distribution
        plt.figure(figsize=(12, 6))
        
        # True bin distribution
        plt.subplot(1, 2, 1)
        sns.countplot(x=y_true_binned)
        plt.title('Distribution of True Values Across Bins')
        plt.xlabel('Bin')
        plt.xticks(range(n_classes), bin_labels)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Predicted bin distribution
        plt.subplot(1, 2, 2)
        sns.countplot(x=y_pred_binned)
        plt.title('Distribution of Predicted Values Across Bins')
        plt.xlabel('Bin')
        plt.xticks(range(n_classes), bin_labels)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('visualizations/confusion_matrix/bin_distributions.png')
        plt.close()
        
        # Create classification report
        report = classification_report(y_true_binned, y_pred_binned, 
                                      target_names=bin_labels, 
                                      output_dict=True)
        
        # Convert report to DataFrame for easier manipulation
        report_df = pd.DataFrame(report).transpose()
        
        # Save the classification report
        with open('visualizations/confusion_matrix/classification_report.txt', 'w') as f:
            f.write("Bin Descriptions:\n")
            for desc in bin_descriptions:
                f.write(f"{desc}\n")
            f.write("\nClassification Report:\n")
            f.write(pd.DataFrame(report).transpose().to_string())
    
    return cm, bin_edges

def plot_error_heatmap(y_true, y_pred, feature1, feature2, feature1_name=None, feature2_name=None):
    """
    Create a heatmap showing prediction errors across two feature dimensions
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    feature1 : array-like
        First feature values
    feature2 : array-like
        Second feature values
    feature1_name : str, optional
        Name of the first feature
    feature2_name : str, optional
        Name of the second feature
        
    Returns:
    --------
    None
    """
    # Create output directory
    os.makedirs('visualizations/error_analysis', exist_ok=True)
    
    # Calculate errors
    errors = y_pred - y_true
    abs_errors = np.abs(errors)
    
    # Create feature names if not provided
    if feature1_name is None:
        feature1_name = 'Feature 1'
    if feature2_name is None:
        feature2_name = 'Feature 2'
    
    # Create a DataFrame with features and errors
    df = pd.DataFrame({
        feature1_name: feature1,
        feature2_name: feature2,
        'error': errors,
        'abs_error': abs_errors
    })
    
    # Create bins for both features
    feature1_bins = pd.qcut(df[feature1_name], 5, labels=False, duplicates='drop')
    feature2_bins = pd.qcut(df[feature2_name], 5, labels=False, duplicates='drop')
    
    df['feature1_bin'] = feature1_bins
    df['feature2_bin'] = feature2_bins
    
    # Calculate average absolute error for each bin combination
    error_matrix = df.groupby(['feature1_bin', 'feature2_bin'])['abs_error'].mean().unstack()
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(error_matrix, annot=True, fmt='.2f', cmap='YlOrRd')
    plt.title(f'Mean Absolute Error by {feature1_name} and {feature2_name}')
    plt.xlabel(feature2_name + ' (binned)')
    plt.ylabel(feature1_name + ' (binned)')
    plt.tight_layout()
    plt.savefig(f'visualizations/error_analysis/error_heatmap_{feature1_name}_{feature2_name}.png')
    plt.close()
    
    # Plot bias heatmap (mean error instead of absolute error)
    bias_matrix = df.groupby(['feature1_bin', 'feature2_bin'])['error'].mean().unstack()
    
    plt.figure(figsize=(10, 8))
    # Use a diverging colormap for bias, centered at 0
    sns.heatmap(bias_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)
    plt.title(f'Mean Error (Bias) by {feature1_name} and {feature2_name}')
    plt.xlabel(feature2_name + ' (binned)')
    plt.ylabel(feature1_name + ' (binned)')
    plt.tight_layout()
    plt.savefig(f'visualizations/error_analysis/bias_heatmap_{feature1_name}_{feature2_name}.png')
    plt.close()

def main():
    """Main function to demonstrate error analysis capabilities"""
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    
    # Create features
    feature1 = np.random.normal(0, 1, n_samples)
    feature2 = np.random.normal(0, 1, n_samples)
    feature3 = np.random.normal(0, 1, n_samples)
    
    # Create target
    y_true = 2 * feature1 - 1.5 * feature2 + 0.5 * feature3 + np.random.normal(0, 1, n_samples)
    
    # Create "predictions" with some errors
    y_pred = 1.8 * feature1 - 1.3 * feature2 + 0.7 * feature3 + np.random.normal(0, 1.5, n_samples)
    
    # Create feature matrix
    feature_matrix = pd.DataFrame({
        'Feature1': feature1,
        'Feature2': feature2,
        'Feature3': feature3
    })
    
    # Run error analysis
    print("Running error analysis...")
    results = analyze_prediction_errors(
        y_true, y_pred, 
        feature_matrix=feature_matrix,
        feature_names=feature_matrix.columns
    )
    
    # Print error statistics
    print("\nError Statistics:")
    for stat, value in results['error_stats'].items():
        print(f"  {stat}: {value:.4f}")
    
    # Create regression confusion matrix
    print("\nCreating regression confusion matrix...")
    cm, bin_edges = create_regression_confusion_matrix(y_true, y_pred, n_classes=5)
    
    # Plot error heatmap
    print("\nCreating error heatmap...")
    plot_error_heatmap(y_true, y_pred, feature1, feature2, 'Feature1', 'Feature2')
    
    print("\nError analysis and visualization completed.")
    print("Check the 'visualizations/error_analysis' and 'visualizations/confusion_matrix' directories for results.")

if __name__ == "__main__":
    main() 