import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import confusion_matrix, classification_report
import os
from scipy import stats as scipy_stats

def analyze_prediction_errors(y_true, y_pred, feature_matrix=None, feature_names=None):
    """
    Analyze prediction errors and their relationship with feature values
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    feature_matrix : pandas DataFrame, optional
        Feature matrix used for predictions
    feature_names : list, optional
        List of feature names
        
    Returns:
    --------
    dict
        Dictionary with error analysis results
    """
    # Calculate errors
    errors = y_true - y_pred
    abs_errors = np.abs(errors)
    squared_errors = errors ** 2
    
    # Basic error statistics
    error_stats = {
        'mean_error': np.mean(errors),
        'median_error': np.median(errors),
        'mean_abs_error': np.mean(abs_errors),
        'median_abs_error': np.median(abs_errors),
        'rmse': np.sqrt(np.mean(squared_errors)),
        'max_abs_error': np.max(abs_errors)
    }
    
    # Create output directory if it doesn't exist
    os.makedirs('outputs/figures/error_analysis', exist_ok=True)
    
    # Plot error distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, kde=True)
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors')
    plt.tight_layout()
    plt.savefig('outputs/figures/error_distribution.png')
    plt.close()
    
    # If feature matrix is provided, analyze errors by feature
    feature_error_corr = {}
    top_error_correlated_features = []
    error_by_feature_bins = {}
    
    if feature_matrix is not None and feature_names is not None:
        # Prepare feature matrix
        if isinstance(feature_matrix, pd.DataFrame):
            X = feature_matrix.copy()
        else:
            X = pd.DataFrame(feature_matrix, columns=feature_names)
        
        # Add errors to the feature matrix
        X['error'] = errors
        X['abs_error'] = abs_errors
        
        # Calculate correlation between features and errors
        error_corr = X.corr(numeric_only=True)['error'].drop(['error', 'abs_error']).sort_values(ascending=False)
        abs_error_corr = X.corr(numeric_only=True)['abs_error'].drop(['error', 'abs_error']).sort_values(ascending=False)
        
        feature_error_corr = {
            'error_correlation': error_corr.to_dict(),
            'abs_error_correlation': abs_error_corr.to_dict()
        }
        
        # Get top correlated features
        top_pos_corr = error_corr.head(3).index.tolist()
        top_neg_corr = error_corr.tail(3).index.tolist()
        top_abs_corr = abs_error_corr.head(5).index.tolist()
        
        top_error_correlated_features = {
            'positive_correlation': top_pos_corr,
            'negative_correlation': top_neg_corr,
            'absolute_error_correlation': top_abs_corr
        }
        
        # Analyze errors by feature bins for top correlated features
        all_top_features = list(set(top_pos_corr + top_neg_corr + top_abs_corr))
        for feature in all_top_features:
            # Create bins for the feature
            try:
                X[f'{feature}_bin'] = pd.qcut(X[feature], q=5, duplicates='drop')
                
                # Calculate mean and median errors by bin
                error_by_bin = X.groupby(f'{feature}_bin').agg({
                    'error': ['mean', 'median', 'count'],
                    'abs_error': ['mean', 'median']
                })
                
                # Convert to dictionary for easier manipulation
                error_by_feature_bins[feature] = {
                    'bin_edges': [str(b) for b in error_by_bin.index.tolist()],
                    'mean_error': error_by_bin['error']['mean'].tolist(),
                    'median_error': error_by_bin['error']['median'].tolist(),
                    'mean_abs_error': error_by_bin['abs_error']['mean'].tolist(),
                    'median_abs_error': error_by_bin['abs_error']['median'].tolist(),
                    'count': error_by_bin['error']['count'].tolist()
                }
                
                # Plot errors by feature bin
                plt.figure(figsize=(12, 8))
                sns.boxplot(x=f'{feature}_bin', y='error', data=X)
                plt.xticks(rotation=45)
                plt.xlabel(feature)
                plt.ylabel('Prediction Error')
                plt.title(f'Distribution of Errors by {feature} Bins')
                plt.tight_layout()
                plt.savefig(f'outputs/figures/error_analysis/error_by_{feature}_bins.png')
                plt.close()
            except Exception as e:
                print(f"Could not analyze errors for feature {feature}: {e}")
    
    # Return error distribution for plotting
    error_distribution = {
        'errors': errors.tolist() if hasattr(errors, 'tolist') else list(errors),
        'abs_errors': abs_errors.tolist() if hasattr(abs_errors, 'tolist') else list(abs_errors)
    }
    
    return {
        'error_stats': error_stats,
        'feature_error_corr': feature_error_corr,
        'top_error_correlated_features': top_error_correlated_features,
        'error_by_feature_bins': error_by_feature_bins,
        'error_distribution': error_distribution
    }

def create_regression_confusion_matrix(y_true, y_pred, n_classes=5, visualize=True, output_path=None):
    """
    Create a confusion matrix for regression by binning predictions and true values
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    n_classes : int
        Number of bins to create
    visualize : bool
        Whether to create a visualization
    output_path : str, optional
        Path to save the visualization
        
    Returns:
    --------
    tuple
        Tuple with confusion matrix and bin edges
    """
    # Determine bin edges
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    
    # Create bins
    bin_edges = np.linspace(min_val, max_val, n_classes + 1)
    
    # Assign each value to a bin
    y_true_bins = np.digitize(y_true, bin_edges) - 1
    y_pred_bins = np.digitize(y_pred, bin_edges) - 1
    
    # Cap at the highest bin index
    y_true_bins = np.minimum(y_true_bins, n_classes - 1)
    y_pred_bins = np.minimum(y_pred_bins, n_classes - 1)
    
    # Create confusion matrix
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for i in range(len(y_true)):
        cm[y_true_bins[i], y_pred_bins[i]] += 1
    
    # Visualize the confusion matrix
    if visualize:
        # Ensure output directory exists
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        else:
            os.makedirs('outputs/figures/confusion_matrix', exist_ok=True)
            output_path = 'outputs/figures/confusion_matrix/regression_confusion_matrix.png'
        
        # Format bin edges for display
        bin_labels = [f"{bin_edges[i]:.0f}-{bin_edges[i+1]:.0f}" for i in range(n_classes)]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=bin_labels, yticklabels=bin_labels)
        plt.xlabel('Predicted Bin')
        plt.ylabel('True Bin')
        plt.title('Regression Confusion Matrix')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    
    return cm, bin_edges

def plot_error_heatmap(y_true, y_pred, feature1, feature2, feature1_name, feature2_name, output_path=None):
    """
    Create a heatmap of prediction errors by two features
    
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
    feature1_name : str
        Name of the first feature
    feature2_name : str
        Name of the second feature
    output_path : str, optional
        Path to save the visualization
        
    Returns:
    --------
    None
    """
    # Calculate errors
    errors = y_true - y_pred
    abs_errors = np.abs(errors)
    
    # Create a DataFrame with features and errors
    df = pd.DataFrame({
        feature1_name: feature1,
        feature2_name: feature2,
        'error': errors,
        'abs_error': abs_errors
    })
    
    # Create bins for the features
    try:
        df[f'{feature1_name}_bin'] = pd.qcut(df[feature1_name], q=5, duplicates='drop')
        df[f'{feature2_name}_bin'] = pd.qcut(df[feature2_name], q=5, duplicates='drop')
        
        # Group by feature bins and calculate mean absolute error
        heatmap_data = df.groupby([f'{feature1_name}_bin', f'{feature2_name}_bin'])['abs_error'].mean().unstack()
        
        # Ensure output directory exists
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        else:
            os.makedirs('outputs/figures/error_analysis', exist_ok=True)
            output_path = f'outputs/figures/error_analysis/error_heatmap_{feature1_name}_{feature2_name}.png'
        
        # Plot heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='YlOrRd')
        plt.xlabel(feature2_name)
        plt.ylabel(feature1_name)
        plt.title(f'Mean Absolute Error by {feature1_name} and {feature2_name}')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    except Exception as e:
        print(f"Could not create error heatmap: {e}")
        
        # Try creating a scatter plot instead
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(df[feature1_name], df[feature2_name], c=df['abs_error'], cmap='YlOrRd', alpha=0.7)
        plt.colorbar(scatter, label='Absolute Error')
        plt.xlabel(feature1_name)
        plt.ylabel(feature2_name)
        plt.title(f'Absolute Error by {feature1_name} and {feature2_name}')
        
        if output_path:
            plt.savefig(output_path)
        else:
            plt.savefig(f'outputs/figures/error_analysis/error_scatter_{feature1_name}_{feature2_name}.png')
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
    print("Check the 'outputs/figures/error_analysis' and 'outputs/figures/confusion_matrix' directories for results.")

if __name__ == "__main__":
    main() 