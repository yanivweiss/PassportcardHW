"""
Fairness analysis module for insurance claim predictions

This module provides tools for analyzing model fairness across different demographic groups,
calculating fairness metrics, and visualizing potential biases in model predictions.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

def calculate_fairness_metrics_regression(y_true, y_pred, group_membership, group_name='Group'):
    """
    Calculate fairness metrics for regression models across different demographic groups.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    group_membership : array-like
        Group membership indicator (e.g., gender, age group)
    group_name : str
        Name of the group variable for plotting and reporting
        
    Returns:
    --------
    dict
        Dictionary containing fairness metrics by group
    """
    # Create a dataframe with all the data
    df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'group': group_membership,
        'error': y_pred - y_true,
        'abs_error': np.abs(y_pred - y_true),
        'squared_error': (y_pred - y_true) ** 2
    })
    
    # Calculate metrics by group
    group_metrics = df.groupby('group').agg({
        'y_true': ['count', 'mean'],
        'y_pred': ['mean'],
        'error': ['mean', 'median'],  # Bias measurement
        'abs_error': ['mean', 'median'],  # Error magnitude
        'squared_error': ['mean']  # For RMSE
    })
    
    # Flatten column names
    group_metrics.columns = ['_'.join(col).strip() for col in group_metrics.columns.values]
    
    # Calculate RMSE for each group
    group_metrics['rmse'] = np.sqrt(group_metrics['squared_error_mean'])
    
    # Calculate R² for each group
    r2_by_group = {}
    for group in df['group'].unique():
        group_data = df[df['group'] == group]
        r2_by_group[group] = r2_score(group_data['y_true'], group_data['y_pred'])
    
    group_metrics['r2'] = pd.Series(r2_by_group)
    
    # Calculate demographic parity (difference in mean predictions)
    overall_mean_pred = df['y_pred'].mean()
    group_metrics['demographic_parity_diff'] = group_metrics['y_pred_mean'] - overall_mean_pred
    
    # Calculate prediction proportionality (ratio of mean prediction to mean actual)
    group_metrics['prediction_proportionality'] = group_metrics['y_pred_mean'] / group_metrics['y_true_mean']
    
    # Create visualizations directory if it doesn't exist
    os.makedirs('visualizations/fairness', exist_ok=True)
    
    # Visualize error distribution by group
    plt.figure(figsize=(12, 8))
    
    # Plot error distribution
    plt.subplot(2, 2, 1)
    sns.boxplot(x='group', y='error', data=df)
    plt.title(f'Error Distribution by {group_name}')
    plt.axhline(y=0, linestyle='--', color='r')
    
    # Plot RMSE by group
    plt.subplot(2, 2, 2)
    sns.barplot(x=group_metrics.index, y=group_metrics['rmse'])
    plt.title(f'RMSE by {group_name}')
    
    # Plot demographic parity
    plt.subplot(2, 2, 3)
    sns.barplot(x=group_metrics.index, y=group_metrics['demographic_parity_diff'])
    plt.title(f'Demographic Parity Difference by {group_name}')
    plt.axhline(y=0, linestyle='--', color='r')
    
    # Plot prediction proportionality
    plt.subplot(2, 2, 4)
    sns.barplot(x=group_metrics.index, y=group_metrics['prediction_proportionality'])
    plt.title(f'Prediction Proportionality by {group_name}')
    plt.axhline(y=1, linestyle='--', color='r')
    
    plt.tight_layout()
    plt.savefig(f'visualizations/fairness/fairness_metrics_{group_name.lower()}.png')
    plt.close()
    
    return group_metrics

def calculate_disparate_impact(df, group_col, threshold_col, threshold, positive_label=1):
    """
    Calculate disparate impact for a given threshold.
    
    For regression problems, we first convert to a binary outcome based on the threshold,
    then calculate the ratio of positive predictions for different groups.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing prediction data
    group_col : str
        Column name with group membership
    threshold_col : str
        Column name with prediction values to threshold
    threshold : float
        Threshold value to convert regression to binary
    positive_label : int or str
        Value representing the positive class
        
    Returns:
    --------
    dict
        Dictionary with disparate impact metrics
    """
    # Create binary outcome based on threshold
    df = df.copy()
    df['threshold_result'] = (df[threshold_col] >= threshold).astype(int)
    
    # Calculate positive rates by group
    group_rates = df.groupby(group_col)['threshold_result'].mean()
    
    # Get reference group (usually the privileged group or group with highest rate)
    reference_group = group_rates.idxmax()
    reference_rate = group_rates[reference_group]
    
    # Calculate disparate impact for each group compared to reference
    disparate_impact = {}
    for group in group_rates.index:
        if group != reference_group and reference_rate > 0:
            disparate_impact[group] = group_rates[group] / reference_rate
        elif group == reference_group:
            disparate_impact[group] = 1.0
    
    # Create results dictionary
    results = {
        'group_rates': group_rates,
        'reference_group': reference_group,
        'disparate_impact': disparate_impact,
        'overall_rate': df['threshold_result'].mean()
    }
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    
    # Plot positive rates by group
    plt.subplot(1, 2, 1)
    sns.barplot(x=group_rates.index, y=group_rates.values)
    plt.title(f'Positive Rate by Group\n(threshold={threshold})')
    plt.axhline(y=results['overall_rate'], linestyle='--', color='r', label='Overall')
    plt.legend()
    
    # Plot disparate impact
    plt.subplot(1, 2, 2)
    di_values = pd.Series(disparate_impact)
    sns.barplot(x=di_values.index, y=di_values.values)
    plt.title(f'Disparate Impact Ratio\n(relative to {reference_group})')
    plt.axhline(y=0.8, linestyle='--', color='r', label='0.8 threshold')
    plt.axhline(y=1.2, linestyle='--', color='r')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'visualizations/fairness/disparate_impact_{group_col.lower()}.png')
    plt.close()
    
    return results

def calculate_calibration_by_group(y_true, y_pred, group_membership, n_bins=10):
    """
    Calculate and visualize prediction calibration by group.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    group_membership : array-like
        Group membership indicator (e.g., gender, age group)
    n_bins : int
        Number of bins for calibration curve
        
    Returns:
    --------
    dict
        Dictionary with calibration curves by group
    """
    # Create a dataframe
    df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'group': group_membership
    })
    
    # Calculate calibration curves for each group
    calibration_curves = {}
    for group in df['group'].unique():
        group_data = df[df['group'] == group].copy()
        
        # Create bins based on predicted values
        group_data['bin'] = pd.qcut(group_data['y_pred'], n_bins, labels=False, duplicates='drop')
        
        # Calculate mean prediction and actual for each bin
        cal_curve = group_data.groupby('bin').agg({
            'y_pred': 'mean',
            'y_true': 'mean'
        }).reset_index()
        
        calibration_curves[group] = cal_curve
    
    # Visualize calibration curves
    plt.figure(figsize=(10, 6))
    
    # Get min and max values for plot
    y_min = min(df['y_true'].min(), df['y_pred'].min())
    y_max = max(df['y_true'].max(), df['y_pred'].max())
    
    # Plot perfect calibration line
    plt.plot([y_min, y_max], [y_min, y_max], 'k--', label='Perfect calibration')
    
    # Plot calibration curves for each group
    for group, curve in calibration_curves.items():
        plt.scatter(curve['y_pred'], curve['y_true'], label=f'Group {group}', alpha=0.7)
        plt.plot(curve['y_pred'], curve['y_true'], '-o', alpha=0.5)
    
    plt.xlabel('Mean predicted value')
    plt.ylabel('Mean actual value')
    plt.title('Calibration Curves by Group')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('visualizations/fairness/calibration_by_group.png')
    plt.close()
    
    return calibration_curves

def audit_model_performance(model, X, y_true, group_col, group_data=None, continuous_cols=None):
    """
    Comprehensive audit of model performance across different population segments.
    
    Parameters:
    -----------
    model : estimator
        Trained model with predict method
    X : pandas DataFrame
        Feature matrix
    y_true : array-like
        True target values
    group_col : str
        Column name for grouping (for fairness analysis)
    group_data : pandas DataFrame, optional
        Additional data with group information if not in X
    continuous_cols : list, optional
        List of continuous features to analyze performance against
        
    Returns:
    --------
    dict
        Dictionary with comprehensive audit results
    """
    # Create predictions
    y_pred = model.predict(X)
    
    # Get group data
    if group_data is not None and group_col in group_data.columns:
        if 'Member_ID' in X.columns and 'Member_ID' in group_data.columns:
            # Merge group information if needs to be joined
            merged_data = pd.merge(
                pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'Member_ID': X['Member_ID']}),
                group_data[['Member_ID', group_col]],
                on='Member_ID',
                how='left'
            )
            group_membership = merged_data[group_col]
        else:
            # If we can't merge, return without group analysis
            print(f"Warning: Cannot merge group data - Member_ID not found in both dataframes")
            group_membership = None
    elif group_col in X.columns:
        # Group information is already in X
        group_membership = X[group_col]
    else:
        # No group information available
        print(f"Warning: Group column '{group_col}' not found in data")
        group_membership = None
    
    # Calculate overall metrics
    overall_metrics = {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }
    
    results = {
        'overall_metrics': overall_metrics,
        'predictions': y_pred
    }
    
    # Fairness analysis by group if available
    if group_membership is not None:
        results['fairness_metrics'] = calculate_fairness_metrics_regression(
            y_true, y_pred, group_membership, group_name=group_col
        )
        
        # Create dataframe for disparate impact analysis
        impact_df = pd.DataFrame({
            'y_true': y_true,
            'y_pred': y_pred,
            group_col: group_membership
        })
        
        # Calculate disparate impact at multiple thresholds
        thresholds = [np.percentile(y_pred, pct) for pct in [50, 75, 90]]
        results['disparate_impact'] = {}
        for threshold in thresholds:
            results['disparate_impact'][threshold] = calculate_disparate_impact(
                impact_df, group_col, 'y_pred', threshold
            )
        
        # Calculate calibration curves
        results['calibration'] = calculate_calibration_by_group(
            y_true, y_pred, group_membership
        )
    
    # Performance analysis across continuous features
    if continuous_cols:
        results['continuous_performance'] = {}
        
        for col in continuous_cols:
            if col in X.columns:
                # Create binned version of the continuous column
                X_copy = X.copy()
                X_copy['pred'] = y_pred
                X_copy['actual'] = y_true
                X_copy['error'] = X_copy['pred'] - X_copy['actual']
                X_copy['abs_error'] = np.abs(X_copy['error'])
                
                # Create bins for the continuous column
                X_copy['bin'] = pd.qcut(X_copy[col], 5, labels=False, duplicates='drop')
                
                # Calculate metrics by bin
                bin_metrics = X_copy.groupby('bin').agg({
                    col: 'mean',
                    'pred': 'mean',
                    'actual': 'mean',
                    'error': 'mean',
                    'abs_error': 'mean'
                })
                
                # Calculate RMSE for each bin
                bin_rmse = []
                for bin_val in X_copy['bin'].unique():
                    bin_data = X_copy[X_copy['bin'] == bin_val]
                    bin_rmse.append(np.sqrt(mean_squared_error(bin_data['actual'], bin_data['pred'])))
                
                bin_metrics['rmse'] = bin_rmse
                
                # Plot error vs. continuous feature
                plt.figure(figsize=(12, 8))
                
                # Plot error vs. feature value
                plt.subplot(2, 2, 1)
                plt.scatter(X_copy[col], X_copy['error'], alpha=0.3)
                plt.axhline(y=0, linestyle='--', color='r')
                plt.xlabel(col)
                plt.ylabel('Error (Predicted - Actual)')
                plt.title(f'Error vs. {col}')
                
                # Plot absolute error by bin
                plt.subplot(2, 2, 2)
                sns.barplot(x=bin_metrics.index, y=bin_metrics['abs_error'])
                plt.xlabel(f'{col} (binned)')
                plt.ylabel('Mean Absolute Error')
                plt.title(f'MAE by {col} Bin')
                
                # Plot RMSE by bin
                plt.subplot(2, 2, 3)
                plt.bar(bin_metrics.index, bin_metrics['rmse'])
                plt.xlabel(f'{col} (binned)')
                plt.ylabel('RMSE')
                plt.title(f'RMSE by {col} Bin')
                
                # Plot mean prediction vs actual by bin
                plt.subplot(2, 2, 4)
                plt.plot(bin_metrics.index, bin_metrics['actual'], 'o-', label='Actual')
                plt.plot(bin_metrics.index, bin_metrics['pred'], 'o-', label='Predicted')
                plt.xlabel(f'{col} (binned)')
                plt.ylabel('Mean Value')
                plt.title(f'Prediction vs Actual by {col} Bin')
                plt.legend()
                
                plt.tight_layout()
                plt.savefig(f'visualizations/fairness/performance_by_{col}.png')
                plt.close()
                
                results['continuous_performance'][col] = bin_metrics
    
    return results

def generate_fairness_report(audit_results, output_path='reports/fairness_report.md'):
    """
    Generate a comprehensive fairness report based on audit results.
    
    Parameters:
    -----------
    audit_results : dict
        Results from the model audit
    output_path : str
        Path to save the report
        
    Returns:
    --------
    str
        Path to the generated report
    """
    # Create reports directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Start building the report
    report = [
        "# Model Fairness and Performance Audit Report\n",
        "## Overall Model Performance\n"
    ]
    
    # Add overall metrics
    overall_metrics = audit_results.get('overall_metrics', {})
    report.append("| Metric | Value |\n| ------ | ----- |")
    for metric, value in overall_metrics.items():
        report.append(f"| {metric.upper()} | {value:.4f} |")
    
    # Add fairness metrics if available
    fairness_metrics = audit_results.get('fairness_metrics')
    if fairness_metrics is not None:
        report.append("\n## Fairness Metrics by Group\n")
        
        # Convert to Markdown table
        metrics_table = fairness_metrics.reset_index()
        cols = metrics_table.columns.tolist()
        
        # Create table header
        report.append("| " + " | ".join(cols) + " |")
        report.append("| " + " | ".join(["---"] * len(cols)) + " |")
        
        # Add rows
        for _, row in metrics_table.iterrows():
            row_str = "| " + " | ".join([f"{val:.4f}" if isinstance(val, float) else str(val) for val in row]) + " |"
            report.append(row_str)
    
    # Add disparate impact analysis if available
    disparate_impact = audit_results.get('disparate_impact')
    if disparate_impact:
        report.append("\n## Disparate Impact Analysis\n")
        
        for threshold, impact in disparate_impact.items():
            report.append(f"\n### Threshold: {threshold:.2f}\n")
            
            # Add group rates
            report.append("#### Positive Rate by Group:")
            report.append("| Group | Rate |")
            report.append("| ----- | ---- |")
            for group, rate in impact['group_rates'].items():
                report.append(f"| {group} | {rate:.4f} |")
            
            # Add disparate impact ratios
            report.append("\n#### Disparate Impact Ratio (relative to reference group):")
            report.append("| Group | Ratio | Status |")
            report.append("| ----- | ----- | ------ |")
            for group, ratio in impact['disparate_impact'].items():
                # Replace these Unicode characters with ASCII equivalents
                status = "FAIR" if 0.8 <= ratio <= 1.2 else "POTENTIAL BIAS"
                report.append(f"| {group} | {ratio:.4f} | {status} |")
    
    # Add information about continuous features if available
    continuous_performance = audit_results.get('continuous_performance')
    if continuous_performance:
        report.append("\n## Performance Across Continuous Features\n")
        
        for feature, metrics in continuous_performance.items():
            report.append(f"\n### Performance by {feature}\n")
            
            # Convert to Markdown table
            metrics_table = metrics.reset_index()
            cols = metrics_table.columns.tolist()
            
            # Create table header
            report.append("| " + " | ".join(cols) + " |")
            report.append("| " + " | ".join(["---"] * len(cols)) + " |")
            
            # Add rows
            for _, row in metrics_table.iterrows():
                row_str = "| " + " | ".join([f"{val:.4f}" if isinstance(val, float) else str(val) for val in row]) + " |"
                report.append(row_str)
            
            # Add observations about patterns
            error_pattern = metrics['error'].std()
            if error_pattern > 0.5 * metrics['error'].mean():
                report.append("\n**Observation**: Error distribution varies significantly across different " +
                         f"values of {feature}, suggesting potential bias.")
            else:
                report.append("\n**Observation**: Error distribution is relatively consistent across different " +
                         f"values of {feature}.")
    
    # Add summary and recommendations
    report.append("\n## Summary and Recommendations\n")
    
    # Check for potential fairness issues
    has_fairness_issues = False
    fairness_issues = []
    
    if fairness_metrics is not None:
        # Check for demographic parity issues
        for group, metrics in fairness_metrics.iterrows():
            if abs(metrics['demographic_parity_diff']) > 0.2 * overall_metrics['rmse']:
                has_fairness_issues = True
                fairness_issues.append(f"- Predictions for group '{group}' show potential bias " +
                                    f"(demographic parity difference: {metrics['demographic_parity_diff']:.4f})")
    
    if disparate_impact:
        # Check for disparate impact issues
        for threshold, impact in disparate_impact.items():
            for group, ratio in impact['disparate_impact'].items():
                if ratio < 0.8 or ratio > 1.2:
                    has_fairness_issues = True
                    fairness_issues.append(f"- At threshold {threshold:.2f}, group '{group}' experiences " +
                                        f"disparate impact (ratio: {ratio:.4f})")
    
    if continuous_performance:
        # Check for varying performance across continuous features
        for feature, metrics in continuous_performance.items():
            if metrics['rmse'].std() > 0.5 * metrics['rmse'].mean():
                has_fairness_issues = True
                fairness_issues.append(f"- Performance varies significantly across different values of {feature}")
    
    # Add fairness assessment
    if has_fairness_issues:
        report.append("### Fairness Assessment: POTENTIAL FAIRNESS ISSUES DETECTED\n")
        report.append("The following potential fairness issues were identified:\n")
        for issue in fairness_issues:
            report.append(issue)
    else:
        report.append("### Fairness Assessment: NO SIGNIFICANT FAIRNESS ISSUES DETECTED\n")
        report.append("The model performs consistently across different groups and feature values.")
    
    # Add recommendations
    report.append("\n### Recommendations:\n")
    
    if has_fairness_issues:
        report.append("1. **Bias Mitigation**: Consider applying bias mitigation techniques such as:")
        report.append("   - Reweighting samples to balance representation")
        report.append("   - Applying fairness constraints during model training")
        report.append("   - Developing separate models for different groups if appropriate")
        report.append("\n2. **Data Collection**: Collect more data from underrepresented groups")
        report.append("\n3. **Feature Engineering**: Revisit feature engineering to reduce bias")
    else:
        report.append("1. **Regular Monitoring**: Continue to monitor model fairness as new data comes in")
        report.append("\n2. **Expanded Analysis**: Consider analyzing fairness across additional demographic dimensions")
        report.append("\n3. **Feedback Collection**: Establish mechanisms to collect feedback from users about model fairness")
    
    # Write the report to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))
    
    return output_path

def main():
    """
    Demo function to show how to use the fairness analysis module
    """
    import numpy as np
    
    # Create synthetic data with two groups
    np.random.seed(42)
    n_samples = 1000
    
    # Group membership
    group = np.random.choice(['A', 'B'], n_samples)
    
    # Create biased data - Group B has higher targets on average
    y_true = np.zeros(n_samples)
    y_true[group == 'A'] = np.random.normal(100, 30, sum(group == 'A'))
    y_true[group == 'B'] = np.random.normal(150, 40, sum(group == 'B'))
    
    # Create biased predictions - under-predict for group B
    y_pred = np.zeros(n_samples)
    y_pred[group == 'A'] = y_true[group == 'A'] + np.random.normal(0, 20, sum(group == 'A'))
    y_pred[group == 'B'] = y_true[group == 'B'] * 0.8 + np.random.normal(0, 30, sum(group == 'B'))
    
    # Create a continuous feature correlated with the target
    feature = np.zeros(n_samples)
    feature[group == 'A'] = y_true[group == 'A'] / 10 + np.random.normal(0, 2, sum(group == 'A'))
    feature[group == 'B'] = y_true[group == 'B'] / 12 + np.random.normal(0, 3, sum(group == 'B'))
    
    # Create data for audit
    X = pd.DataFrame({
        'feature': feature,
        'group': group,
        'Member_ID': range(n_samples)
    })
    
    # Create a dummy model
    class DummyModel:
        def predict(self, X):
            return y_pred
    
    model = DummyModel()
    
    # Run audit
    audit_results = audit_model_performance(
        model, X, y_true, 'group', 
        continuous_cols=['feature']
    )
    
    # Generate report
    report_path = generate_fairness_report(audit_results)
    print(f"Report generated: {report_path}")

if __name__ == '__main__':
    main() 