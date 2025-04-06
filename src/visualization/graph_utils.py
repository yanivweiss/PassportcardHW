"""
Graph utilities for creating beautiful and informative visualizations.

This module provides utilities for creating high-quality, publication-ready
visualizations for the PassportCard insurance claims prediction project.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import shap

# Set the visual style for all plots
def set_visualization_style():
    """
    Set the global visualization style for consistent, professional plots.
    """
    sns.set(style="whitegrid", font_scale=1.2)
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.titlesize'] = 18
    
    # Set the brand color palette
    colors = ["#003366", "#0077b6", "#00b4d8", "#90e0ef", "#caf0f8", "#f5f5f5", 
              "#ffcad4", "#f08080", "#e63946", "#9d0208"]
    sns.set_palette(sns.color_palette(colors))

def enhanced_correlation_heatmap(df, title="Feature Correlation Heatmap", 
                                figsize=(14, 12), output_path=None):
    """
    Create an enhanced correlation heatmap with improved readability and aesthetics.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing the features to correlate
    title : str, optional
        Title for the plot
    figsize : tuple, optional
        Figure size (width, height)
    output_path : str, optional
        Path to save the figure, if None the figure is displayed
    
    Returns:
    --------
    matplotlib Figure
        The generated figure
    """
    set_visualization_style()
    
    # Calculate correlation matrix
    corr = df.corr()
    
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Create custom diverging colormap
    cmap = LinearSegmentedColormap.from_list("blue_white_red",
                                           ["#003366", "#f5f5f5", "#9d0208"], N=256)
    
    # Draw the heatmap
    fig, ax = plt.subplots(figsize=figsize)
    heatmap = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1.0, vmin=-1.0, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .8, "label": "Correlation Coefficient"})
    
    # Improve the colorbar
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=10)
    
    # Set title and adjust layout
    plt.title(title, fontsize=18, pad=20)
    plt.tight_layout()
    
    # Save or display the figure
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return fig

def feature_importance_plot(feature_names, importance_values, title="Feature Importance", 
                          figsize=(12, 10), top_n=15, output_path=None):
    """
    Create an enhanced feature importance plot with improved readability and aesthetics.
    
    Parameters:
    -----------
    feature_names : list
        List of feature names
    importance_values : array-like
        Importance values for each feature
    title : str, optional
        Title for the plot
    figsize : tuple, optional
        Figure size (width, height)
    top_n : int, optional
        Number of top features to display
    output_path : str, optional
        Path to save the figure, if None the figure is displayed
    
    Returns:
    --------
    matplotlib Figure
        The generated figure
    """
    set_visualization_style()
    
    # Create a DataFrame for easier manipulation
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_values
    }).sort_values('Importance', ascending=False)
    
    # Get top N features
    top_importance = importance_df.head(top_n)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create a color gradient based on importance
    colors = sns.color_palette("Blues_r", n_colors=len(top_importance))
    
    # Plot horizontal bars
    bars = ax.barh(top_importance['Feature'], top_importance['Importance'], color=colors)
    
    # Add value labels at the end of each bar
    for bar in bars:
        width = bar.get_width()
        ax.text(width + width*0.01, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center', fontsize=10)
    
    # Customize the plot
    ax.set_xlabel('Importance', fontsize=14)
    ax.set_ylabel('Feature', fontsize=14)
    ax.set_title(title, fontsize=18, pad=20)
    
    # Add grid lines for easier reading
    ax.xaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or display the figure
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return fig

def distribution_comparison_plot(actual, predicted, title="Actual vs Predicted Distribution", 
                               figsize=(14, 8), output_path=None):
    """
    Create a comparison plot between actual and predicted distributions.
    
    Parameters:
    -----------
    actual : array-like
        Actual values
    predicted : array-like
        Predicted values
    title : str, optional
        Title for the plot
    figsize : tuple, optional
        Figure size (width, height)
    output_path : str, optional
        Path to save the figure, if None the figure is displayed
    
    Returns:
    --------
    matplotlib Figure
        The generated figure
    """
    set_visualization_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the histograms
    sns.histplot(actual, color="#003366", label="Actual", alpha=0.7, kde=True, ax=ax)
    sns.histplot(predicted, color="#e63946", label="Predicted", alpha=0.7, kde=True, ax=ax)
    
    # Add mean lines
    plt.axvline(np.mean(actual), color="#003366", linestyle='dashed', linewidth=2, 
                label=f"Actual Mean: {np.mean(actual):.2f}")
    plt.axvline(np.mean(predicted), color="#e63946", linestyle='dashed', linewidth=2, 
                label=f"Predicted Mean: {np.mean(predicted):.2f}")
    
    # Customize the plot
    ax.set_xlabel("Value", fontsize=14)
    ax.set_ylabel("Frequency", fontsize=14)
    ax.set_title(title, fontsize=18, pad=20)
    
    # Add statistical information in a text box
    textstr = f"""
    Statistics:
    Actual - Mean: {np.mean(actual):.2f}, Median: {np.median(actual):.2f}, Std: {np.std(actual):.2f}
    Predicted - Mean: {np.mean(predicted):.2f}, Median: {np.median(predicted):.2f}, Std: {np.std(predicted):.2f}
    """
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    
    # Add legend
    ax.legend(loc='upper right', fontsize=12)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or display the figure
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return fig

def predictions_vs_actual_plot(y_true, y_pred, title="Predictions vs. Actual Values", 
                             figsize=(12, 10), output_path=None):
    """
    Create an enhanced scatter plot of predicted vs actual values.
    
    Parameters:
    -----------
    y_true : array-like
        Actual values
    y_pred : array-like
        Predicted values
    title : str, optional
        Title for the plot
    figsize : tuple, optional
        Figure size (width, height)
    output_path : str, optional
        Path to save the figure, if None the figure is displayed
    
    Returns:
    --------
    matplotlib Figure
        The generated figure
    """
    set_visualization_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create the scatter plot with a color gradient based on density
    scatter = ax.scatter(y_true, y_pred, alpha=0.6, edgecolor='w', linewidth=0.5, 
               c=np.abs(y_true - y_pred), cmap='Blues_r')
    
    # Add the perfect prediction line
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.plot(lims, lims, 'r-', alpha=0.7, zorder=0, label="Perfect Prediction")
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    
    # Add color bar to show error magnitude
    cbar = plt.colorbar(scatter)
    cbar.set_label('Absolute Error', rotation=270, labelpad=20)
    
    # Calculate metrics
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    
    # Add metrics to the plot
    textstr = f"""
    Metrics:
    RMSE: {rmse:.2f}
    MAE: {mae:.2f}
    R²: {r2:.3f}
    """
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    
    # Customize the plot
    ax.set_xlabel('Actual Values', fontsize=14)
    ax.set_ylabel('Predicted Values', fontsize=14)
    ax.set_title(title, fontsize=18, pad=20)
    ax.legend(loc='lower right')
    
    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or display the figure
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return fig

def error_distribution_plot(y_true, y_pred, title="Error Distribution", 
                          figsize=(14, 10), output_path=None):
    """
    Create a comprehensive error analysis plot.
    
    Parameters:
    -----------
    y_true : array-like
        Actual values
    y_pred : array-like
        Predicted values
    title : str, optional
        Title for the plot
    figsize : tuple, optional
        Figure size (width, height)
    output_path : str, optional
        Path to save the figure, if None the figure is displayed
    
    Returns:
    --------
    matplotlib Figure
        The generated figure
    """
    set_visualization_style()
    
    # Calculate errors
    errors = y_true - y_pred
    
    # Create the figure with 2 subplots in a grid
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1])
    
    # Error histogram with KDE
    ax0 = plt.subplot(gs[0, 0])
    sns.histplot(errors, kde=True, ax=ax0, color="#0077b6")
    ax0.axvline(0, color='r', linestyle='--')
    ax0.set_title("Error Distribution", fontsize=16)
    ax0.set_xlabel("Error (Actual - Predicted)", fontsize=14)
    ax0.set_ylabel("Frequency", fontsize=14)
    
    # Add statistical information
    mean_error = np.mean(errors)
    median_error = np.median(errors)
    std_error = np.std(errors)
    
    textstr = f"""
    Error Statistics:
    Mean: {mean_error:.2f}
    Median: {median_error:.2f}
    Std Dev: {std_error:.2f}
    """
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax0.text(0.05, 0.95, textstr, transform=ax0.transAxes, fontsize=12,
             verticalalignment='top', bbox=props)
    
    # Error vs Predicted scatter plot
    ax1 = plt.subplot(gs[0, 1])
    scatter = ax1.scatter(y_pred, errors, alpha=0.6, edgecolor='w', linewidth=0.5, 
                         c=y_true, cmap='viridis')
    ax1.axhline(0, color='r', linestyle='--')
    ax1.set_title("Error vs Predicted Values", fontsize=16)
    ax1.set_xlabel("Predicted Values", fontsize=14)
    ax1.set_ylabel("Error (Actual - Predicted)", fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Add color bar to show actual values
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Actual Value', rotation=270, labelpad=20)
    
    # Error Quantile Plot
    ax2 = plt.subplot(gs[1, :])
    
    # Sort errors and calculate quantiles
    sorted_errors = np.sort(errors)
    quantiles = np.linspace(0, 1, len(sorted_errors))
    
    # Plot quantiles
    ax2.plot(quantiles, sorted_errors, 'b-')
    ax2.axhline(0, color='r', linestyle='--')
    
    # Add reference lines at 25%, 50%, and 75% quantiles
    q25 = np.percentile(errors, 25)
    q50 = np.percentile(errors, 50)
    q75 = np.percentile(errors, 75)
    
    ax2.axvline(0.25, color='gray', linestyle=':', alpha=0.7)
    ax2.axvline(0.5, color='gray', linestyle=':', alpha=0.7)
    ax2.axvline(0.75, color='gray', linestyle=':', alpha=0.7)
    
    ax2.axhline(q25, color='gray', linestyle=':', alpha=0.7)
    ax2.axhline(q50, color='gray', linestyle=':', alpha=0.7)
    ax2.axhline(q75, color='gray', linestyle=':', alpha=0.7)
    
    ax2.text(0.25, ax2.get_ylim()[0], '25%', ha='center', va='bottom')
    ax2.text(0.5, ax2.get_ylim()[0], '50%', ha='center', va='bottom')
    ax2.text(0.75, ax2.get_ylim()[0], '75%', ha='center', va='bottom')
    
    ax2.set_title("Error Quantile Plot", fontsize=16)
    ax2.set_xlabel("Quantile", fontsize=14)
    ax2.set_ylabel("Error Value", fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Add overall title
    plt.suptitle(title, fontsize=18, y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save or display the figure
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return fig 