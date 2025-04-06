"""
SHAP visualization utilities for model explainability.

This module provides enhanced SHAP visualization utilities to create beautiful
and informative explainability visualizations for the insurance claims model.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec

def enhanced_shap_summary_plot(shap_values, features, feature_names=None, 
                             title="SHAP Feature Impact Summary", 
                             figsize=(12, 10), output_path=None):
    """
    Create an enhanced SHAP summary plot with improved aesthetics and readability.
    
    Parameters:
    -----------
    shap_values : numpy array
        SHAP values for all instances and features
    features : pandas DataFrame
        Feature values for all instances
    feature_names : list, optional
        List of feature names (if features is not a DataFrame)
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
    # Convert features to DataFrame if needed
    if not isinstance(features, pd.DataFrame) and feature_names is not None:
        features = pd.DataFrame(features, columns=feature_names)
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Create SHAP summary plot
    shap.summary_plot(shap_values, features, show=False, plot_size=figsize)
    
    # Customize the plot
    plt.title(title, fontsize=18, pad=20)
    plt.tight_layout()
    
    # Save figure if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return plt.gcf()

def enhanced_shap_waterfall_plot(explainer, shap_values, features, instance_idx, 
                               feature_names=None, max_display=10,
                               title="SHAP Waterfall Plot for Single Prediction", 
                               figsize=(12, 10), output_path=None):
    """
    Create an enhanced SHAP waterfall plot for a single prediction.
    
    Parameters:
    -----------
    explainer : shap.Explainer
        SHAP explainer object
    shap_values : numpy array
        SHAP values for all instances and features
    features : pandas DataFrame or numpy array
        Feature values for all instances
    instance_idx : int
        Index of the instance to explain
    feature_names : list, optional
        List of feature names (if features is not a DataFrame)
    max_display : int, optional
        Maximum number of features to display
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
    # Convert to DataFrame if needed
    if not isinstance(features, pd.DataFrame) and feature_names is not None:
        features = pd.DataFrame(features, columns=feature_names)
    
    # Get the instance data
    if isinstance(features, pd.DataFrame):
        instance = features.iloc[instance_idx:instance_idx+1]
    else:
        instance = features[instance_idx:instance_idx+1]
    
    # Get SHAP values for the instance
    if isinstance(shap_values, list):
        # For multi-output models
        instance_shap_values = shap_values[0][instance_idx]
        expected_value = explainer.expected_value[0]
    else:
        instance_shap_values = shap_values[instance_idx]
        expected_value = explainer.expected_value
    
    # Create a SHAP Explanation object
    if isinstance(features, pd.DataFrame):
        feature_names = features.columns
    
    explanation = shap.Explanation(
        values=instance_shap_values,
        base_values=expected_value,
        data=instance.values[0],
        feature_names=feature_names
    )
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Create waterfall plot
    shap.plots.waterfall(explanation, max_display=max_display, show=False)
    
    # Customize the plot
    plt.title(title, fontsize=18, pad=20)
    plt.tight_layout()
    
    # Save figure if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return plt.gcf()

def enhanced_shap_force_plot(explainer, shap_values, features, instance_idx, 
                           feature_names=None, title="SHAP Force Plot", 
                           figsize=(20, 3), output_path=None):
    """
    Create an enhanced SHAP force plot for a single prediction.
    
    Parameters:
    -----------
    explainer : shap.Explainer
        SHAP explainer object
    shap_values : numpy array
        SHAP values for all instances and features
    features : pandas DataFrame or numpy array
        Feature values for all instances
    instance_idx : int
        Index of the instance to explain
    feature_names : list, optional
        List of feature names (if features is not a DataFrame)
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
    # Convert to DataFrame if needed
    if not isinstance(features, pd.DataFrame) and feature_names is not None:
        features = pd.DataFrame(features, columns=feature_names)
    
    # Get the instance data
    if isinstance(features, pd.DataFrame):
        instance = features.iloc[instance_idx:instance_idx+1]
    else:
        instance = features[instance_idx:instance_idx+1]
    
    # Get SHAP values for the instance
    if isinstance(shap_values, list):
        # For multi-output models
        instance_shap_values = shap_values[0][instance_idx:instance_idx+1]
        expected_value = explainer.expected_value[0]
    else:
        instance_shap_values = shap_values[instance_idx:instance_idx+1]
        expected_value = explainer.expected_value
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Create force plot
    shap.force_plot(
        expected_value, 
        instance_shap_values, 
        instance, 
        matplotlib=True, 
        show=False,
        text_rotation=45  # Rotate feature text for better readability
    )
    
    # Customize the plot
    plt.title(title, fontsize=18, pad=10)
    plt.tight_layout()
    
    # Save figure if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return plt.gcf()

def enhanced_shap_dependence_plot(shap_values, features, feature_idx, feature_names=None,
                                interaction_idx=None, title=None, figsize=(12, 8),
                                output_path=None):
    """
    Create an enhanced SHAP dependence plot for a specific feature.
    
    Parameters:
    -----------
    shap_values : numpy array
        SHAP values for all instances and features
    features : pandas DataFrame or numpy array
        Feature values for all instances
    feature_idx : int or str
        Index or name of the feature to plot
    feature_names : list, optional
        List of feature names (if features is not a DataFrame)
    interaction_idx : int or str, optional
        Index or name of the feature to plot interaction with
    title : str, optional
        Title for the plot (if None, automatically generated)
    figsize : tuple, optional
        Figure size (width, height)
    output_path : str, optional
        Path to save the figure, if None the figure is displayed
    
    Returns:
    --------
    matplotlib Figure
        The generated figure
    """
    # Convert to DataFrame if needed
    if not isinstance(features, pd.DataFrame) and feature_names is not None:
        features = pd.DataFrame(features, columns=feature_names)
    
    # Get feature name
    if isinstance(features, pd.DataFrame):
        if isinstance(feature_idx, int):
            feature_name = features.columns[feature_idx]
        else:
            feature_name = feature_idx
            feature_idx = list(features.columns).index(feature_name)
    else:
        feature_name = feature_names[feature_idx] if feature_names is not None else f"Feature {feature_idx}"
    
    # Create title if not provided
    if title is None:
        if interaction_idx is not None:
            if isinstance(interaction_idx, int) and isinstance(features, pd.DataFrame):
                interaction_name = features.columns[interaction_idx]
            elif isinstance(interaction_idx, str):
                interaction_name = interaction_idx
            else:
                interaction_name = f"Feature {interaction_idx}"
            title = f"SHAP Dependence Plot: {feature_name} (colored by {interaction_name})"
        else:
            title = f"SHAP Dependence Plot: {feature_name}"
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Create dependence plot
    shap.dependence_plot(
        feature_idx, 
        shap_values, 
        features, 
        interaction_index=interaction_idx,
        show=False
    )
    
    # Customize the plot
    plt.title(title, fontsize=18, pad=20)
    plt.xlabel(feature_name, fontsize=14)
    plt.ylabel(f"SHAP value for {feature_name}", fontsize=14)
    plt.tight_layout()
    
    # Save figure if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return plt.gcf()

def generate_comprehensive_shap_analysis(model, X, feature_names=None, 
                                       output_dir="outputs/figures/shap_analysis",
                                       n_samples=100):
    """
    Generate a comprehensive set of SHAP visualizations for model explainability.
    
    Parameters:
    -----------
    model : estimator
        Trained model with predict method
    X : pandas DataFrame or numpy array
        Feature matrix
    feature_names : list, optional
        List of feature names (if X is not a DataFrame)
    output_dir : str, optional
        Directory to save visualizations
    n_samples : int, optional
        Number of samples to use for SHAP analysis (for large datasets)
    
    Returns:
    --------
    dict
        Dictionary with paths to generated visualizations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert numpy array to DataFrame if necessary
    if isinstance(X, np.ndarray) and feature_names is not None:
        X = pd.DataFrame(X, columns=feature_names)
    
    # Sample data if needed
    if n_samples < X.shape[0]:
        X_sample = X.sample(n_samples, random_state=42)
    else:
        X_sample = X
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(X_sample)
    
    # Generate visualization paths
    viz_paths = {}
    
    # 1. Summary plot
    summary_path = os.path.join(output_dir, "shap_summary_plot.png")
    enhanced_shap_summary_plot(
        shap_values, 
        X_sample, 
        output_path=summary_path
    )
    viz_paths['summary'] = summary_path
    
    # 2. Bar summary plot
    bar_summary_path = os.path.join(output_dir, "shap_bar_summary.png")
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance", fontsize=18, pad=20)
    plt.tight_layout()
    plt.savefig(bar_summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    viz_paths['bar_summary'] = bar_summary_path
    
    # 3. Example force plots for a few instances
    force_plots = {}
    for i in range(min(5, X_sample.shape[0])):
        force_path = os.path.join(output_dir, f"force_plot_instance_{i}.png")
        enhanced_shap_force_plot(
            explainer, 
            shap_values, 
            X_sample, 
            i, 
            output_path=force_path,
            title=f"SHAP Force Plot - Instance {i}"
        )
        force_plots[f'instance_{i}'] = force_path
    viz_paths['force_plots'] = force_plots
    
    # 4. Waterfall plots for a few instances
    waterfall_plots = {}
    for i in range(min(5, X_sample.shape[0])):
        waterfall_path = os.path.join(output_dir, f"waterfall_plot_instance_{i}.png")
        enhanced_shap_waterfall_plot(
            explainer, 
            shap_values, 
            X_sample, 
            i, 
            output_path=waterfall_path,
            title=f"SHAP Waterfall Plot - Instance {i}"
        )
        waterfall_plots[f'instance_{i}'] = waterfall_path
    viz_paths['waterfall_plots'] = waterfall_plots
    
    # 5. Dependence plots for top features
    dependence_plots = {}
    
    # Get top feature indices
    if isinstance(shap_values, list):
        mean_abs_shap = np.mean(np.abs(shap_values[0]), axis=0)
    else:
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    
    top_indices = np.argsort(-mean_abs_shap)[:5]
    
    for idx in top_indices:
        feature_name = X_sample.columns[idx] if isinstance(X_sample, pd.DataFrame) else f"Feature {idx}"
        dependence_path = os.path.join(output_dir, f"dependence_plot_{feature_name.replace(' ', '_')}.png")
        
        enhanced_shap_dependence_plot(
            shap_values, 
            X_sample, 
            idx, 
            output_path=dependence_path
        )
        dependence_plots[feature_name] = dependence_path
    viz_paths['dependence_plots'] = dependence_plots
    
    # 6. Multiclass output handling (if applicable)
    if isinstance(shap_values, list) and len(shap_values) > 1:
        multiclass_plots = {}
        for class_idx in range(len(shap_values)):
            class_summary_path = os.path.join(output_dir, f"class_{class_idx}_summary.png")
            plt.figure(figsize=(12, 10))
            shap.summary_plot(shap_values[class_idx], X_sample, show=False)
            plt.title(f"SHAP Summary - Class {class_idx}", fontsize=18, pad=20)
            plt.tight_layout()
            plt.savefig(class_summary_path, dpi=300, bbox_inches='tight')
            plt.close()
            multiclass_plots[f'class_{class_idx}'] = class_summary_path
        viz_paths['multiclass'] = multiclass_plots
    
    return viz_paths

def generate_lime_explanation(explainer, model, X, instance_idx, feature_names=None,
                           num_features=10, output_path=None):
    """
    Generate a LIME explanation visualization for a specific instance.
    
    Parameters:
    -----------
    explainer : lime.lime_tabular.LimeTabularExplainer
        LIME explainer object
    model : estimator
        Trained model with predict method
    X : pandas DataFrame or numpy array
        Feature matrix
    instance_idx : int
        Index of the instance to explain
    feature_names : list, optional
        List of feature names (if X is not a DataFrame)
    num_features : int, optional
        Number of features to include in the explanation
    output_path : str, optional
        Path to save the figure, if None the figure is displayed
    
    Returns:
    --------
    lime.explanation.Explanation
        LIME explanation object
    """
    # Try to import lime, giving a helpful error if not installed
    try:
        import lime
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError:
        print("LIME is not installed. Install it with 'pip install lime'")
        return None
    
    # Convert to DataFrame if needed
    if isinstance(X, np.ndarray) and feature_names is not None:
        feature_names = list(feature_names)
        X_df = pd.DataFrame(X, columns=feature_names)
    else:
        X_df = X.copy()
        feature_names = list(X_df.columns)
    
    # Get the instance to explain
    instance = X_df.iloc[instance_idx].values if isinstance(X_df, pd.DataFrame) else X[instance_idx]
    
    # Create predict function based on model type
    if hasattr(model, "predict_proba"):
        predict_fn = model.predict_proba
    else:
        predict_fn = model.predict
    
    # Generate explanation
    explanation = explainer.explain_instance(
        instance, 
        predict_fn, 
        num_features=num_features
    )
    
    # Create visualization if output_path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Save explanation visualization
        fig = explanation.as_pyplot_figure()
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return explanation 