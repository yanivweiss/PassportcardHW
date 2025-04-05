"""
Feature selection module.

This module provides various feature selection techniques for regression models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import os
from typing import Dict, List, Tuple, Union

def rank_features_by_correlation(X: pd.DataFrame, y: pd.Series, 
                               output_dir: str = 'visualizations/feature_selection') -> pd.DataFrame:
    """
    Rank features by correlation with target variable.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    output_dir : str
        Directory to save visualization outputs
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with feature correlation rankings
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate correlation between each feature and target
    correlations = []
    
    for column in X.columns:
        if pd.api.types.is_numeric_dtype(X[column]):
            corr = X[column].corr(y)
            p_value = f_regression(X[[column]], y)[1][0]
            correlations.append({
                'feature': column,
                'correlation': corr,
                'abs_correlation': abs(corr),
                'p_value': p_value,
                'significant': p_value < 0.05
            })
    
    # Convert to DataFrame and sort by absolute correlation
    corr_df = pd.DataFrame(correlations)
    corr_df = corr_df.sort_values('abs_correlation', ascending=False)
    
    # Plot top correlations
    top_n = min(20, len(corr_df))
    top_corr_df = corr_df.head(top_n)
    
    plt.figure(figsize=(12, 8))
    bars = plt.barh(
        top_corr_df['feature'], 
        top_corr_df['correlation'],
        color=[plt.cm.RdBu(0.1 + 0.8 * (x + 1) / 2) for x in top_corr_df['correlation']]
    )
    plt.axvline(x=0, color='black', linestyle='-')
    plt.title('Feature Correlation with Target', fontsize=14)
    plt.xlabel('Correlation Coefficient', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_correlation.png")
    plt.close()
    
    return corr_df

def select_with_univariate_tests(X: pd.DataFrame, y: pd.Series, 
                              k: int = 'all',
                              scoring: str = 'f_regression',  
                              output_dir: str = 'visualizations/feature_selection') -> Dict:
    """
    Select features using univariate statistical tests.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    k : int
        Number of top features to select (default: all)
    scoring : str
        Scoring method ('f_regression' or 'mutual_info')
    output_dir : str
        Directory to save visualization outputs
    
    Returns:
    --------
    dict
        Dictionary with selected features and scores
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert 'all' to number of features
    if k == 'all':
        k = X.shape[1]
    
    # Select appropriate scoring function
    if scoring == 'f_regression':
        score_func = f_regression
        method_name = 'F-Regression'
    elif scoring == 'mutual_info':
        score_func = mutual_info_regression
        method_name = 'Mutual Information'
    else:
        raise ValueError("scoring must be 'f_regression' or 'mutual_info'")
    
    # Apply feature selection
    selector = SelectKBest(score_func=score_func, k=k)
    selector.fit(X, y)
    
    # Get scores and selected features
    scores = selector.scores_
    selected_indices = selector.get_support(indices=True)
    selected_features = X.columns[selected_indices].tolist()
    
    # Create DataFrame with scores
    feature_scores = pd.DataFrame({
        'feature': X.columns,
        'score': scores,
        'selected': np.isin(range(len(X.columns)), selected_indices)
    })
    feature_scores = feature_scores.sort_values('score', ascending=False)
    
    # Plot feature scores
    top_n = min(20, len(feature_scores))
    top_features = feature_scores.head(top_n)
    
    plt.figure(figsize=(12, 8))
    bar_colors = ['green' if sel else 'gray' for sel in top_features['selected']]
    plt.barh(top_features['feature'], top_features['score'], color=bar_colors)
    plt.title(f'Top Features by {method_name}', fontsize=14)
    plt.xlabel('Score', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/kbest_{scoring}.png")
    plt.close()
    
    return {
        'selected_features': selected_features,
        'feature_scores': feature_scores,
        'selector': selector
    }

def select_with_rfe(X: pd.DataFrame, y: pd.Series, 
                  estimator = None,
                  n_features_to_select: int = None,
                  step: float = 0.1,
                  use_cv: bool = True,
                  cv: int = 5,
                  output_dir: str = 'visualizations/feature_selection') -> Dict:
    """
    Select features using Recursive Feature Elimination.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    estimator : estimator
        Base estimator to use (default: RandomForestRegressor)
    n_features_to_select : int
        Number of features to select (default: half of features)
    step : float
        Step size for RFE (proportion of features to remove at each step)
    use_cv : bool
        Whether to use cross-validation for RFE
    cv : int
        Number of cross-validation folds
    output_dir : str
        Directory to save visualization outputs
    
    Returns:
    --------
    dict
        Dictionary with selected features and results
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Set default estimator if not provided
    if estimator is None:
        estimator = RandomForestRegressor(n_estimators=100, random_state=42)
        estimator_name = "RandomForest"
    else:
        estimator_name = type(estimator).__name__
    
    # Set default number of features to select
    if n_features_to_select is None:
        n_features_to_select = X.shape[1] // 2
    
    # Apply RFE
    if use_cv:
        selector = RFECV(
            estimator=estimator,
            step=step,
            cv=cv,
            scoring='neg_root_mean_squared_error',
            min_features_to_select=n_features_to_select,
            n_jobs=-1
        )
    else:
        selector = RFE(
            estimator=estimator,
            n_features_to_select=n_features_to_select,
            step=step
        )
    
    selector.fit(X, y)
    
    # Get selected features
    selected_indices = selector.get_support(indices=True)
    selected_features = X.columns[selected_indices].tolist()
    
    # Create DataFrame with feature ranking
    if hasattr(selector, 'ranking_'):
        feature_ranking = pd.DataFrame({
            'feature': X.columns,
            'ranking': selector.ranking_,
            'selected': np.isin(range(len(X.columns)), selected_indices)
        })
        feature_ranking = feature_ranking.sort_values('ranking')
    else:
        feature_ranking = pd.DataFrame({
            'feature': selected_features,
            'selected': True
        })
    
    # Plot results
    if use_cv and hasattr(selector, 'cv_results_'):
        # Plot cross-validation scores for different numbers of features
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, len(selector.cv_results_['mean_test_score']) + 1), 
                -selector.cv_results_['mean_test_score'], 'o-')
        plt.axvline(x=selector.n_features_, color='r', linestyle='--')
        plt.title(f'RFECV Result ({estimator_name})', fontsize=14)
        plt.xlabel('Number of Features', fontsize=12)
        plt.ylabel('RMSE (Negative Score)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/rfecv_{estimator_name.lower()}.png")
        plt.close()
    
    # Plot selected features
    top_n = min(20, len(feature_ranking))
    top_features = feature_ranking.head(top_n)
    
    plt.figure(figsize=(12, 8))
    bar_colors = ['green' if sel else 'gray' for sel in top_features['selected']]
    if 'ranking' in top_features.columns:
        plt.barh(top_features['feature'], 1/top_features['ranking'], color=bar_colors)
        plt.title(f'Top Features by RFE with {estimator_name}', fontsize=14)
        plt.xlabel('Inverse Ranking (higher is better)', fontsize=12)
    else:
        plt.barh(top_features['feature'], np.arange(len(top_features), 0, -1), color=bar_colors)
        plt.title(f'Selected Features by RFE with {estimator_name}', fontsize=14)
        plt.xlabel('Feature Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/rfe_{estimator_name.lower()}.png")
    plt.close()
    
    return {
        'selected_features': selected_features,
        'feature_ranking': feature_ranking,
        'selector': selector
    }

def select_with_l1_regularization(X: pd.DataFrame, y: pd.Series, 
                               alpha: float = 0.01,
                               model_type: str = 'lasso',
                               output_dir: str = 'visualizations/feature_selection') -> Dict:
    """
    Select features using L1 regularization (Lasso).
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    alpha : float
        Regularization strength
    model_type : str
        Type of model ('lasso', 'ridge', or 'elasticnet')
    output_dir : str
        Directory to save visualization outputs
    
    Returns:
    --------
    dict
        Dictionary with selected features and model
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create model based on type
    if model_type.lower() == 'lasso':
        model = Lasso(alpha=alpha, random_state=42)
        model_name = 'Lasso'
    elif model_type.lower() == 'ridge':
        model = Ridge(alpha=alpha, random_state=42)
        model_name = 'Ridge'
    elif model_type.lower() == 'elasticnet':
        model = ElasticNet(alpha=alpha, l1_ratio=0.5, random_state=42)
        model_name = 'ElasticNet'
    else:
        raise ValueError("model_type must be 'lasso', 'ridge', or 'elasticnet'")
    
    # Fit model
    model.fit(X, y)
    
    # Get coefficients and selected features
    coefs = model.coef_
    feature_coefs = pd.DataFrame({
        'feature': X.columns,
        'coefficient': coefs,
        'abs_coefficient': np.abs(coefs)
    })
    feature_coefs = feature_coefs.sort_values('abs_coefficient', ascending=False)
    
    # Get selected features (non-zero coefficients)
    selected_features = feature_coefs[feature_coefs['coefficient'] != 0]['feature'].tolist()
    
    # Plot coefficients
    top_n = min(20, len(feature_coefs))
    top_features = feature_coefs.head(top_n)
    
    plt.figure(figsize=(12, 8))
    plt.barh(
        top_features['feature'], 
        top_features['coefficient'],
        color=[plt.cm.RdBu(0.1 + 0.8 * (x + 1) / 2) for x in top_features['coefficient']]
    )
    plt.axvline(x=0, color='black', linestyle='-')
    plt.title(f'{model_name} Coefficients (alpha={alpha})', fontsize=14)
    plt.xlabel('Coefficient', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{model_type.lower()}_coefficients.png")
    plt.close()
    
    return {
        'selected_features': selected_features,
        'feature_coefs': feature_coefs,
        'model': model
    }

def select_with_tree_importance(X: pd.DataFrame, y: pd.Series, 
                             model_type: str = 'random_forest',
                             output_dir: str = 'visualizations/feature_selection') -> Dict:
    """
    Select features using tree-based feature importance.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    model_type : str
        Type of model ('random_forest' or 'xgboost')
    output_dir : str
        Directory to save visualization outputs
    
    Returns:
    --------
    dict
        Dictionary with selected features and model
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create model based on type
    if model_type.lower() == 'random_forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model_name = 'Random Forest'
    elif model_type.lower() == 'xgboost':
        model = XGBRegressor(n_estimators=100, random_state=42)
        model_name = 'XGBoost'
    else:
        raise ValueError("model_type must be 'random_forest' or 'xgboost'")
    
    # Fit model
    model.fit(X, y)
    
    # Get feature importance
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': importance
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    # Set a threshold for selection
    threshold = 0.01  # Features with importance > 1%
    selected_features = feature_importance[feature_importance['importance'] > threshold]['feature'].tolist()
    
    # Plot feature importance
    top_n = min(20, len(feature_importance))
    top_features = feature_importance.head(top_n)
    
    plt.figure(figsize=(12, 8))
    plt.barh(top_features['feature'], top_features['importance'], color='teal')
    plt.title(f'{model_name} Feature Importance', fontsize=14)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{model_type.lower()}_importance.png")
    plt.close()
    
    return {
        'selected_features': selected_features,
        'feature_importance': feature_importance,
        'model': model
    }

def compare_feature_selection_methods(
    X: pd.DataFrame, 
    y: pd.Series,
    n_features: int = 20, 
    output_dir: str = 'visualizations/feature_selection'
) -> Dict:
    """
    Compare different feature selection methods.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    n_features : int
        Number of top features to compare
    output_dir : str
        Directory to save visualization outputs
    
    Returns:
    --------
    dict
        Dictionary with feature selection comparison results
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    print("Comparing feature selection methods...")
    
    # Run all feature selection methods
    print("1. Correlation analysis...")
    corr_results = rank_features_by_correlation(X, y, output_dir)
    corr_features = corr_results.head(n_features)['feature'].tolist()
    
    print("2. F-regression test...")
    f_reg_results = select_with_univariate_tests(X, y, k=n_features, scoring='f_regression', output_dir=output_dir)
    f_reg_features = f_reg_results['feature_scores'].head(n_features)['feature'].tolist()
    
    print("3. Mutual information...")
    mi_results = select_with_univariate_tests(X, y, k=n_features, scoring='mutual_info', output_dir=output_dir)
    mi_features = mi_results['feature_scores'].head(n_features)['feature'].tolist()
    
    print("4. Lasso regularization...")
    lasso_results = select_with_l1_regularization(X, y, alpha=0.01, model_type='lasso', output_dir=output_dir)
    lasso_features = lasso_results['feature_coefs'].head(n_features)['feature'].tolist()
    
    print("5. Random Forest importance...")
    rf_results = select_with_tree_importance(X, y, model_type='random_forest', output_dir=output_dir)
    rf_features = rf_results['feature_importance'].head(n_features)['feature'].tolist()
    
    print("6. XGBoost importance...")
    xgb_results = select_with_tree_importance(X, y, model_type='xgboost', output_dir=output_dir)
    xgb_features = xgb_results['feature_importance'].head(n_features)['feature'].tolist()
    
    # Collect all unique features from all methods
    all_features = list(set(corr_features + f_reg_features + mi_features + lasso_features + rf_features + xgb_features))
    
    # Create comparison matrix
    comparison = pd.DataFrame(index=all_features)
    
    # Add ranks from each method
    methods = {
        'Correlation': corr_results[['feature', 'abs_correlation']].rename(columns={'abs_correlation': 'score'}),
        'F-Regression': f_reg_results['feature_scores'][['feature', 'score']],
        'Mutual Info': mi_results['feature_scores'][['feature', 'score']],
        'Lasso': lasso_results['feature_coefs'][['feature', 'abs_coefficient']].rename(columns={'abs_coefficient': 'score'}),
        'Random Forest': rf_results['feature_importance'][['feature', 'importance']].rename(columns={'importance': 'score'}),
        'XGBoost': xgb_results['feature_importance'][['feature', 'importance']].rename(columns={'importance': 'score'})
    }
    
    # Add rank columns
    for method_name, method_df in methods.items():
        # Create temporary ranking
        temp_df = method_df.sort_values('score', ascending=False).reset_index(drop=True)
        temp_df['rank'] = temp_df.index + 1
        
        # Add rank to comparison
        for feature in all_features:
            if feature in temp_df['feature'].values:
                rank = temp_df[temp_df['feature'] == feature]['rank'].values[0]
                comparison.loc[feature, method_name] = rank
            else:
                comparison.loc[feature, method_name] = np.nan
    
    # Calculate average rank and count of methods that selected each feature
    comparison['Avg Rank'] = comparison.mean(axis=1)
    comparison['Methods'] = comparison.count(axis=1)
    
    # Sort by average rank
    comparison = comparison.sort_values('Avg Rank')
    
    # Identify consensus features
    consensus_threshold = len(methods) // 2  # Selected by at least half of methods
    consensus_features = comparison[comparison['Methods'] >= consensus_threshold].index.tolist()
    
    # Plot feature selection comparison
    top_comparison = comparison.head(min(30, len(comparison)))
    
    # Create heatmap
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        top_comparison.drop(['Avg Rank', 'Methods'], axis=1),
        cmap='YlGnBu_r',
        linewidths=0.5,
        annot=True,
        fmt='.0f',
        cbar_kws={'label': 'Rank (lower is better)'}
    )
    plt.title('Feature Selection Method Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/method_comparison_heatmap.png")
    plt.close()
    
    # Create bar chart of feature consensus
    plt.figure(figsize=(12, 8))
    consensus_scores = comparison.sort_values('Methods', ascending=False).head(20)
    plt.barh(consensus_scores.index, consensus_scores['Methods'], color='purple')
    plt.title('Feature Selection Consensus', fontsize=14)
    plt.xlabel('Number of Methods', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.axvline(x=consensus_threshold, color='red', linestyle='--', 
               label=f'Consensus Threshold ({consensus_threshold})')
    plt.legend()
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_selection_consensus.png")
    plt.close()
    
    print(f"Feature selection comparison complete. {len(consensus_features)} consensus features identified.")
    
    return {
        'comparison': comparison,
        'consensus_features': consensus_features,
        'methods': {
            'correlation': corr_results,
            'f_regression': f_reg_results,
            'mutual_info': mi_results,
            'lasso': lasso_results,
            'random_forest': rf_results,
            'xgboost': xgb_results
        }
    }

if __name__ == "__main__":
    print("This module provides functions for feature selection.")
    print("Import and use these functions to select the most important features.") 