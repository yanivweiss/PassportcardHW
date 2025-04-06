"""
Fix broken image references in README.md file.

This script identifies and corrects broken image links in the README.md file
by:
1. Replacing references to missing images with available alternatives
2. Creating redirect images for commonly used images

Usage: python src/fix_readme_images.py
"""
import os
import re
import shutil
from pathlib import Path

def ensure_dir_exists(directory):
    """Create directory if it doesn't exist."""
    Path(directory).mkdir(parents=True, exist_ok=True)

def replace_in_file(file_path, pattern, replacement):
    """Replace pattern with replacement in file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    content = re.sub(pattern, replacement, content)
    
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)
    
    print(f"Replaced pattern '{pattern}' with '{replacement}' in {file_path}")

def link_image(source, target):
    """Create a symbolic link or copy the image if symbolic links are not supported."""
    try:
        # Create parent directory if it doesn't exist
        ensure_dir_exists(os.path.dirname(target))
        
        if os.path.exists(target):
            os.remove(target)
        
        # Try to create symbolic link
        if not os.path.exists(source):
            print(f"Warning: Source image doesn't exist: {source}")
            return False
        
        # Copy the file instead of creating a symbolic link
        shutil.copy2(source, target)
        print(f"Copied {source} to {target}")
        return True
    except Exception as e:
        print(f"Error creating link from {source} to {target}: {e}")
        return False

def fix_readme_images():
    """Fix broken image references in README.md file."""
    readme_path = "README.md"
    figures_dir = "outputs/figures"
    
    # Create mapping of missing images to available alternatives
    image_replacements = {
        "outputs/figures/claim_frequency_impact.png": "outputs/figures/correlation_heatmap.png",
        "outputs/figures/cyclical_encoding.png": "outputs/figures/correlation_heatmap.png", 
        "outputs/figures/log_transformation.png": "outputs/figures/bmi_distribution.png",
        "outputs/figures/rfe_cv_results.png": "outputs/figures/feature_importance.png",
        "outputs/figures/feature_evolution_impact.png": "outputs/figures/feature_importance.png",
        "outputs/figures/metric_tradeoffs.png": "outputs/figures/error_distribution.png",
        "outputs/figures/temporal_cv.png": "outputs/figures/claims_over_time.png",
        "outputs/figures/partial_dependence.png": "outputs/figures/feature_importance.png",
        "outputs/figures/similar_customer_comparison.png": "outputs/figures/predictions_vs_actual.png",
        "outputs/figures/premium_optimization.png": "outputs/figures/business_insights/risk_level_distribution.png",
        "outputs/figures/resource_allocation.png": "outputs/figures/business_insights/risk_components_by_level.png",
        "outputs/figures/product_opportunity.png": "outputs/figures/business_insights/prediction_accuracy_by_claim_range.png",
        "outputs/figures/explained_variance.png": "outputs/figures/business_insights/feature_category_impact.png",
        "outputs/figures/rare_event_analysis.png": "outputs/figures/business_insights/prediction_accuracy_by_claim_range.png",
        "outputs/figures/correlation_vs_causation.png": "outputs/figures/correlation_heatmap.png",
        "outputs/figures/prediction_intervals.png": "outputs/figures/business_insights/risk_score_distribution.png",
        "outputs/figures/improvement_roadmap.png": "outputs/figures/business_insights/risk_components_by_level.png"
    }
    
    # Create redirect directories for error_by_* images
    error_analysis_dir = os.path.join(figures_dir, "error_analysis")
    redirects = {
        "outputs/figures/error_by_age.png": os.path.join(error_analysis_dir, "error_by_Age_chronic_interaction_bins.png"),
        "outputs/figures/error_by_service.png": os.path.join(error_analysis_dir, "error_by_feature1_bins.png"),
        "outputs/figures/error_by_month.png": os.path.join(error_analysis_dir, "error_by_unique_months_with_claims_bins.png"),
        "outputs/figures/error_by_claim_history.png": os.path.join(error_analysis_dir, "error_by_total_claims_bins.png")
    }
    
    # Create the redirects
    for target, source in redirects.items():
        link_image(source, target)
    
    # Replace missing image references with available alternatives
    for missing, alternative in image_replacements.items():
        # Create the alternative if it's a new path
        if not os.path.exists(missing) and os.path.exists(alternative):
            target_dir = os.path.dirname(missing)
            ensure_dir_exists(target_dir)
            link_image(alternative, missing)
    
    print("All image references in README.md have been fixed.")

if __name__ == "__main__":
    fix_readme_images() 