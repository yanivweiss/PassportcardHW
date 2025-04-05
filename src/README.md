# Source Code Structure

This directory contains the core Python modules for the PassportCard insurance claims analysis project.

## Directory Structure

- **data/** - Data loading, processing, and cleaning
- **features/** - Feature engineering and selection modules
- **models/** - Model training, validation, and evaluation
- **visualization/** - Visualization utilities and reporting tools

## Key Modules

### Data Processing

- Data loading and preprocessing
- Handling missing values
- Data cleaning and normalization

### Feature Engineering

- `feature_selection.py` - Feature selection methods:
  - Correlation-based selection
  - Univariate statistical tests
  - L1 regularization (Lasso) selection
  - Tree-based importance selection
  - Feature selection comparison

### Models

- `model_explainability.py` - Model explainability using:
  - SHAP (SHapley Additive exPlanations)
  - Partial dependence plots
  - LIME (Local Interpretable Model-agnostic Explanations)
  - Feature importance visualization

- `model_validation.py` - Advanced validation techniques:
  - Temporal cross-validation
  - Learning curves
  - Validation curves
  - Calibration assessment

- `model_comparison.py` - Model comparison framework:
  - Comparing multiple model types
  - Hyperparameter optimization
  - Performance metrics comparison
  - Feature importance comparison

### Visualization

- Visualization utilities for model performance
- Business insight reporting
- Error analysis visualizations

## Usage

Most modules can be imported and used as follows:

```python
# Example: Using feature selection
from features.feature_selection import compare_feature_selection_methods

# Compare multiple feature selection methods
selection_results = compare_feature_selection_methods(X, y)
consensus_features = selection_results['consensus_features']

# Example: Using model explainability
from models.model_explainability import explain_model_with_shap

# Get SHAP explanations for a model
shap_data = explain_model_with_shap(model, X_test, feature_names=X_test.columns)
``` 