# PassportCard Insurance Claims Prediction

This project develops a machine learning system to predict future insurance claims for PassportCard policyholders.

## Project Overview

The system uses historical claims data and member information to predict the total claim amount a customer is expected to make in the next six months. This prediction helps in risk assessment, pricing, and resource allocation.

## Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Data](#data)
- [Running the Prediction Pipeline](#running-the-prediction-pipeline)
- [Running Tests](#running-tests)
- [Key Components](#key-components)
- [Model Information](#model-information)
- [Fairness and Bias Mitigation](#fairness-and-bias-mitigation)
- [Explainability](#explainability)
- [License](#license)

## Project Structure

```
├── data/                   # Data files
│   ├── raw/                # Raw data files
│   └── processed/          # Processed data files
├── models/                 # Trained models
├── notebooks/              # Jupyter notebooks
├── outputs/                # Output files
│   ├── figures/            # Visualizations
│   └── tables/             # Generated tables and reports
├── src/                    # Source code
│   ├── data/               # Data processing modules
│   ├── features/           # Feature engineering modules
│   ├── models/             # Model training and evaluation modules
│   └── visualization/      # Visualization modules
├── tests/                  # Test files
└── docs/                   # Documentation
```

## Installation

To set up the project:

```bash
# Clone the repository
git clone https://github.com/your-username/passportcard-insurance-claims.git
cd passportcard-insurance-claims

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Data

The system uses two main data sources:

1. **Claims Data**: Historical claims made by policyholders, including amount, service date, service type, etc.
2. **Member Data**: Information about policyholders, including demographics, policy details, and questionnaire responses.

## Running the Prediction Pipeline

To run the full prediction pipeline:

```bash
# Run the end-to-end prediction pipeline
python src/run_prediction_pipeline.py

# Force retraining even if model exists (optional)
python src/run_prediction_pipeline.py --force-train
```

The pipeline performs the following steps:
1. Load and preprocess claims and member data
2. Engineer features from the data
3. Train a model or load a previously trained model
4. Make predictions
5. Analyze results and generate visualizations

Prediction results are saved to:
- `outputs/tables/prediction_results.csv`
- `outputs/figures/predictions/`

## Running Tests

To run all tests:

```bash
python run_tests.py
```

For more specific test runs:

```bash
# Run with verbose output
python run_tests.py --verbose

# Run a specific test class
python run_tests.py --test TestPredictionPipeline
```

See [tests/README.md](tests/README.md) for more information on testing.

## Key Components

### Data Preparation

- `enhanced_data_preparation.py`: Advanced data cleaning and preprocessing
- `enhanced_feature_engineering.py`: Feature creation and transformation

### Feature Engineering

- `advanced_temporal_features.py`: Time-based feature extraction
- `enhanced_risk_scores.py`: Risk score calculation based on member attributes

### Modeling

- `xgboost_modeling.py`: XGBoost model training and evaluation
- `advanced_modeling.py`: Advanced modeling techniques and hyperparameter optimization

### Analysis

- `error_analysis.py`: Prediction error analysis
- `fairness_analysis.py`: Fairness metrics and bias detection
- `explainability.py`: Model explainability using SHAP values

## Model Information

The primary model used is XGBoost, chosen for its:
- High predictive performance on tabular data
- Ability to handle missing values
- Feature importance ranking
- Non-linear relationship modeling

Key features influencing predictions include:
- Member questionnaire responses
- Past claim behavior
- Demographics
- Policy attributes

## Fairness and Bias Mitigation

The system includes tools to detect and mitigate unfair bias:
- Demographic parity analysis
- Group fairness metrics
- Bias mitigation techniques:
  - Sample weighting
  - Adversarial debiasing
  - Post-processing calibration

## Explainability

Model predictions are explained using:
- SHAP (SHapley Additive exPlanations) values
- Feature importance rankings
- Partial dependence plots
- Prediction error analysis

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Recent Updates

### Code Cleanup and Refactoring (2023-04-07)

The codebase has been significantly cleaned up and refactored to improve maintainability and reduce duplication:

1. **Consolidated Pipeline**: Multiple overlapping run scripts have been consolidated into a single unified pipeline in `src/run_prediction_pipeline.py`.

2. **Simplified Entry Point**: Added a single entry point script (`main.py`) in the root directory that provides a clean interface to the pipeline with various command-line options.

3. **Removed Unused Files**: Empty notebook files and redundant scripts have been removed to simplify the codebase.

4. **Improved Testing**: Added comprehensive tests for the consolidated pipeline in `tests/test_pipeline.py`.

To run the consolidated pipeline:

```bash
# Run the full pipeline with all features
python main.py

# Run with basic features only
python main.py --basic-features

# Force training a new model
python main.py --force-train

# Skip business report generation
python main.py --no-report

# Run with test data (smaller dataset)
python main.py --test
```

See the CHANGELOG.md for a full list of changes. 