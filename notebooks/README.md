# Data Analysis Notebooks

This directory contains Jupyter notebooks for data exploration, analysis, and model development.

## Notebooks Overview

1. **01_exploratory_data_analysis.ipynb** - Comprehensive EDA with visualizations and statistical analysis
   - Data distribution analysis
   - Missing value patterns
   - Feature relationships
   - Time series analysis of claims
   - Statistical hypothesis testing

# PassportCard Insurance Claims Prediction Notebooks

This directory contains Jupyter notebooks that provide interactive exploration and analysis of the PassportCard insurance claims prediction project.

## Notebook Overview

### 1. PassportCard_Insurance_Claims_Prediction.ipynb

**Purpose**: Main project notebook covering data exploration and cleaning.

**Contents**:
- Project introduction and overview
- Data loading and descriptive statistics
- Exploratory data analysis
- Claims distribution analysis
- Service type analysis
- Temporal patterns analysis
- Missing value handling
- Outlier detection and treatment

### 2. PassportCard_Model_Development.ipynb

**Purpose**: Focused on model development, evaluation, and optimization.

**Contents**:
- Feature engineering
- Data preparation for modeling
- Model selection and comparison
- XGBoost hyperparameter tuning
- Feature importance analysis
- Model predictions visualization
- Error analysis
- Model deployment considerations

### 3. PassportCard_Business_Applications.ipynb

**Purpose**: Demonstrates business applications and actionable insights from the model.

**Contents**:
- Risk assessment and segmentation
- Premium optimization strategies
- Resource allocation recommendations
- Product development opportunities
- Business impact analysis

## Getting Started

To run these notebooks:

1. Ensure you have all required dependencies installed:
```bash
pip install -r ../requirements.txt
```

2. Launch Jupyter:
```bash
jupyter notebook
```

3. Open the desired notebook from the browser interface

4. Run the cells in sequence to replicate the analysis

## Notes

- These notebooks are designed to work with the data files provided in the project's `data/` directory
- Some visualizations may require additional packages beyond the core requirements
- Each notebook can be run independently, but they are designed to be used in sequence

## Requirements

To run these notebooks, you'll need:

1. Python 3.8+ environment
2. Jupyter Notebook or JupyterLab
3. Required Python packages:
   - pandas
   - numpy
   - matplotlib
   - seaborn
   - scipy
   - missingno (for missing value visualization)

You can install the required packages using:

```bash
pip install pandas numpy matplotlib seaborn scipy missingno jupyter
```

## Running the Notebooks

1. Start Jupyter Notebook or JupyterLab:
   ```bash
   jupyter notebook
   # or
   jupyter lab
   ```

2. Navigate to the desired notebook and open it

3. Run cells sequentially to reproduce the analysis

## Notes

- Some notebooks require data files that should be placed in the `../data/` directory
- For best results, run cells in order as some analyses build on previous results
- If you encounter missing package errors, install the required packages using pip or conda 