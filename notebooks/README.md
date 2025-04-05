# Data Analysis Notebooks

This directory contains Jupyter notebooks for data exploration, analysis, and model development.

## Notebooks Overview

1. **01_exploratory_data_analysis.ipynb** - Comprehensive EDA with visualizations and statistical analysis
   - Data distribution analysis
   - Missing value patterns
   - Feature relationships
   - Time series analysis of claims
   - Statistical hypothesis testing

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