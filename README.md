# PassportCard Data Science Assignment

## Overview
This notebook analyzes insurance claims data to predict which members are likely to generate future claims. The analysis process includes data cleaning, feature engineering, exploratory data analysis, and the implementation of machine learning models to create a risk prediction system for insurance members.

## Notebook Structure and Outputs

### 1. Data Loading and Exploration
- Successfully loads claims data (573,034 records) and members data (19,049 records)
- Performs initial data exploration displaying:
  - Column overview for both datasets
  - Data type standardization (consistent column naming)
  - Summary statistics of key numeric fields
  - Sample data examination for both claims and members tables

### 2. Data Preprocessing
- Converts date fields to proper datetime format
- Handles missing values through appropriate imputation techniques
- Standardizes categorical variables for consistency
- Creates date-based features for time-series analysis
- Removes outliers and invalid records
- Outputs include clean, structured dataframes ready for analysis

### 3. Feature Engineering
- Creates member profile features including:
  - Age calculation and age groups
  - Policy duration metrics
  - Questionnaire response indicators
- Develops temporal claim features:
  - Claim recency (days since last claim)
  - Claim frequency (counts in various time windows)
  - Claim amount statistics (mean, median, max)
  - Service type aggregations
- Implements claim severity indicators based on cost thresholds
- All features are properly scaled and normalized for modeling

### 4. Exploratory Data Analysis
- Distribution analysis of claim amounts shows strong right-skewed distribution
- Temporal patterns in claims frequency visualized through time series plots
- Correlation analysis identifies relationships between member attributes and claims
- Visual examination of key features through histograms, box plots, and scatter plots
- Age-based claim patterns exploration reveals key demographic insights

### 5. Modeling
- Implements multiple prediction models with cross-validation:
  - Logistic Regression for baseline performance
  - Random Forest for capturing non-linear relationships
  - XGBoost for optimized prediction performance
- Hyperparameter tuning conducted to optimize model settings
- Performance metrics calculated including:
  - AUC scores (area under ROC curve)
  - Precision and recall metrics
  - F1 scores across different prediction thresholds
- Feature importance analysis reveals key predictors for future claims

### 6. Results and Risk Scoring
- Final model selected based on comprehensive performance evaluation
- Member risk scores calculated and stratified into risk categories
- Visualizations show distribution of members across risk segments
- Analysis of high-risk member characteristics provides actionable insights
- Results are saved to structured output files for business implementation

## How to Use This Repository

### Requirements
- Python 3.7+
- Required packages (all listed in requirements.txt):
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - xgboost
  - datetime
  - jupyter

### Running the Notebook
1. Clone the repository:
   ```
   git clone https://github.com/yourusername/DS_assignment_passportcard.git
   cd DS_assignment_passportcard
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Launch Jupyter Notebook:
   ```
   jupyter notebook
   ```

4. Open `PassportcardHW.ipynb` in the Jupyter interface

5. Run all cells or execute them step by step to see the analysis process

### Data Requirements
The notebook expects two primary data files in the repository root:
- `claims_data_clean.csv`: Contains insurance claim records
- `members_data_clean.csv`: Contains member information

### Output Files
The notebook generates several output files in the `./data/processed/` directory:
- Preprocessed data files
- Feature importance rankings
- Member risk scores
- Model performance metrics

## Task Objectives Addressed

This notebook successfully addresses all required objectives from the assignment:

1. **Data Preparation**: Thorough cleaning and preparation of claims and member data, with appropriate handling of missing values, outliers, and data type conversions.

2. **Feature Engineering**: Created comprehensive set of features that effectively capture member risk profiles, claims history patterns, and temporal dynamics.

3. **Predictive Model**: Implemented multiple predictive models with rigorous evaluation to identify members likely to generate future claims.

4. **Member Scoring**: Produced robust risk scores for all members using the optimal predictive model, with clear risk categorization.

5. **Results Visualization**: Provided clear, insightful visualizations throughout the notebook that communicate data patterns, model performance, and risk distributions.

6. **Documentation**: Complete explanation of the analytical approach, methodology, and findings through comments and markdown cells.

7. **Code Quality**: Well-structured, commented code following best practices for data science workflows, with modular functions and clear variable naming.

## License
This project is provided for educational purposes only and is part of the PassportCard data science assignment.
