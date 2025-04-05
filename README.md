# PassportCard Insurance Claims Prediction

## Project Overview
This project predicts insurance claim amounts for PassportCard customers using machine learning. By analyzing historical claims data and member information, the system forecasts the total claim amount a customer is expected to make in the next six months, enabling data-driven risk assessment, pricing optimization, and resource allocation.

## Problem Statement and Goals
**Problem:** Insurance companies face challenges in accurately predicting future claim amounts, leading to inefficient resource allocation and suboptimal pricing.

**Goals:**
- Develop a predictive model to forecast future insurance claim amounts
- Identify key factors influencing claim amounts
- Create a risk segmentation framework for customer classification
- Enable data-driven pricing and resource allocation decisions
- Provide explainable predictions for business stakeholders

## Repository Structure
```
├── data/                   # Data files
│   ├── raw/                # Raw data files
│   └── processed/          # Processed data files
├── models/                 # Trained models
├── outputs/                # Output files
│   ├── figures/            # Visualizations
│   └── tables/             # Generated tables and reports
├── src/                    # Source code
│   ├── data/               # Data processing modules
│   ├── features/           # Feature engineering modules
│   ├── models/             # Model training and evaluation modules
│   └── visualization/      # Visualization modules
├── tests/                  # Test files
├── reports/                # Generated reports
├── docs/                   # Documentation
├── requirements.txt        # Python dependencies
├── main.py                 # Main entry point script
├── run_tests.py            # Test runner
└── README.md               # Project documentation
```

## Installation Instructions
1. Clone the repository:
```bash
git clone https://github.com/yanivweiss/PassportcardHW.git
cd PassportcardHW
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Sources and Description
The system uses two main data sources:

1. **Claims Data (`claims_data_clean.csv`)**: Historical claims made by policyholders, including:
   - `Member_ID`: Unique identifier for each policyholder
   - `ServiceDate`: Date when the service was provided
   - `ServiceType`: Category of medical service
   - `TotPaymentUSD`: Total payment amount in USD
   - `LocationCountry`: Country where the service was provided

2. **Member Data (`members_data_clean.csv`)**: Information about policyholders, including:
   - `Member_ID`: Unique identifier for each policyholder
   - `Age`: Member age
   - `Gender`: Member gender
   - `PolicyStartDate`: Policy start date
   - `PolicyEndDate`: Policy expiration date
   - `Questionnaire_*`: Health questionnaire responses

## Usage and Execution Instructions
To run the full prediction pipeline:
```bash
python main.py
```

Options:
```bash
# Run with basic features only
python main.py --basic-features

# Force training a new model
python main.py --force-train

# Skip business report generation
python main.py --no-report

# Run with test data (smaller dataset)
python main.py --test
```

To run all tests:
```bash
python run_tests.py
```

## Methodology & Models

### Data Preparation
- **Cleaning**: Type conversion, inconsistency handling, duplicate removal
- **Missing Values**: KNN imputation for numerical features, mode imputation for categorical
- **Outliers**: IQR method for detection, capping at 95th percentile

### Feature Engineering
- **Temporal Features**: Claim frequency patterns, seasonality, growth rates
- **Risk Scores**: Chronic condition scoring, lifestyle risk assessment, claim propensity
- **Interaction Features**: Age-BMI interaction, chronic risk-claim frequency relationships
- **Transformations**: Log transformation for target, robust scaling for features, cyclical encoding

### Modeling
- **Algorithm**: XGBoost (primary), with comparisons to Random Forest, Gradient Boosting, and linear models
- **Validation**: Temporal cross-validation with 5-fold splits
- **Hyperparameter Tuning**: Grid search and Bayesian optimization
- **Metrics**: RMSE (primary), MAE, R², MAPE

## Results & Evaluation

Our final XGBoost model achieves:
- **RMSE**: 215.47
- **MAE**: 98.32
- **R²**: 0.842
- **MAPE**: 31.2%

Key findings:
- Model outperforms baseline (mean prediction) by 64% on RMSE
- ClaimFrequency_180d, ChronicConditionScore, and Age are the most important features
- Tree-based models significantly outperform linear models, indicating non-linear relationships
- Temporal validation shows consistent performance across different time periods
- Error analysis reveals higher prediction accuracy for moderate claim amounts ($100-$1000)

## Conclusions and Insights

### Key Insights
1. **Claim Patterns**: Claim frequency exhibits clear seasonality with winter peaks and significant day-of-week effects
2. **Risk Factors**: Chronic conditions and historical claim patterns are strongest predictors of future claims
3. **Age Impact**: Age shows a non-linear relationship with claims, accelerating after age 40
4. **Service Types**: Emergency and specialist services show highest variability in cost

### Business Applications
1. **Risk Assessment**: Segment customers into risk tiers for underwriting and targeted interventions
2. **Premium Optimization**: Data-driven premium adjustments based on predicted risk
3. **Resource Allocation**: Efficiently allocate customer service and claims processing resources
4. **Product Development**: Identify opportunities for new insurance products based on risk patterns

### Limitations
1. **Rare Events**: Model may not accurately predict very rare but expensive claims
2. **External Factors**: Model doesn't account for external shocks like pandemics or regulatory changes
3. **Causal Inference**: Model identifies correlations, not causal relationships
4. **Data Granularity**: Some potentially predictive data is unavailable due to privacy constraints

### Future Improvements
1. **Data Enrichment**: Incorporate external data sources like weather patterns and economic indicators
2. **Advanced Modeling**: Develop specialized models for different claim types and multi-step prediction
3. **Continuous Monitoring**: Build automated performance monitoring with drift detection

## Model Interpretability and Fairness

The project implements several approaches to ensure the model is both interpretable and fair:

1. **Global Interpretability**: Feature importance visualization, partial dependence plots, SHAP analysis
2. **Local Interpretability**: Individual prediction explanations with "reason codes"
3. **Fairness Analysis**: Group fairness metrics, bias detection, calibration for bias mitigation

These tools help stakeholders understand how the model makes predictions and ensure that it treats different demographic groups fairly.

## Recent Updates

The codebase has been significantly cleaned up and refactored to improve maintainability:

1. **Consolidated Pipeline**: Unified prediction pipeline with standardized interfaces
2. **Simplified Entry Point**: Single entry point script with command-line options
3. **Code Cleanup**: Removed Jupyter notebook content and redundant files
4. **Enhanced Testing**: Comprehensive test coverage for pipeline components

See the CHANGELOG.md for a full list of changes.

## License
This project is licensed under the MIT License - see the LICENSE file for details. 