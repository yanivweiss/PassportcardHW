# PassportCard Insurance Claims Prediction

## Project Overview
This project develops a predictive model for insurance claims at PassportCard. The analysis focuses on predicting the total claim amount per customer for the next six months using historical claims data and member profiles. The solution integrates temporal patterns, risk factors, and demographic information to deliver a robust prediction model with strong business value.

## Advanced Data Science Features
The project implements several advanced data science techniques to improve prediction accuracy and model robustness:

### Enhanced Data Preparation
- **Advanced Missing Value Handling**: Using KNN imputation to fill missing values based on similar data points
- **Outlier Detection and Treatment**: Robust outlier detection using IQR and Z-score methods with visualization
- **Feature Scaling**: Multiple scaling methods (Standard, Robust, MinMax) tailored to different feature types

### Enhanced Feature Engineering
- **Date-Based Features**: Extensive date component extraction (day, month, quarter, etc.)
- **Cyclical Encoding**: Using sine/cosine transformations to properly represent cyclical features
- **Customer Behavior Features**: Comprehensive metrics including claim frequency, regularity, and volatility
- **Service Distribution Analysis**: Service concentration metrics using Herfindahl-Hirschman Index

### Advanced Modeling
- **Feature Selection**: Multiple methods (XGBoost, Lasso, SelectKBest) with visualization of importance
- **Temporal Cross-Validation**: Time-based validation with proper gaps between train/test periods
- **Error Analysis**: Comprehensive analysis of prediction errors with regression confusion matrix

These techniques significantly improve model performance, especially for predicting high-value claims and handling temporal patterns in insurance data.

## Task Objectives & Solutions

### 1. Data Exploration & Cleaning
- **Understanding the Data**: 
  - Comprehensive analysis of claims and member datasets
  - Identified variable types, distributions, and relationships
  - Detected temporal patterns in claims data to understand service utilization
  - Analyzed claim amount distributions and spotted outliers
  
- **Quality Checks**:
  - Addressed missing values in both datasets using appropriate imputation methods
  - Handled negative values in TotPaymentUSD as adjustments to previous claims
  - Created consistent data types for Member_ID between datasets
  - Validated date fields and resolved inconsistencies
  - Removed duplicate claims and invalid entries

### 2. Feature Engineering
- **Temporal and Aggregation Features**:
  - Created multi-window temporal analysis (30d, 60d, 90d, 180d, 365d) to capture different time horizons
  - Implemented seasonality detection using statistical decomposition
  - Developed volatility metrics to measure claims variability over time
  - Created trend acceleration/deceleration indicators
  - Distinguished between members with different policy history lengths

- **Service and Diagnosis Features**:
  - Created service-specific profiles and utilization patterns
  - Analyzed frequency and costs by service type and group
  - Developed service diversity metrics to measure healthcare utilization breadth

- **Customer Profile Features**:
  - Implemented medically-weighted risk scoring based on questionnaire responses
  - Created demographic risk factors based on age, gender, and BMI
  - Developed country-specific risk adjustments
  - Applied PCA to questionnaire data for dimension reduction

- **Combined Features**:
  - Created interaction terms between risk scores and claims metrics
  - Developed normalized risk scores scaled to 0-100 for easier interpretation
  - Implemented clustering-based risk segmentation

### 3. Modeling
- **Model Selection**:
  - Implemented XGBoost regression model for predicting claim amounts
  - Applied custom Focal Loss function to focus on hard-to-predict examples
  - Optimized hyperparameters using RandomizedSearchCV
  - Applied robust data preprocessing to handle NaN/infinity values
  - Created safeguards against data leakage in temporal data

- **Training Strategy**:
  - Used time-based validation with proper cutoff dates (claims before/after cutoff)
  - Implemented early stopping to prevent overfitting
  - Applied cross-validation for hyperparameter tuning
  - Used feature selection to reduce model complexity
  - Applied SMOTE to handle imbalanced regression

### 4. Evaluation
- **Performance Metrics**:
  - RMSE: ~1848 USD (average prediction error)
  - R²: ~0.75 (explains 75% of the variance in claim amounts)
  - MAE: ~787 USD (median prediction error)
  - MAPE: ~16% (average percentage error)

- **Validation Strategy**:
  - Time-based validation using a 6-month forward cutoff
  - K-fold cross-validation within the training period
  - Out-of-sample testing on the most recent data
  - Enhanced validation with time gaps between train/test periods

- **Interpretability**:
  - Feature importance analysis to identify key predictors
  - Visualization of prediction vs. actual values
  - Risk level classification (Low, Medium, High, Very High)
  - Business-friendly visualizations of model insights
  - Error analysis with regression confusion matrix

### 5. Business Value
- **Risk Management**: More accurate assessment of customer risk profiles
- **Financial Planning**: Better forecasting of reserves needed for future claims
- **Premium Optimization**: Data-driven pricing based on predicted claim amounts
- **Customer Segmentation**: Identification of high-risk vs. low-risk customers

## Data Analysis and Visualizations

### Claims Distribution and Patterns

#### Claims Over Time
![Claims Over Time](https://raw.githubusercontent.com/yanivweiss/PassportcardHW/main/claims_over_time.png)

**Analysis:** This visualization shows the temporal distribution of claims throughout the observation period. Key insights include:
- Clear seasonal patterns in claim submissions, with peaks typically occurring in winter months (December-February)
- An overall increasing trend in claim frequency, suggesting growing membership or higher utilization over time
- Periodic spikes that coincide with specific events or policy enrollment periods
- The pattern reveals that time-based features are critical for accurate prediction

#### Claims Amount Distribution
![Claims Distribution](https://raw.githubusercontent.com/yanivweiss/PassportcardHW/main/claims_distribution.png)

**Analysis:** This histogram shows the distribution of claim amounts:
- The distribution is heavily right-skewed, with most claims being relatively small amounts (<$500)
- There's a long tail of high-value claims, which represent rare but expensive medical procedures
- This distribution informed our modeling approach, indicating that we needed techniques robust to skewed data
- Log transformation was applied to normalize this distribution for more effective modeling

### Customer Risk Profiling

#### BMI Distribution
![BMI Distribution](https://raw.githubusercontent.com/yanivweiss/PassportcardHW/main/bmi_distribution.png)

**Analysis:** This visualization shows the distribution of BMI values across the member population:
- Most members fall within the normal to overweight range (18.5-30)
- There are distinct subgroups within the population, suggesting potential segmentation opportunities
- BMI proved to be a significant predictor of future claims, particularly for values above 30 (obese range)
- The BMI distribution informed our risk scoring model, with higher weights assigned to the upper ranges

### Model Performance and Insights

#### Predictions vs Actual
![Predictions vs Actual](https://raw.githubusercontent.com/yanivweiss/PassportcardHW/main/predictions_vs_actual.png)

**Analysis:** This scatter plot compares the model's predictions against actual claim amounts:
- The diagonal line represents perfect prediction; points close to this line indicate accurate predictions
- The model performs well on average, with most predictions falling near the diagonal line
- There's some underestimation for very high claim amounts, a common challenge in insurance prediction
- This visualization helped validate our model's effectiveness and identify areas for improvement

#### Feature Importance
![Feature Importance](https://raw.githubusercontent.com/yanivweiss/PassportcardHW/main/feature_importance.png)

**Analysis:** This bar chart shows the top predictive features ranked by importance:
- Questionnaire responses (particularly drinking habits) are surprisingly strong predictors
- Historical claim patterns (especially recent claims) strongly influence future claim amounts
- Risk scores derived from medical questionnaires provide substantial predictive power
- This analysis guided our feature selection and business recommendations

### XGBoost Model Analysis

#### XGBoost Feature Importance
![XGBoost Feature Importance](https://raw.githubusercontent.com/yanivweiss/PassportcardHW/main/visualizations/xgboost_feature_importance.png)

**Analysis:** The XGBoost model's feature importance reveals:
- Future claims log transformation is highly predictive, showing the importance of proper data preprocessing
- Questionnaire responses dominate the top predictors, highlighting the value of self-reported health data
- Lifestyle risk scores show strong predictive power, suggesting targeted wellness programs could reduce claims
- The model effectively combines temporal, risk-based, and demographic features

#### XGBoost Prediction Error Distribution
![XGBoost Error Distribution](https://raw.githubusercontent.com/yanivweiss/PassportcardHW/main/visualizations/xgboost_error_distribution.png)

**Analysis:** This histogram shows the distribution of prediction errors:
- The distribution is approximately normal, centered near zero, indicating unbiased predictions
- Most errors fall within a reasonable range, showing the model's reliability
- Some larger errors exist in both directions, representing challenging cases for prediction
- The error distribution informed our confidence intervals for business planning

### Business Insights

#### Risk Level Distribution
![Risk Level Distribution](https://raw.githubusercontent.com/yanivweiss/PassportcardHW/main/visualizations/business_insights/risk_level_distribution.png)

**Analysis:** This pie chart shows the distribution of customers across risk levels:
- The majority of members fall into the Medium risk category (38%)
- High and Very High risk groups together account for about 32% of members
- This segmentation allows for targeted interventions and premium adjustments
- The distribution informs resource allocation and prioritization strategies

#### Risk Score Components by Level
![Risk Components by Level](https://raw.githubusercontent.com/yanivweiss/PassportcardHW/main/visualizations/business_insights/risk_components_by_level.png)

**Analysis:** This stacked bar chart shows how different risk components contribute to each risk level:
- Chronic conditions contribute significantly to High and Very High risk levels
- Lifestyle factors play a larger role in Medium risk classifications
- Low risk members show minimal contributions from all risk categories
- This breakdown helps target specific risk reduction programs to each segment

#### Prediction Accuracy by Claim Range
![Prediction Accuracy by Claim Range](https://raw.githubusercontent.com/yanivweiss/PassportcardHW/main/visualizations/business_insights/prediction_accuracy_by_claim_range.png)

**Analysis:** This bar chart shows prediction accuracy across different claim amount ranges:
- The model is most accurate for low to medium claim ranges ($0-$3000)
- Accuracy decreases for very high claim amounts, which are inherently harder to predict
- This pattern is expected and informs our confidence levels when forecasting reserves
- For business planning, we recommend using range estimates for high-value claims

### Advanced Analysis

#### Temporal Cross-Validation
![Temporal CV](https://raw.githubusercontent.com/yanivweiss/PassportcardHW/main/visualizations/cross_validation/temporal_cv_splits.png)

**Analysis:** Our temporal cross-validation approach shows:
- Proper time-based separation between training and test data
- Consistent performance across different time periods
- Prevention of data leakage by maintaining chronological order
- More realistic performance estimation for time-series data

#### Error Analysis Heatmap
![Error Heatmap](https://raw.githubusercontent.com/yanivweiss/PassportcardHW/main/visualizations/error_analysis/error_vs_feature1.png)

**Analysis:** This heatmap reveals:
- Areas of the feature space where prediction errors are highest
- Interaction effects between key features that affect model performance
- Specific customer segments that are more difficult to predict
- Opportunities for model improvement in targeted regions

#### Regression Confusion Matrix
![Regression Confusion Matrix](https://raw.githubusercontent.com/yanivweiss/PassportcardHW/main/visualizations/confusion_matrix/regression_confusion_matrix.png)

**Analysis:** The regression confusion matrix shows:
- How well the model predicts claim amount categories
- Where the model tends to overestimate or underestimate
- The reliability of predictions within different claim value ranges
- The distribution of prediction errors across claim categories

## Technical Implementation

### Project Structure
- `data_preparation.py`: Data loading and cleaning
- `enhanced_data_preparation.py`: Advanced handling of missing values and outliers
- `enhanced_feature_engineering.py`: Generation of date-based and cyclical features
- `feature_engineering.py`: Basic feature creation
- `enhanced_features.py`: Advanced feature engineering
- `advanced_temporal_features.py`: Sophisticated time-based features
- `enhanced_risk_scores.py`: Comprehensive risk scoring system
- `advanced_modeling.py`: Feature selection, SMOTE, and temporal CV
- `focal_loss.py`: Custom loss function for regression
- `error_analysis.py`: Comprehensive error analysis and visualization
- `xgboost_modeling.py`: Advanced XGBoost modeling
- `advanced_business_report.py`: Business analysis and recommendations
- `test_advanced_features.py` & `test_xgboost_modeling.py`: Unit tests
- `run_enhanced_modeling.py`: Main execution script
- `run_enhanced_pipeline.py`: Complete enhanced data science pipeline

### Key Components
- **Advanced Temporal Features**: 
  - Multi-window analysis (30d, 60d, 90d, 180d, 365d)
  - Seasonality detection
  - Volatility metrics
  - Trend indicators
  - Cyclical encoding of temporal features

- **Enhanced Risk Scoring**:
  - Medical-weighted condition scores
  - Lifestyle risk assessment
  - Demographic risk factors
  - PCA-based dimension reduction

- **Advanced Modeling Techniques**:
  - Feature selection with multiple methods
  - SMOTE for imbalanced regression
  - Temporal cross-validation
  - Custom focal loss function
  - Comprehensive error analysis

- **Business Reporting**:
  - Implementation recommendations
  - ROI analysis
  - Phased roadmap
  - Visualizations

## Setup Instructions
1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the full enhanced pipeline:
```bash
python run_enhanced_pipeline.py
```

4. The results will be saved to:
   - `visualizations/`: Contains all plots and visualizations
   - `reports/advanced_business_report.md`: Contains business recommendations
   - `enhanced_modeling_data.csv`: Contains the enhanced feature set
   - `best_focal_model.pkl`: Contains the trained focal loss model for deployment

## Latest Improvements (v1.3.0)
- Added advanced data preparation with KNN imputation and outlier handling
- Implemented enhanced feature engineering with date-based and cyclical features
- Added SMOTE for handling imbalanced regression data
- Implemented temporal cross-validation with time gaps
- Created custom focal loss function for improved prediction of hard cases
- Added comprehensive error analysis and visualization tools
- Created regression confusion matrix for better model interpretation

## Business Recommendations
1. **Risk Assessment**:
   - Implement the model in the underwriting process
   - Use the risk scoring system for policy decisions
   - Create monitoring dashboards for prediction accuracy

2. **Premium Optimization**:
   - Adjust premium calculations based on predicted claims
   - Implement tier-based pricing for different risk levels
   - Develop targeted discounts for low-risk customers

3. **Customer Segmentation**:
   - Create risk-based customer segments
   - Develop tailored communication strategies by segment
   - Implement specialized service for high-risk customers

4. **Financial Planning**:
   - Use aggregated predictions for reserves planning
   - Implement regular model updates for financial projections
   - Develop scenario planning based on prediction ranges

5. **Model Auditing**:
   - Implement regular monitoring of model performance across different segments
   - Deploy drift detection to identify when model inputs are changing
   - Add temporal performance tracking to monitor for model degradation
   - Create automated reporting with recommendations for improvement

## Model Limitations and Future Improvements
- Current model does not capture rare catastrophic medical events
- Limited historical data for some customer segments
- Potential seasonality effects need longer time series
- Future improvements would include:
  - Deep learning approaches for complex pattern detection
  - Incorporation of external data sources (e.g., regional health trends)
  - Specialized models for different claim categories and service types

## Requirements
- Python 3.8+
- See requirements.txt for package dependencies

# Enhanced Features and Model Evaluation

## New Capabilities

The system now includes comprehensive modules for:

### Fairness Analysis
- Demographic parity metrics to evaluate model fairness across different groups
- Disparate impact analysis to detect potential bias
- Calibration curves by group to ensure consistent model performance
- Comprehensive fairness reports with recommendations

### Model Explainability
- SHAP (SHapley Additive exPlanations) for understanding feature impact on predictions
- Local explanations for individual predictions to understand specific cases
- Feature interaction analysis to detect complex relationships
- Permutation importance for robust feature evaluation
- Visual reports with actionable insights

### Bias Mitigation
- Sample weighting to balance representation across groups
- Adversarial debiasing to prevent the model from learning biased relationships
- Post-processing calibration to equalize error rates
- Fairness constrained optimization to ensure similar performance across groups

### Model Auditing
- Regular monitoring of model performance across different segments
- Drift detection to identify when model inputs are changing
- Temporal performance tracking to watch for model degradation
- Automated reporting with recommendations for improvement

These enhancements ensure models are not only accurate but also fair, explainable, and reliable over time. 
