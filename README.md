# PassportCard Insurance Claims Prediction

## Project Overview
This project develops a predictive model for insurance claims at PassportCard. The analysis focuses on predicting the total claim amount per customer for the next six months using historical claims data and member profiles. The solution integrates temporal patterns, risk factors, and demographic information to deliver a robust prediction model with strong business value.

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
  - Optimized hyperparameters using RandomizedSearchCV
  - Applied robust data preprocessing to handle NaN/infinity values
  - Created safeguards against data leakage in temporal data

- **Training Strategy**:
  - Used time-based validation with proper cutoff dates (claims before/after cutoff)
  - Implemented early stopping to prevent overfitting
  - Applied cross-validation for hyperparameter tuning
  - Used feature selection to reduce model complexity

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

- **Interpretability**:
  - Feature importance analysis to identify key predictors
  - Visualization of prediction vs. actual values
  - Risk level classification (Low, Medium, High, Very High)
  - Business-friendly visualizations of model insights

### 5. Business Value
- **Risk Management**: More accurate assessment of customer risk profiles
- **Financial Planning**: Better forecasting of reserves needed for future claims
- **Premium Optimization**: Data-driven pricing based on predicted claim amounts
- **Customer Segmentation**: Identification of high-risk vs. low-risk customers

## Technical Implementation

### Project Structure
- `data_preparation.py`: Data loading and cleaning
- `feature_engineering.py`: Basic feature creation
- `enhanced_features.py`: Advanced feature engineering
- `advanced_temporal_features.py`: Sophisticated time-based features
- `enhanced_risk_scores.py`: Comprehensive risk scoring system
- `xgboost_modeling.py`: Advanced XGBoost modeling
- `advanced_business_report.py`: Business analysis and recommendations
- `test_advanced_features.py` & `test_xgboost_modeling.py`: Unit tests
- `run_enhanced_modeling.py`: Main execution script

### Key Components
- **Advanced Temporal Features**: 
  - Multi-window analysis (30d, 60d, 90d, 180d, 365d)
  - Seasonality detection
  - Volatility metrics
  - Trend indicators

- **Enhanced Risk Scoring**:
  - Medical-weighted condition scores
  - Lifestyle risk assessment
  - Demographic risk factors
  - PCA-based dimension reduction

- **XGBoost Modeling**:
  - Hyperparameter optimization
  - Feature importance analysis
  - Robust preprocessing
  - Model visualization

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

3. Run the full analysis:
```bash
python run_enhanced_modeling.py
```

4. The results will be saved to:
   - `visualizations/`: Contains all plots and visualizations
   - `reports/advanced_business_report.md`: Contains business recommendations
   - `integrated_features.csv`: Contains the enhanced feature set
   - `best_xgboost_model.pkl`: Contains the trained model for deployment

## Latest Improvements (v1.2.1)
- Fixed data type handling to improve reliability across datasets
- Enhanced risk score calculation with claims data integration
- Improved metric calculations for better evaluation accuracy
- Added more robust error handling throughout the pipeline

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