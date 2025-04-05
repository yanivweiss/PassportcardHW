# PassportCard Insurance Claims Prediction

## Project Overview
This project develops a predictive model for insurance claims at PassportCard. The analysis focuses on predicting the total claim amount per customer for the next six months using historical claims data and member profiles.

## Business Value
- **Risk Management**: More accurate assessment of customer risk profiles
- **Financial Planning**: Better forecasting of reserves needed for future claims
- **Premium Optimization**: Data-driven pricing based on predicted claim amounts
- **Customer Segmentation**: Identification of high-risk vs. low-risk customers

## Key Features
- Advanced temporal feature extraction with seasonality and trend detection
- Sophisticated risk scoring based on health questionnaires and demographics
- Ensemble modeling approach with multiple algorithms
- SHAP-based model interpretation for business insights
- Comprehensive business recommendations

## Technical Approach
1. **Data Preparation**: Cleaning, validation, and structuring of claims and member data
2. **Feature Engineering**: Creation of sophisticated temporal, categorical, and interaction features
3. **Model Development**: Training and optimization of ensemble models
4. **Evaluation**: Assessment of model performance with multiple metrics
5. **Interpretation**: Visualization and explanation of model predictions
6. **Deployment**: Preparation for production implementation

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
python run_enhanced_analysis.py
```

4. The results will be saved to:
   - `visualizations/`: Contains all plots and visualizations
   - `business_report.md`: Contains business recommendations
   - `enhanced_features.csv`: Contains the enhanced feature set
   - `best_model.pkl`: Contains the trained model for deployment

## Project Structure
- `data_preparation.py`: Data loading and cleaning
- `feature_engineering.py`: Basic feature creation
- `enhanced_features.py`: Advanced feature engineering
- `advanced_modeling.py`: Model training and optimization
- `run_enhanced_analysis.py`: Main execution script
- `requirements.txt`: Python package dependencies
- `CHANGELOG.md`: Record of project improvements
- `README.md`: Project documentation

## Data Description
The analysis uses two primary datasets:
1. **Claims Data**:
   - Contains historical insurance claims
   - Key fields: ClaimNumber, TotPaymentUSD, ServiceDate, ServiceGroup, Member_ID

2. **Members Data**:
   - Contains customer and policy information
   - Key fields: Member_ID, PolicyStartDate, Demographics, Questionnaire responses

## Model Performance
The model achieves strong predictive performance with:
- RMSE: ~750 USD (average prediction error)
- R²: ~0.45 (explains 45% of the variance in claim amounts)
- MAE: ~150 USD (median prediction error)

## Business Recommendations
1. **Risk Assessment**:
   - Implement the model in the underwriting process
   - Develop a risk scoring system based on model outputs
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

## Next Steps
1. Implement the model in a production environment
2. Develop a user-friendly interface for underwriters
3. Establish regular model retraining and monitoring
4. Expand the model to include additional data sources
5. Create specialized models for different claim categories

## Requirements
- Python 3.8+
- See requirements.txt for package dependencies 