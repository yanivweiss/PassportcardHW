# Changelog

## [1.2.1] - 2023-04-07

### Fixed
- Resolved issue with Member_ID data type mismatch during risk score feature interaction
- Improved risk score calculation to successfully incorporate claims data
- Fixed MAPE calculation in XGBoost evaluation metrics to handle small values correctly
- Added more robust error handling throughout the pipeline
- Fixed file existence error in main script when moving the business report

## [1.2.0] - 2023-04-07

### Added
- Advanced temporal features with sophisticated time series analysis
  - Added multi-window temporal features (30d, 60d, 90d, 180d, 365d)
  - Implemented statistical seasonality detection using seasonal decomposition
  - Added volatility metrics to capture claims variability over time
  - Created acceleration/deceleration indicators for trend analysis
- Enhanced risk scoring system
  - Implemented medically-weighted risk factors for chronic conditions
  - Added clustering-based risk segmentation
  - Created normalized risk scores scaled to 0-100 for easier interpretation
  - Implemented PCA-based dimension reduction for questionnaire data
- Advanced XGBoost modeling
  - Added advanced hyperparameter tuning with randomized search
  - Implemented early stopping and learning rate scheduling
  - Added detailed visualization of model learning curves
  - Created feature importance analysis tools
- Comprehensive business analysis and reporting
  - Developed advanced business recommendations with implementation steps
  - Created ROI analysis for proposed initiatives
  - Added phased implementation roadmap
  - Created visualizations for business insights

### Improved
- Feature engineering pipeline
  - Integrated all feature types (basic, enhanced, temporal, risk) into a unified framework
  - Added feature interactions between risk scores and claim metrics
  - Improved categorical feature handling
- Model training and evaluation
  - Added more robust cross-validation
  - Enhanced prediction visualization and analysis
  - Improved model persistence and saving
- Testing
  - Added comprehensive tests for all new modules
  - Created synthetic test data generators
  - Added validation checks for all feature engineering steps

### Documentation
- Enhanced business report with actionable recommendations
- Added visualization of business insights
- Updated code documentation for all new modules
- Added implementation roadmap with expected ROI

## [1.1.0] - 2023-04-06

### Added
- Comprehensive unit testing suite
  - Added tests for data preparation
  - Added tests for feature engineering
  - Added tests for modeling functionality
- Test data generation capabilities
  - Created synthetic claims data generator
  - Created synthetic member profiles generator
- Improved error handling for optional dependencies
  - Added graceful fallback when SHAP is not available
  - Implemented alternative feature importance calculations

### Fixed
- Resolved datetime handling with numpy integer types
- Fixed feature selection for high-dimensional data
- Improved model compatibility with various data formats

### Documentation
- Updated README with comprehensive setup instructions
- Added detailed CHANGELOG
- Enhanced business report with actionable insights
- Added code documentation for all major functions

## [1.0.0] - 2023-04-05

### Added
- Initial implementation of PassportCard claims prediction model

### Enhanced Features
- Created sophisticated temporal features for better trend analysis
  - Added claim acceleration and seasonality detection
  - Implemented quarter-over-quarter and year-over-year growth metrics
  - Added recency-based features for recent claims behavior
- Added service type profiling with entropy measurement
- Implemented comprehensive risk scoring system
  - Created category-specific risk scores (chronic, cancer, lifestyle)
  - Added demographic risk factors (age, BMI)
  - Implemented PCA-based risk component extraction
- Added interaction terms between risk scores and claim metrics
- Implemented non-linear transformations for key features
  - Added log transformations for skewed features
  - Created square and square root transformations

### Model Improvements
- Implemented an ensemble modeling approach
  - Integrated LightGBM, XGBoost, Random Forest, and Gradient Boosting
  - Added voting and stacking ensemble methods
- Added hyperparameter tuning capabilities
  - Implemented grid search for basic tuning
  - Added randomized search for advanced tuning
- Added automatic feature selection to improve model performance
  - Implemented importance-based selection
  - Added SelectFromModel for automatic thresholding
- Enhanced model evaluation with additional metrics
  - Added MAPE (Mean Absolute Percentage Error) 
  - Enhanced R² and RMSE reporting

### Visualizations and Interpretation
- Added model interpretation capabilities
  - Created feature importance visualization
  - Implemented feature distribution analysis
  - Added correlation analysis for feature relationships
- Created comprehensive business report with recommendations
  - Added risk assessment guidelines
  - Included premium optimization strategies
  - Provided customer segmentation approach
  - Outlined financial planning applications

### Technical Improvements
- Implemented modular code architecture
  - Separated data preparation, feature engineering, and modeling
  - Created enhanced features module
  - Added advanced modeling module
- Added error handling and validation checks
- Implemented model saving and loading capabilities
- Added CSV export for enhanced features
- Created visualization directory for storing plot outputs