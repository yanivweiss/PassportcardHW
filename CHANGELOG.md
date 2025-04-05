# Changelog

## [2.0.1] - 2025-04-05

### Changed
- Completely rewrote README.md to focus on the unbiased modeling approach
- Emphasized member-only feature engineering and its results
- Added detailed analysis of the model performance and feature importance
- Updated visualization descriptions to reflect the unbiased approach

## [2.0.0] - 2025-04-05

### Added
- New unbiased modeling approach that avoids using claim features for prediction
- Member-based feature engineering focused on demographics, health indicators, and questionnaires
- Improved model integrity by preventing data leakage between features and target

### Changed
- Completely revised feature engineering process to remove bias
- Updated pipeline to use only member attributes for prediction
- Modified visualization and reporting to reflect new model approach

### Fixed
- Addressed the circular reasoning issue where claims were used to predict future claims
- Eliminated correlation bias between independent and dependent variables
- Improved model generalizability by focusing on true predictive factors

All notable changes to this project will be documented in this file.

## [1.3.1] - 2023-04-06

### Optimized
- Evaluated the effectiveness of all advanced features, identifying which provide the most significant impact
- Customer behavior features proved highly valuable - 9 of the top 20 important features
- Most impactful customer behavior features: days_since_last_claim, claim_regularity, and claim variance metrics
- Date-based features and cyclical encoding significantly improved temporal pattern detection
- Outlier detection and handling improved model resilience, especially for extreme claim amounts

### Improved
- Model performance increased dramatically with advanced features:
  - RMSE improved by 23.32% (from 4290.81 to 3290.35)
  - MAE improved by 23.79% (from 2712.98 to 2067.51)
  - R² improved by 159.92% (from -0.35 to +0.21)
- Fixed compatibility issues with latest imbalanced-learn library
- Fixed error in regression confusion matrix calculation
- Improved Q-Q plot implementation for better error distribution analysis

## [1.3.0] - 2023-04-06

### Added
- Enhanced data preparation techniques with KNN imputation and outlier detection
- Advanced feature engineering with date-based and cyclical features
- Customer behavior metrics including claim frequency, regularity, and volatility
- Service distribution analysis with Herfindahl-Hirschman Index
- Feature selection with multiple methods (XGBoost, Lasso, SelectKBest)
- SMOTE implementation for imbalanced regression data
- Temporal cross-validation with proper time gaps
- Custom focal loss function focusing on hard-to-predict examples
- Comprehensive error analysis and visualization tools
- Regression confusion matrix for better model interpretation
- Error heatmaps to identify challenging regions in feature space

### Changed
- Improved data preprocessing pipeline with advanced scaling methods
- Enhanced model training with better handling of temporal data
- Updated README with documentation on advanced techniques
- Restructured project for better modularity and readability

## [1.2.4] - 2023-04-05

### Fixed
- Updated README.md image paths to use raw.githubusercontent.com URLs (https://raw.githubusercontent.com/yanivweiss/PassportcardHW/main/)
- Changed from github.com/raw format to raw.githubusercontent.com format for improved image rendering
- This format directly accesses GitHub's raw content servers for more reliable embedding

## [1.2.3] - 2023-04-05

### Fixed
- Updated README.md image paths to use absolute GitHub URLs (https://github.com/yanivweiss/PassportcardHW/raw/main/) for more reliable image display
- Changed from relative paths to absolute GitHub URLs to ensure consistent rendering across all platforms

## [1.2.2] - 2023-04-05

### Fixed
- Updated README.md image paths to correctly display images
- Fixed relative paths for images in subfolders (visualizations and visualizations/business_insights)
- Improved overall document formatting

## [1.2.1] - 2023-04-05

### Added
- Fixed data type handling to improve reliability across datasets
- Enhanced risk score calculation with claims data integration
- Improved metric calculations for better evaluation accuracy
- Added more robust error handling throughout the pipeline

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

## 2023-04-05
### Added
- Comprehensive fairness analysis module for evaluating model fairness across different demographic groups
- Explainability module with SHAP-based feature importance visualization
- Bias mitigation techniques including reweighting, adversarial debiasing, and fairness constraints
- Model auditing framework for regularly monitoring model performance across different segments
- Unit tests for all new components

### Changed
- Improved feature selection workflow
- Enhanced temporal cross-validation for better model evaluation
- Removed SMOTE implementation (as it's more suitable for classification than regression)
- Refactored code for better modularity and maintainability

### Fixed
- Unicode character handling in reports
- Various minor bugs and edge cases
- Improved error handling throughout the codebase

## 2023-03-22
### Added
- Advanced feature engineering techniques
  - Customer behavior features
  - Service distribution analysis
  - Cyclical encoding for date features
  - Questionnaire response analysis
  - Claims pattern recognition
  - Recency, frequency, and monetary features
  - Risk score calculations
- Enhanced data preprocessing
  - Advanced outlier detection
  - Better missing value handling
  - KNN imputation
  - Feature scaling options
- Expanded model evaluation
  - RMSE, MAE, R², MAPE metrics
  - Error analysis tools
  - Regression confusion matrix
  - Temporal cross-validation
- Feature selection methods
  - XGBoost importance
  - Lasso feature selection
  - Mutual information
  - Recursive feature elimination
- SMOTE implementation for imbalanced regression data
- Focal loss implementation for regression
- Comprehensive testing suite
- Visualizations for all major components

### Changed
- Refactored pipeline for better modularity
- Improved documentation and code comments
- Enhanced error handling

## 2023-03-10
### Added
- Initial version with basic data preparation and feature engineering
- Simple model training script
- Base feature set
- Basic visualizations

## [1.3.3] - 2023-04-07

### Removed
- Removed Custom Focal Loss implementation as it was not needed
- Simplified the codebase by focusing on standard loss functions

### Fixed
- Fixed test cases to match the updated codebase
- Ensured all tests pass after removal of focal loss

## [1.3.2] - 2023-04-07

### Changed
- Moved Model Auditing from a standalone module to the Business Recommendations section of README.md
- Fixed issues with explainability module:
  - Improved SHAP value handling for different model types
  - Enhanced error handling in comparison of global and local explanations
  - Better support for different SHAP value shapes and formats
- Fixed potential encoding issues in report generation