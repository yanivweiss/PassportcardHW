# Changelog

All notable changes to the PassportCard Insurance Claims Prediction project will be documented in this file.

## [1.1.4] - 2023-04-08

### Fixed
- Updated image paths in README.md to use correct locations for existing images:
  - Changed `risk_score_distribution.png` to use path from `outputs/figures/business_insights/`
  - Changed `model_comparison.png` to use path from `outputs/figures/feature_evaluation/`
  - Changed `residual_plot.png` to use path from `outputs/figures/predictions/`
- Removed references to non-existing images and their detailed descriptions:
  - Removed `missing_value_heatmap.png`
  - Removed `outlier_box_plot.png`
  - Removed `error_distribution_before_after_capping.png`
  - Removed `feature_interaction.png`
  - Removed `scaling_comparison.png`
  - Removed `risk_segmentation.png`
  - Removed `temporal_stability_analysis.png`
- Preserved crucial information from removed sections by incorporating it into surrounding text

## [1.1.3] - 2023-04-07

### Fixed
- Removed duplicate graph references in README.md
- Assigned unique graph paths for each visualization
- Ensured each graph is shown only once with its most relevant description
- Fixed incorrect image references for:
  - Missing Value Heatmap
  - Outlier Box Plot
  - Error Distribution
  - Risk Segmentation
  - Feature Interaction
  - Scaling Comparison
  - Model Comparison

## [1.1.2] - 2023-04-07

### Changed
- Completely reorganized README.md to create a more professional documentation
- Improved structure by grouping similar topics together
- Fixed all image paths to ensure proper rendering
- Eliminated redundancy, particularly in the SHAP analysis section
- Added clearer section hierarchy with improved heading levels
- Streamlined explanations for better readability
- Consolidated related information
- Reduced unnecessary text while preserving important details

## [1.1.1] - 2023-04-07

### Fixed
- Fixed duplicate graph references in README.md
- Ensured every graph has specific analysis tailored to its content
- Improved graph naming consistency throughout the README
- Added quantitative details to visualization analyses for better clarity
- Replaced generic explanations with graph-specific insights

## [1.1.0] - 2023-04-06

### Added
- Added comprehensive SHAP analysis section to the README with enhanced visualizations
- Added visualization modules in `src/visualization/` for generating beautiful graphs
- Added SHAP visualization utilities for model explainability
- Added visualization generation script (`src/generate_visualizations.py`)

### Changed
- Improved graph aesthetics and readability throughout the project
- Fixed broken image links in README
- Enhanced correlation heatmap visualization
- Enhanced feature importance visualization
- Enhanced predictions vs actual visualization
- Enhanced error distribution visualization
- Added detailed error analysis visualizations

### Fixed
- Fixed all broken links to graphs in README
- Fixed inconsistent path separators (backslash vs. forward slash)
- Fixed distribution comparison visualization

## [1.0.0] - 2023-04-05

### Added
- Initial release of the PassportCard Insurance Claims Prediction system
- Comprehensive README with project overview and detailed documentation
- Feature engineering pipeline
- Model development and evaluation
- Business insights and applications
- Limitations and assumptions analysis

## [3.3.0] - 2025-04-06

### Removed
- Deleted all Jupyter notebook files and related content:
  - Removed all .ipynb files from notebooks directory
  - Removed notebooks/.ipynb_checkpoints directory
  - Removed .ipynb_checkpoints directory
  - Deleted run_jupyter.py and fix_notebook.py scripts
  - Removed notebooks directory
- Updated requirements.txt to remove Jupyter dependencies:
  - Removed notebook
  - Removed jupyterlab
  - Removed nbclassic
  - Removed ipywidgets
- Updated README.md to remove all references to Jupyter notebooks
  - Updated project structure sections
  - Updated installation and usage instructions
  - Simplified project workflow

## [3.2.3] - 2023-04-06

### Fixed
- Fixed JSON structure in 2_PassportCard_Model_Development.ipynb to resolve "Notebook does not appear to be JSON" error
- Corrected formatting in the notebook's first markdown cell by properly separating the title from the description

## [3.2.2] - 2023-04-05

### Added
- Created run_jupyter.py script to fix Jupyter Notebook installation issues:
  - Adds missing notebook.app module structure if needed
  - Automatically installs and upgrades necessary Jupyter packages
  - Provides multiple fallback launch methods to ensure Jupyter starts correctly

## [3.2.1] - 2023-04-05

### Removed
- Deleted `test_notebooks_exist` method from `tests/test_notebook_generation.py` to remove redundant notebook existence verification

## [3.2.0] - 2023-04-05

### Removed
- Removed all utility scripts starting with "fix_" prefix:
  - fix_manual.py
  - fix_jupyter.py
  - fix_all_issues.py
  - fix_jupyterpath.py
  - fix_model_training.py
  - fix_model_data_prep.py
  - fix_notebook2_directly.py
  - fix_notebook2_improved.py
- Cleaned up repository structure by removing temporary fix scripts
- Removed all jupyter-related files from root folder:
  - jupyter_check.py
  - run_jupyter.bat
  - JUPYTER_GUIDE.md
  - launch_jupyter.py
  - run_jupyter.py
- Removed additional utility and setup files:
  - passportcard_fix_all.py
  - setup.bat
  - setup.py
  - update_notebook_paths.py
  - update_notebook_sequence.py
  - debug_script.py

## [1.4.0] - 2023-04-05

### Added
- Converted README.md content into comprehensive interactive Jupyter notebooks:
  - `PassportCard_Insurance_Claims_Prediction.ipynb`: Data exploration and cleaning
  - `PassportCard_Model_Development.ipynb`: Model development and evaluation
  - `PassportCard_Business_Applications.ipynb`: Business applications
- Created unit tests in `tests/test_notebook_generation.py` to verify notebook validity
- Updated notebooks directory structure and documentation
- Fixed JSON structure issues in notebooks for proper Jupyter compatibility

### Fixed
- Updated `.gitignore` to track Jupyter notebook files
- Ensured all notebooks have valid nbformat structure
- Created comprehensive notebook generation script

## [1.3.0] - 2023-04-07

### Added
- Enhanced README.md with comprehensive data science documentation:
  - Added detailed data exploration and cleaning section
  - Added in-depth feature engineering explanation
  - Added model development and evaluation details
  - Added model interpretability section
  - Added business applications section
  - Added limitations and assumptions documentation

## [1.2.0] - 2023-04-07

### Removed
- Removed unused and empty notebook files
- Removed redundant run scripts with overlapping functionality

### Changed
- Consolidated multiple run scripts into a unified pipeline
- Centralized common utility functions into shared modules
- Updated import paths for better code organization
- Created a single entry point script in the root directory (main.py)

### Added
- Added comprehensive test for the consolidated pipeline
- Added command-line options to control pipeline behavior

## [3.1.0] - 2025-04-05

### Added
- Repository reorganization for better structure and maintainability
  - Organized repository into a standardized directory structure
  - Created clear separation between code, data, outputs, and documentation
  - Added dedicated directories for models, tests, and outputs
  - Moved visualization files to outputs/figures directory
  - Reorganized Python modules into appropriate subdirectories in src/

### Changed
- Updated README.md to document the new repository structure
- Updated file paths in documentation and code references
- Consolidated related files into appropriate directories

## [3.0.0] - 2025-04-05

### Added
- Major data science workflow improvements:
  - Organized project into standardized structure (data/, notebooks/, src/, models/, reports/)
  - Created comprehensive EDA notebook with visualization and statistical analysis
  - Implemented SHAP and other model explainability tools in src/models/model_explainability.py
  - Added model validation module with temporal cross-validation in src/models/model_validation.py
  - Created model comparison framework for evaluating multiple algorithms in src/models/model_comparison.py

### Changed
- Improved modeling approach with statistical rigor:
  - Enhanced validation using temporal cross-validation
  - Added learning curves and validation curves for hyperparameter analysis
  - Implemented prediction error analysis for diagnostic insights
  - Created feature importance comparison across models

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

## [2.1.0] - 2023-04-05

### Added
- End-to-end prediction pipeline in `src/run_prediction_pipeline.py`
- Comprehensive test suite in `tests/test_prediction_pipeline.py`
- Test runner script in `run_tests.py` for running all tests
- Testing documentation in `tests/README.md`
- Performance optimizations for DataFrame operations in prediction functions
- Residual plot visualization for error analysis
- Improved error handling throughout the pipeline
- Negative prediction warnings

### Fixed
- DataFrame fragmentation warning in prediction code
- FutureWarning about numeric_only in DataFrame.corr()
- File path references to match new directory structure
- Edge cases in MAPE calculation for small target values
- Error visualization file paths

### Changed
- Improved feature creation for better performance
- Enhanced error analysis with more comprehensive metrics
- More detailed model evaluation in result reports

## [2.0.0] - 2023-04-05

### Changed
- Reorganized project structure
- Moved all Python files to `src/` directory
- Moved all test files to `tests/` directory
- Moved model files to `models/` directory
- Moved CSV files to `data/processed/` directory
- Moved image files to `outputs/figures/` directory
- Moved documentation to `docs/` directory
- Moved Jupyter notebooks to `notebooks/` directory
- Updated all file paths in code to match new directory structure
- Updated README.md to reflect new repository structure

## [1.2.0] - 2023-04-02

### Added
- Fairness analysis for model evaluation
- Group fairness metrics
- Bias detection and visualization tools
- Post-processing calibration for bias mitigation

### Fixed
- Multiple minor bugs in feature engineering
- Improved error handling in model training

## [1.1.0] - 2023-04-07

### Added
- Created comprehensive Jupyter notebooks from README content:
  - `PassportCard_Insurance_Claims_Prediction.ipynb`: Main notebook with data exploration and cleaning
  - `PassportCard_Model_Development.ipynb`: Focused on model development and evaluation
  - `PassportCard_Business_Applications.ipynb`: Demonstrates business applications of the model

### Improved
- Enhanced accessibility of project information through interactive notebooks
- Added visualizations and interactive elements to improve understanding
- Provided executable code to reproduce analysis steps and insights

## [1.0.0] - 2023-03-15

### Added
- Initial release of insurance claims prediction system
- Basic feature engineering
- XGBoost regression model
- Basic evaluation metrics
- Visualization of model performance

## Bug Fixes - April 6, 2025

### Issue: Model Training Error
- Fixed error when converting date columns to numeric values in model training
- Added explicit filtering of non-numeric columns in the prepare_for_modeling function
- Added a safety check in the evaluate_model function to ensure only numeric data is passed to models

### Issue: Jupyter Notebook.app Module Error
- Fixed "ModuleNotFoundError: No module named 'notebook.app'" when running Jupyter Notebook
- Created the missing package structure in site-packages
- Added reliable Jupyter launcher script (launch_jupyter.py) with multiple fallback options

### Additional Improvements
- Added comprehensive fix script (passportcard_fix_all.py) to address both issues at once
- Created more detailed documentation in the README
- Applied fixes with minimal changes to preserve the original notebook structure

## [1.1.1] - 2023-04-06

### Fixed
- Fixed all broken image links in README.md
- Created missing image files using appropriate alternatives
- Ensured consistent path formatting for image references
- Created image redirects for better organization and maintainability
- Added script src/fix_readme_images.py to manage image references

## [1.1.5] - 2023-04-08

### Fixed
- Updated additional image paths in README.md to correct file locations:
  - Fixed `outlier_box_plot.png` to use correct path from `outputs/figures/outliers/boxplot_TotPaymentUSD.png`
  - Fixed `error_distribution_before_after_capping.png` to reference existing `error_distribution.png`
  - Fixed `risk_score_distribution.png` to use available `feature_importance.png`
  - Fixed `missing_value_heatmap.png` to use available `error_distribution.png`
  - Updated all image paths to match the actual file structure in the repository
- Ensured all image references in README point to existing files
