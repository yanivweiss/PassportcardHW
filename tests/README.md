# Tests for PassportCard Insurance Claims Prediction System

This directory contains tests for the insurance claims prediction system. The tests validate the functionality of various components such as data preparation, feature engineering, model training, and prediction.

## How to Run Tests

### Run All Tests

To run all tests, use the following command from the project root directory:

```bash
python run_tests.py
```

For more detailed output:

```bash
python run_tests.py --verbose
```

### Run Specific Tests

To run a specific test class or method:

```bash
python run_tests.py --test TestPredictionPipeline
```

You can also run individual test modules directly:

```bash
python -m tests.test_prediction_pipeline
python -m tests.test_predictions
python -m tests.test_bias_mitigation
```

## Available Test Modules

1. **test_prediction_pipeline.py** - Tests the end-to-end prediction pipeline
2. **test_predictions.py** - Tests loading the model and making predictions
3. **test_bias_mitigation.py** - Tests bias detection and mitigation techniques
4. **test_explainability.py** - Tests model explainability methods
5. **test_fairness_analysis.py** - Tests fairness analysis metrics and visualizations
6. **test_advanced_features.py** - Tests advanced feature engineering techniques
7. **test_xgboost_modeling.py** - Tests XGBoost model training and evaluation

## Interpreting Test Results

When running tests, you'll see output like:

```
.....F....

======================================================================
FAIL: test_name (tests.module.TestClass)
----------------------------------------------------------------------
Traceback (most recent call last):
  ...
AssertionError: ...
```

- `.` indicates a passing test
- `F` indicates a failing test
- `E` indicates an error occurred during test execution
- `s` indicates a skipped test

### Test Summary

At the end of the test run, you'll see a summary showing:

- Total number of tests run
- Number of successful tests
- Number of failed tests
- List of failed tests
- Total time elapsed

## Adding New Tests

To add new tests:

1. Create a new Python file named `test_*.py` in the tests directory
2. Import the `unittest` module
3. Create one or more test classes that inherit from `unittest.TestCase`
4. Add test methods that start with `test_`
5. Use assertions to validate expected behavior

Example:

```python
import unittest

class TestMyFeature(unittest.TestCase):
    def test_something(self):
        result = my_function()
        self.assertEqual(result, expected_value)
```

## Test Data

Some tests require test data to run properly. Most tests will skip automatically if the required data is not available. The test data should be placed in the appropriate directories:

- `data/processed/` - Processed data files
- `models/` - Saved model files

## Continuous Integration

Tests are automatically run whenever code is pushed to the repository. The CI workflow checks that all tests pass before allowing code to be merged into the main branch. 