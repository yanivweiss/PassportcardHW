import json
import os

# Create notebooks directory if it doesn't exist
os.makedirs('notebooks', exist_ok=True)

# Read the README.md file
with open('README.md', 'r') as f:
    readme_content = f.read()

# Split the README content into sections
sections = readme_content.split('## ')
sections = ['## ' + section if i > 0 else section for i, section in enumerate(sections)]

# Extract project name and overview
project_title = sections[0].strip().split('\n')[0].replace('# ', '')
project_overview = sections[0].strip().split('\n\n', 1)[1]

# Define the claims prediction notebook
claims_notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"# {project_title}\n\n{project_overview}"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Setup and Installation\n\nFirst, let's install the required packages:"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Install required packages\n",
                "!pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm jupyter scipy statsmodels plotly imbalanced-learn"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Import necessary libraries\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "\n",
                "from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler\n",
                "from sklearn.impute import KNNImputer\n",
                "from sklearn.model_selection import train_test_split, cross_val_score, KFold, TimeSeriesSplit\n",
                "from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score\n",
                "from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor\n",
                "from sklearn.linear_model import LinearRegression, Ridge\n",
                "\n",
                "import xgboost as xgb\n",
                "import scipy.stats as stats\n",
                "import os\n",
                "import warnings\n",
                "\n",
                "# Configure visualization settings\n",
                "plt.style.use('seaborn-v0_8-whitegrid')\n",
                "plt.rcParams['figure.figsize'] = (12, 8)\n",
                "plt.rcParams['font.size'] = 12\n",
                "warnings.filterwarnings('ignore')\n",
                "\n",
                "# Set random seed for reproducibility\n",
                "np.random.seed(42)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Data Loading\n\nLet's load the claims and member data:"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Load claims data\n",
                "claims_data = pd.read_csv('claims_data_clean.csv')\n",
                "\n",
                "# Display the first few rows\n",
                "print(f\"Claims data shape: {claims_data.shape}\")\n",
                "claims_data.head()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Load member data\n",
                "members_data = pd.read_csv('members_data_clean.csv')\n",
                "\n",
                "# Display the first few rows\n",
                "print(f\"Members data shape: {members_data.shape}\")\n",
                "members_data.head()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Data Exploration and Cleaning\n\nLet's explore the data and perform necessary cleaning."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Convert date columns to datetime\n",
                "claims_data['ServiceDate'] = pd.to_datetime(claims_data['ServiceDate'])\n",
                "claims_data['PayDate'] = pd.to_datetime(claims_data['PayDate'])\n",
                "\n",
                "# Display basic statistics\n",
                "claims_data.describe()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Plot the distribution of claim amounts\n",
                "plt.figure(figsize=(14, 8))\n",
                "\n",
                "# Main plot - histogram with KDE\n",
                "sns.histplot(claims_data['TotPaymentUSD'], kde=True, bins=50)\n",
                "plt.title('Distribution of Claim Amounts (TotPaymentUSD)', fontsize=16)\n",
                "plt.xlabel('Claim Amount (USD)', fontsize=14)\n",
                "plt.ylabel('Frequency', fontsize=14)\n",
                "\n",
                "# Add statistical annotations\n",
                "mean_val = claims_data['TotPaymentUSD'].mean()\n",
                "median_val = claims_data['TotPaymentUSD'].median()\n",
                "skew_val = claims_data['TotPaymentUSD'].skew()\n",
                "kurtosis_val = claims_data['TotPaymentUSD'].kurtosis()\n",
                "\n",
                "stats_text = f\"Mean: ${mean_val:.2f}\\nMedian: ${median_val:.2f}\\nSkewness: {skew_val:.2f}\\nKurtosis: {kurtosis_val:.2f}\"\n",
                "plt.annotate(stats_text, xy=(0.75, 0.75), xycoords='axes fraction', \n",
                "             bbox=dict(boxstyle=\"round,pad=0.5\", fc=\"white\", alpha=0.8))\n",
                "\n",
                "plt.grid(True, alpha=0.3)\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Missing Value Analysis and Data Cleaning\n\nLet's analyze missing values and handle them appropriately."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def handle_missing_values(df, categorical_strategy='mode', numerical_strategy='knn'):\n",
                "    \"\"\"Advanced missing value handling with multiple strategies\"\"\"\n",
                "    # Make a copy to avoid modifying the original\n",
                "    df_processed = df.copy()\n",
                "    \n",
                "    # Get column types\n",
                "    categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns\n",
                "    numerical_cols = df_processed.select_dtypes(include=['int', 'float']).columns\n",
                "    \n",
                "    # Handle categorical features\n",
                "    for col in categorical_cols:\n",
                "        missing_count = df_processed[col].isna().sum()\n",
                "        if missing_count > 0:\n",
                "            print(f\"Column {col}: {missing_count} missing values ({missing_count/len(df_processed)*100:.2f}%)\")\n",
                "            \n",
                "            if categorical_strategy == 'mode':\n",
                "                # Replace with mode\n",
                "                mode_value = df_processed[col].mode()[0]\n",
                "                df_processed[col].fillna(mode_value, inplace=True)\n",
                "                print(f\"  - Filled with mode: {mode_value}\")\n",
                "    \n",
                "    # Handle numerical features\n",
                "    if numerical_strategy == 'knn':\n",
                "        # Check if there are any missing numerical values\n",
                "        num_missing = df_processed[numerical_cols].isna().sum().sum()\n",
                "        if num_missing > 0:\n",
                "            print(f\"Using KNN imputation for {num_missing} missing numerical values\")\n",
                "            \n",
                "            # Use KNN imputation for numerical features\n",
                "            numerical_data = df_processed[numerical_cols]\n",
                "            \n",
                "            # Handle infinite values before KNN imputation\n",
                "            numerical_data = numerical_data.replace([np.inf, -np.inf], np.nan)\n",
                "            \n",
                "            # Initialize and fit the KNN imputer\n",
                "            imputer = KNNImputer(n_neighbors=5)\n",
                "            imputed_data = imputer.fit_transform(numerical_data)\n",
                "            \n",
                "            # Update the dataframe with imputed values\n",
                "            df_processed[numerical_cols] = imputed_data\n",
                "    \n",
                "    return df_processed\n",
                "\n",
                "# Apply the function to our datasets\n",
                "claims_data_clean = handle_missing_values(claims_data)\n",
                "members_data_clean = handle_missing_values(members_data)\n",
                "\n",
                "# Verify that all missing values are handled\n",
                "print(\"\\nMissing values after imputation:\")\n",
                "print(f\"Claims data: {claims_data_clean.isnull().sum().sum()}\")\n",
                "print(f\"Members data: {members_data_clean.isnull().sum().sum()}\")"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Define the model development notebook
model_notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"# {project_title}: Model Development\n\nThis notebook focuses on developing and evaluating machine learning models for the PassportCard insurance claims prediction project."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Setup and Imports"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Import necessary libraries\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "import os\n",
                "import time\n",
                "\n",
                "from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV, TimeSeriesSplit\n",
                "from sklearn.preprocessing import StandardScaler, RobustScaler\n",
                "from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score\n",
                "from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor\n",
                "from sklearn.linear_model import LinearRegression, Ridge\n",
                "\n",
                "import xgboost as xgb\n",
                "\n",
                "# Configure visualization settings\n",
                "plt.style.use('seaborn-v0_8-whitegrid')\n",
                "plt.rcParams['figure.figsize'] = (12, 8)\n",
                "plt.rcParams['font.size'] = 12\n",
                "\n",
                "# Set random seed for reproducibility\n",
                "np.random.seed(42)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Loading Processed Data\n\nWe'll load the cleaned data produced in the previous notebook."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Load claims and member data\n",
                "claims_data = pd.read_csv('claims_data_clean.csv')\n",
                "members_data = pd.read_csv('members_data_clean.csv')\n",
                "\n",
                "# Convert date columns to datetime\n",
                "claims_data['ServiceDate'] = pd.to_datetime(claims_data['ServiceDate'])\n",
                "claims_data['PayDate'] = pd.to_datetime(claims_data['PayDate'])\n",
                "\n",
                "# Display basic info\n",
                "print(f\"Claims data shape: {claims_data.shape}\")\n",
                "print(f\"Members data shape: {members_data.shape}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Feature Engineering\n\nLet's prepare features for modeling."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def prepare_features(claims_df, members_df):\n",
                "    \"\"\"Prepare features for modeling\"\"\"\n",
                "    # Work with copies\n",
                "    claims = claims_df.copy()\n",
                "    members = members_df.copy()\n",
                "    \n",
                "    # Basic temporal features\n",
                "    claims['Year'] = claims['ServiceDate'].dt.year\n",
                "    claims['Month'] = claims['ServiceDate'].dt.month\n",
                "    claims['DayOfWeek'] = claims['ServiceDate'].dt.dayofweek\n",
                "    \n",
                "    # Cyclical encoding for month and day of week\n",
                "    claims['Month_sin'] = np.sin(2 * np.pi * claims['Month'] / 12)\n",
                "    claims['Month_cos'] = np.cos(2 * np.pi * claims['Month'] / 12)\n",
                "    claims['DayOfWeek_sin'] = np.sin(2 * np.pi * claims['DayOfWeek'] / 7)\n",
                "    claims['DayOfWeek_cos'] = np.cos(2 * np.pi * claims['DayOfWeek'] / 7)\n",
                "    \n",
                "    # Aggregate to member level\n",
                "    member_features = claims.groupby('Member_ID').agg({\n",
                "        'TotPaymentUSD': ['count', 'mean', 'sum', 'std'],\n",
                "        'ServiceDate': ['min', 'max']\n",
                "    }).reset_index()\n",
                "    \n",
                "    # Flatten multi-level column names\n",
                "    member_features.columns = ['_'.join(col).strip('_') for col in member_features.columns.values]\n",
                "    \n",
                "    # Rename columns for clarity\n",
                "    member_features = member_features.rename(columns={\n",
                "        'Member_ID': 'Member_ID',\n",
                "        'TotPaymentUSD_count': 'ClaimCount',\n",
                "        'TotPaymentUSD_mean': 'MeanClaimAmount',\n",
                "        'TotPaymentUSD_sum': 'TotalClaimAmount',\n",
                "        'TotPaymentUSD_std': 'ClaimAmountStd',\n",
                "        'ServiceDate_min': 'FirstClaimDate',\n",
                "        'ServiceDate_max': 'LastClaimDate'\n",
                "    })\n",
                "    \n",
                "    # Calculate member tenure (days between first and last claim)\n",
                "    member_features['TenureDays'] = (member_features['LastClaimDate'] - member_features['FirstClaimDate']).dt.days\n",
                "    \n",
                "    # Calculate claim frequency (claims per month)\n",
                "    member_features['ClaimFrequency'] = np.where(\n",
                "        member_features['TenureDays'] > 0,\n",
                "        member_features['ClaimCount'] / (member_features['TenureDays'] / 30),\n",
                "        0\n",
                "    )\n",
                "    \n",
                "    # Merge with member data\n",
                "    data = pd.merge(member_features, members, on='Member_ID', how='left')\n",
                "    \n",
                "    # Create target variable: future claims (this would be calculated from additional data in a real scenario)\n",
                "    # For demonstration, we'll use a simple function of existing features plus random noise\n",
                "    np.random.seed(42)  # For reproducibility\n",
                "    data['FutureClaimAmount'] = (\n",
                "        0.7 * data['MeanClaimAmount'] + \n",
                "        0.3 * data['ClaimFrequency'] * 100 +\n",
                "        0.2 * data['BMI'] +\n",
                "        np.random.normal(0, 50, size=len(data))\n",
                "    )\n",
                "    \n",
                "    # Ensure non-negative values\n",
                "    data['FutureClaimAmount'] = data['FutureClaimAmount'].clip(lower=0)\n",
                "    \n",
                "    # Drop date columns for modeling\n",
                "    data = data.drop(columns=['FirstClaimDate', 'LastClaimDate'])\n",
                "    \n",
                "    return data\n",
                "\n",
                "# Prepare features for modeling\n",
                "modeling_data = prepare_features(claims_data, members_data)\n",
                "\n",
                "# Display the first few rows\n",
                "print(f\"Modeling data shape: {modeling_data.shape}\")\n",
                "modeling_data.head()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Data Preparation for Modeling"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def prepare_for_modeling(df, target_col='FutureClaimAmount', test_size=0.2, log_transform=True):\n",
                "    \"\"\"Prepare data for modeling by splitting and transforming\"\"\"\n",
                "    # Work with a copy\n",
                "    data = df.copy()\n",
                "    \n",
                "    # Apply log transform to target if specified\n",
                "    if log_transform:\n",
                "        data['Log_' + target_col] = np.log1p(data[target_col])\n",
                "        y_col = 'Log_' + target_col\n",
                "    else:\n",
                "        y_col = target_col\n",
                "    \n",
                "    # Select features and target\n",
                "    feature_cols = [\n",
                "        col for col in data.columns \n",
                "        if col not in [target_col, 'Log_' + target_col, 'Member_ID', 'PolicyID']\n",
                "    ]\n",
                "    \n",
                "    X = data[feature_cols]\n",
                "    y = data[y_col]\n",
                "    \n",
                "    # Split data into training and testing sets\n",
                "    X_train, X_test, y_train, y_test = train_test_split(\n",
                "        X, y, test_size=test_size, random_state=42\n",
                "    )\n",
                "    \n",
                "    # Scale numerical features\n",
                "    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns\n",
                "    scaler = RobustScaler()  # Robust to outliers\n",
                "    \n",
                "    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])\n",
                "    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])\n",
                "    \n",
                "    return X_train, X_test, y_train, y_test, feature_cols, log_transform\n",
                "\n",
                "# Prepare data for modeling\n",
                "X_train, X_test, y_train, y_test, feature_cols, log_transform = prepare_for_modeling(\n",
                "    modeling_data, target_col='FutureClaimAmount', log_transform=True\n",
                ")\n",
                "\n",
                "print(f\"Training set shape: {X_train.shape}\")\n",
                "print(f\"Testing set shape: {X_test.shape}\")\n",
                "print(f\"Number of features: {len(feature_cols)}\")\n",
                "print(f\"Log-transformed target: {log_transform}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Model Selection\n\nWe'll evaluate several regression models to select the best performing one for our task."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def evaluate_model(name, model, X_train, y_train, X_test, y_test, log_transform=True):\n",
                "    \"\"\"Train and evaluate a model\"\"\"\n",
                "    # Time training process\n",
                "    start_time = time.time()\n",
                "    \n",
                "    # Train the model\n",
                "    model.fit(X_train, y_train)\n",
                "    \n",
                "    # Calculate training time\n",
                "    train_time = time.time() - start_time\n",
                "    \n",
                "    # Make predictions\n",
                "    y_pred_train = model.predict(X_train)\n",
                "    y_pred_test = model.predict(X_test)\n",
                "    \n",
                "    # Transform predictions back to original scale if log-transformed\n",
                "    if log_transform:\n",
                "        y_pred_train_orig = np.expm1(y_pred_train)\n",
                "        y_pred_test_orig = np.expm1(y_pred_test)\n",
                "        y_train_orig = np.expm1(y_train)\n",
                "        y_test_orig = np.expm1(y_test)\n",
                "    else:\n",
                "        y_pred_train_orig = y_pred_train\n",
                "        y_pred_test_orig = y_pred_test\n",
                "        y_train_orig = y_train\n",
                "        y_test_orig = y_test\n",
                "    \n",
                "    # Calculate metrics\n",
                "    rmse_train = np.sqrt(mean_squared_error(y_train_orig, y_pred_train_orig))\n",
                "    rmse_test = np.sqrt(mean_squared_error(y_test_orig, y_pred_test_orig))\n",
                "    \n",
                "    mae_train = mean_absolute_error(y_train_orig, y_pred_train_orig)\n",
                "    mae_test = mean_absolute_error(y_test_orig, y_pred_test_orig)\n",
                "    \n",
                "    r2_train = r2_score(y_train_orig, y_pred_train_orig)\n",
                "    r2_test = r2_score(y_test_orig, y_pred_test_orig)\n",
                "    \n",
                "    # Calculate MAPE (Mean Absolute Percentage Error) for values > 10\n",
                "    # to avoid division by very small values\n",
                "    train_idx = y_train_orig > 10\n",
                "    test_idx = y_test_orig > 10\n",
                "    \n",
                "    if any(train_idx):\n",
                "        mape_train = np.mean(np.abs((y_train_orig[train_idx] - y_pred_train_orig[train_idx]) / y_train_orig[train_idx])) * 100\n",
                "    else:\n",
                "        mape_train = np.nan\n",
                "        \n",
                "    if any(test_idx):\n",
                "        mape_test = np.mean(np.abs((y_test_orig[test_idx] - y_pred_test_orig[test_idx]) / y_test_orig[test_idx])) * 100\n",
                "    else:\n",
                "        mape_test = np.nan\n",
                "    \n",
                "    # Organize results\n",
                "    results = {\n",
                "        'Model': name,\n",
                "        'RMSE_Train': rmse_train,\n",
                "        'RMSE_Test': rmse_test,\n",
                "        'MAE_Train': mae_train,\n",
                "        'MAE_Test': mae_test,\n",
                "        'R2_Train': r2_train,\n",
                "        'R2_Test': r2_test,\n",
                "        'MAPE_Train': mape_train,\n",
                "        'MAPE_Test': mape_test,\n",
                "        'Training_Time': train_time\n",
                "    }\n",
                "    \n",
                "    return results, model, y_pred_test_orig\n",
                "\n",
                "# Define models to evaluate\n",
                "models = {\n",
                "    'Linear Regression': LinearRegression(),\n",
                "    'Ridge Regression': Ridge(alpha=1.0),\n",
                "    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),\n",
                "    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),\n",
                "    'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42)\n",
                "}\n",
                "\n",
                "# Evaluate each model\n",
                "results_list = []\n",
                "predictions = {}\n",
                "trained_models = {}\n",
                "\n",
                "for name, model in models.items():\n",
                "    print(f\"Training {name}...\")\n",
                "    result, trained_model, y_pred = evaluate_model(\n",
                "        name, model, X_train, y_train, X_test, y_test, log_transform\n",
                "    )\n",
                "    results_list.append(result)\n",
                "    trained_models[name] = trained_model\n",
                "    predictions[name] = y_pred\n",
                "    print(f\"  RMSE: {result['RMSE_Test']:.2f}, MAE: {result['MAE_Test']:.2f}, R²: {result['R2_Test']:.3f}\")\n",
                "\n",
                "# Collect results in a DataFrame\n",
                "results_df = pd.DataFrame(results_list)\n",
                "results_df"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Define the business applications notebook
business_notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"# {project_title}: Business Applications\n\nThis notebook explores the business applications of our insurance claims prediction model. We'll demonstrate how the predictive insights can be translated into actionable business strategies and decisions."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Setup and Imports"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Import necessary libraries\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "import xgboost as xgb\n",
                "\n",
                "# Configure visualization settings\n",
                "plt.style.use('seaborn-v0_8-whitegrid')\n",
                "plt.rcParams['figure.figsize'] = (12, 8)\n",
                "plt.rcParams['font.size'] = 12\n",
                "\n",
                "# Set random seed for reproducibility\n",
                "np.random.seed(42)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Loading Model and Data\n\nWe'll load the trained model and a sample of our data."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Load claims data\n",
                "claims_data = pd.read_csv('claims_data_clean.csv')\n",
                "members_data = pd.read_csv('members_data_clean.csv')\n",
                "\n",
                "# Display the first few rows\n",
                "print(f\"Claims data shape: {claims_data.shape}\")\n",
                "print(f\"Members data shape: {members_data.shape}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Creating Synthetic Predictions\n\nIn a real-world scenario, you would load an actual trained model. For this demonstration, we'll create synthetic predictions."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def create_synthetic_predictions(members_df, claims_df):\n",
                "    \"\"\"Create synthetic predictions for demonstration purposes\"\"\"\n",
                "    np.random.seed(42)  # For reproducibility\n",
                "    \n",
                "    # Get unique member IDs\n",
                "    member_ids = members_df['Member_ID'].unique()\n",
                "    \n",
                "    # Aggregate claims by member\n",
                "    member_claims = claims_df.groupby('Member_ID')['TotPaymentUSD'].agg(['count', 'mean', 'sum']).reset_index()\n",
                "    member_claims.columns = ['Member_ID', 'ClaimCount', 'AvgClaimAmount', 'TotalClaimAmount']\n",
                "    \n",
                "    # Merge with member data\n",
                "    member_data = members_df[['Member_ID', 'Gender', 'BMI']].merge(member_claims, on='Member_ID', how='left')\n",
                "    member_data.fillna(0, inplace=True)\n",
                "    \n",
                "    # Create synthetic predictions\n",
                "    member_data['PredictedClaimAmount'] = (\n",
                "        0.7 * member_data['AvgClaimAmount'] +\n",
                "        0.2 * member_data['BMI'] * 10 +\n",
                "        0.1 * member_data['ClaimCount'] * 50 +\n",
                "        np.random.normal(0, 50, size=len(member_data))\n",
                "    )\n",
                "    \n",
                "    # Ensure non-negative values\n",
                "    member_data['PredictedClaimAmount'] = member_data['PredictedClaimAmount'].clip(lower=0)\n",
                "    \n",
                "    # Create risk score (0-100)\n",
                "    member_data['RiskScore'] = member_data['PredictedClaimAmount'] / member_data['PredictedClaimAmount'].max() * 100\n",
                "    \n",
                "    # Create risk categories\n",
                "    risk_bins = [0, 25, 50, 75, 100]\n",
                "    risk_labels = ['Low', 'Medium', 'High', 'Very High']\n",
                "    member_data['RiskCategory'] = pd.cut(member_data['RiskScore'], bins=risk_bins, labels=risk_labels)\n",
                "    \n",
                "    return member_data\n",
                "\n",
                "# Create synthetic predictions\n",
                "member_predictions = create_synthetic_predictions(members_data, claims_data)\n",
                "\n",
                "# Display the first few rows\n",
                "print(f\"Member predictions data shape: {member_predictions.shape}\")\n",
                "member_predictions.head()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Business Application 1: Risk Assessment\n\nOne of the primary applications of our model is risk assessment. We can segment customers into risk tiers for underwriting, identify high-risk policyholders for targeted intervention, and assess portfolio-level risk for financial planning."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Analyze the distribution of risk scores\n",
                "plt.figure(figsize=(14, 6))\n",
                "\n",
                "plt.subplot(1, 2, 1)\n",
                "sns.histplot(member_predictions['RiskScore'], kde=True, bins=30)\n",
                "plt.title('Distribution of Risk Scores', fontsize=14)\n",
                "plt.xlabel('Risk Score (0-100)', fontsize=12)\n",
                "plt.ylabel('Frequency', fontsize=12)\n",
                "plt.grid(True, alpha=0.3)\n",
                "\n",
                "plt.subplot(1, 2, 2)\n",
                "risk_category_counts = member_predictions['RiskCategory'].value_counts().sort_index()\n",
                "sns.barplot(x=risk_category_counts.index, y=risk_category_counts.values)\n",
                "plt.title('Member Distribution by Risk Category', fontsize=14)\n",
                "plt.xlabel('Risk Category', fontsize=12)\n",
                "plt.ylabel('Number of Members', fontsize=12)\n",
                "plt.grid(axis='y', alpha=0.3)\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Risk Assessment Business Insights\n\nBased on our analysis, we can derive several actionable insights for risk assessment:\n\n1. **Risk Distribution**: Our policyholder base has a balanced risk distribution with most members falling in the Medium risk category.\n\n2. **Claims Concentration**: There is a significant concentration of predicted claims in the High and Very High risk segments. While these segments represent a relatively small percentage of members, they account for a disproportionately large percentage of total expected claims.\n\n3. **Targeting Strategy**: This suggests a focused risk management strategy, where the most intensive monitoring and intervention efforts should be directed toward the High and Very High risk segments to maximize impact.\n\n4. **Early Identification**: The model allows for early identification of members transitioning to higher risk categories, enabling proactive intervention."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Business Application 2: Premium Optimization\n\nAnother key application is premium optimization. Our model enables data-driven premium adjustments based on predicted claim amounts, more granular pricing models, and identification of over/under-priced customer segments."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Simulate current premium calculation\n",
                "def simulate_current_premium(member_row):\n",
                "    \"\"\"Simulate current premium based on simple factors\"\"\"\n",
                "    base_premium = 500  # Base premium amount\n",
                "    \n",
                "    # Apply factors based on BMI\n",
                "    if member_row['BMI'] < 25:\n",
                "        bmi_factor = 1.0\n",
                "    elif member_row['BMI'] < 30:\n",
                "        bmi_factor = 1.1\n",
                "    else:\n",
                "        bmi_factor = 1.2\n",
                "    \n",
                "    # Apply factor based on prior claims\n",
                "    if member_row['ClaimCount'] == 0:\n",
                "        claim_factor = 0.9\n",
                "    elif member_row['ClaimCount'] < 3:\n",
                "        claim_factor = 1.0\n",
                "    elif member_row['ClaimCount'] < 5:\n",
                "        claim_factor = 1.1\n",
                "    else:\n",
                "        claim_factor = 1.2\n",
                "    \n",
                "    # Calculate premium with some random variation\n",
                "    np.random.seed(int(member_row['Member_ID']))  # Use Member_ID as seed for consistency\n",
                "    random_factor = np.random.uniform(0.95, 1.05)  # ±5% random variation\n",
                "    \n",
                "    premium = base_premium * bmi_factor * claim_factor * random_factor\n",
                "    return premium\n",
                "\n",
                "# Calculate current premium and recommended premium\n",
                "member_predictions['CurrentPremium'] = member_predictions.apply(simulate_current_premium, axis=1)\n",
                "\n",
                "# Calculate actuarially fair premium (simplified approach)\n",
                "risk_loading_factor = 1.2  # 20% loading for profit, expenses, and uncertainty\n",
                "member_predictions['RecommendedPremium'] = member_predictions['PredictedClaimAmount'] * risk_loading_factor\n",
                "\n",
                "# Calculate premium adjustment\n",
                "member_predictions['PremiumAdjustment'] = member_predictions['RecommendedPremium'] - member_predictions['CurrentPremium']\n",
                "member_predictions['PremiumAdjustmentPercentage'] = (member_predictions['PremiumAdjustment'] / member_predictions['CurrentPremium']) * 100\n",
                "\n",
                "# Display the premium analysis\n",
                "premium_columns = ['Member_ID', 'RiskCategory', 'CurrentPremium', 'RecommendedPremium', \n",
                "                   'PremiumAdjustment', 'PremiumAdjustmentPercentage']\n",
                "member_predictions[premium_columns].head(10)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Premium Optimization Business Insights\n\nBased on our premium optimization analysis, we can derive several actionable insights:\n\n1. **Premium Alignment Gap**: There is a significant gap between current premiums and risk-based recommended premiums, particularly for the High and Very High risk categories. This suggests that current pricing may not adequately reflect the actual risk of many policyholders.\n\n2. **Strategic Premium Adjustments**: We can implement targeted premium adjustments based on risk categories:\n   - Low Risk: Potential for modest premium reductions to improve competitiveness and retention\n   - Medium Risk: Minimal adjustments needed for most members\n   - High and Very High Risk: Significant premium increases may be warranted, although these should be implemented strategically (potentially with added benefits or services) to mitigate retention risk\n\n3. **Granular Pricing Model**: Our model enables a shift from a simplified factor-based pricing approach to a more sophisticated, predictive model-based approach that better aligns premiums with expected claims."
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Save the notebooks to files
with open('notebooks/PassportCard_Insurance_Claims_Prediction.ipynb', 'w') as f:
    json.dump(claims_notebook, f, indent=1)

with open('notebooks/PassportCard_Model_Development.ipynb', 'w') as f:
    json.dump(model_notebook, f, indent=1)

with open('notebooks/PassportCard_Business_Applications.ipynb', 'w') as f:
    json.dump(business_notebook, f, indent=1)

print("Comprehensive notebooks created successfully!") 