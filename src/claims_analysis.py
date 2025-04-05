# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings

# Configure visualizations
plt.style.use('default')  # Using default style instead of seaborn
sns.set_theme()  # Set seaborn theme
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Load the datasets
claims_df = pd.read_csv('claims_data_clean.csv')
members_df = pd.read_csv('members_data_clean.csv')

print("Claims Data Shape:", claims_df.shape)
print("\nMembers Data Shape:", members_df.shape)

print("\nClaims Data Columns:")
print(claims_df.columns.tolist())

print("\nMembers Data Columns:")
print(members_df.columns.tolist())

print("\nClaims Data Info:")
claims_df.info()

print("\nMembers Data Info:")
members_df.info()

# Display first few rows of each dataset
print("\nClaims Data Sample:")
print(claims_df.head())

print("\nMembers Data Sample:")
print(members_df.head())

# Basic statistics for numerical columns
print("\nClaims Data Statistics:")
print(claims_df.describe())

print("\nMembers Data Statistics:")
print(members_df.describe()) 