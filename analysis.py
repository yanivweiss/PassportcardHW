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
plt.style.use('default')
sns.set_theme()
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def load_and_inspect_data():
    """Load and inspect the datasets"""
    # Load the datasets
    claims_df = pd.read_csv('claims_data_clean.csv')
    members_df = pd.read_csv('members_data_clean.csv')

    print("Claims Data Shape:", claims_df.shape)
    print("\nMembers Data Shape:", members_df.shape)

    print("\nClaims Data Info:")
    claims_df.info()

    print("\nMembers Data Info:")
    members_df.info()

    return claims_df, members_df

def check_data_quality(claims_df, members_df):
    """Check data quality issues"""
    # Check missing values
    print("Missing values in claims data:")
    print(claims_df.isnull().sum()[claims_df.isnull().sum() > 0])

    print("\nMissing values in members data:")
    print(members_df.isnull().sum()[members_df.isnull().sum() > 0])

    # Check duplicates
    print("\nDuplicate claims:", claims_df.duplicated().sum())
    print("Duplicate members:", members_df.duplicated().sum())

def analyze_claims(claims_df):
    """Analyze claims data"""
    # Convert date columns to datetime
    claims_df['ServiceDate'] = pd.to_datetime(claims_df['ServiceDate'])
    claims_df['PayDate'] = pd.to_datetime(claims_df['PayDate'])

    # Analyze claims by service group
    service_group_stats = claims_df.groupby('ServiceGroup')['TotPaymentUSD'].agg(['count', 'mean', 'sum']).sort_values('sum', ascending=False)
    print("\nClaims by Service Group:")
    print(service_group_stats)

    # Plot claims distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(data=claims_df, x='TotPaymentUSD', bins=50)
    plt.title('Distribution of Claim Amounts')
    plt.xlabel('Claim Amount (USD)')
    plt.ylabel('Count')
    plt.savefig('claims_distribution.png')
    plt.close()

    # Analyze claims over time
    claims_by_date = claims_df.groupby('ServiceDate')['TotPaymentUSD'].agg(['count', 'sum']).reset_index()
    claims_by_date.set_index('ServiceDate', inplace=True)

    plt.figure(figsize=(15, 6))
    plt.plot(claims_by_date.index, claims_by_date['sum'])
    plt.title('Total Claims Amount Over Time')
    plt.xlabel('Date')
    plt.ylabel('Total Claims Amount (USD)')
    plt.savefig('claims_over_time.png')
    plt.close()

def analyze_members(members_df):
    """Analyze members data"""
    # Convert date columns to datetime
    members_df['PolicyStartDate'] = pd.to_datetime(members_df['PolicyStartDate'])
    members_df['PolicyEndDate'] = pd.to_datetime(members_df['PolicyEndDate'])
    members_df['DateOfBirth'] = pd.to_datetime(members_df['DateOfBirth'])

    # Analyze member demographics
    print("\nMember Demographics:")
    print("\nCountry of Origin Distribution:")
    print(members_df['CountryOfOrigin'].value_counts().head())

    print("\nCountry of Destination Distribution:")
    print(members_df['CountryOfDestination'].value_counts().head())

    print("\nPayer Type Distribution:")
    print(members_df['PayerType'].value_counts())

    # Plot BMI distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=members_df, x='BMI', bins=30)
    plt.title('Distribution of Member BMI')
    plt.xlabel('BMI')
    plt.ylabel('Count')
    plt.savefig('bmi_distribution.png')
    plt.close()

def main():
    """Main analysis function"""
    print("Loading and inspecting data...")
    claims_df, members_df = load_and_inspect_data()

    print("\nChecking data quality...")
    check_data_quality(claims_df, members_df)

    print("\nAnalyzing claims data...")
    analyze_claims(claims_df)

    print("\nAnalyzing members data...")
    analyze_members(members_df)

if __name__ == "__main__":
    main() 