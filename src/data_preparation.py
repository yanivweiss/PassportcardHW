import pandas as pd
import numpy as np
from datetime import datetime

def load_data():
    """Load the claims and members data"""
    claims_df = pd.read_csv('claims_data_clean.csv')
    members_df = pd.read_csv('members_data_clean.csv')
    return claims_df, members_df

def clean_dates(df, date_columns):
    """Convert date columns to datetime format"""
    for col in date_columns:
        df[col] = pd.to_datetime(df[col])
    return df

def handle_missing_values(claims_df, members_df):
    """Handle missing values in both datasets"""
    # Claims data
    claims_df['ServiceType'].fillna('Unknown', inplace=True)
    claims_df['Sex'].fillna(claims_df['Sex'].mode()[0], inplace=True)

    # Members data
    members_df['PolicyEndDate'].fillna(pd.Timestamp.now(), inplace=True)
    members_df['CountryOfOrigin'].fillna('Unknown', inplace=True)
    members_df['BMI'].fillna(members_df['BMI'].mean(), inplace=True)
    
    # Fill questionnaire nulls with 0 (assuming no condition)
    questionnaire_cols = [col for col in members_df.columns if col.startswith('Questionnaire_')]
    members_df[questionnaire_cols] = members_df[questionnaire_cols].fillna(0)
    
    return claims_df, members_df

def handle_negative_claims(claims_df):
    """Process negative claim amounts"""
    # Create adjustment flag
    claims_df['is_adjustment'] = claims_df['TotPaymentUSD'] < 0
    
    # Group related claims by ClaimNumber
    claims_grouped = claims_df.groupby('ClaimNumber').agg({
        'TotPaymentUSD': 'sum',
        'ServiceDate': 'min',
        'ServiceGroup': 'first',
        'ServiceType': 'first',
        'PolicyID': 'first',
        'Member_ID': 'first'
    }).reset_index()
    
    return claims_grouped

def calculate_member_features(claims_df, members_df):
    """Calculate additional member features"""
    # Calculate age
    members_df['Age'] = (pd.Timestamp.now() - members_df['DateOfBirth']).dt.total_seconds() / (365.25 * 24 * 60 * 60)
    
    # Calculate policy duration
    members_df['PolicyDuration'] = (members_df['PolicyEndDate'] - members_df['PolicyStartDate']).dt.total_seconds() / (365.25 * 24 * 60 * 60)
    
    # Calculate risk score based on questionnaire responses
    questionnaire_cols = [col for col in members_df.columns if col.startswith('Questionnaire_')]
    members_df['RiskScore'] = members_df[questionnaire_cols].sum(axis=1)
    
    return members_df

def prepare_data_for_modeling():
    """Main function to prepare data for modeling"""
    # Load data
    claims_df, members_df = load_data()
    
    # Clean dates
    claims_df = clean_dates(claims_df, ['ServiceDate', 'PayDate'])
    members_df = clean_dates(members_df, ['PolicyStartDate', 'PolicyEndDate', 'DateOfBirth'])
    
    # Handle missing values
    claims_df, members_df = handle_missing_values(claims_df, members_df)
    
    # Process claims
    claims_grouped = handle_negative_claims(claims_df)
    
    # Calculate additional features
    members_df = calculate_member_features(claims_df, members_df)
    
    return claims_grouped, members_df 