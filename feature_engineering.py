import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def calculate_historical_features(claims_df, cutoff_date=None):
    """Calculate historical claim features for each member"""
    if cutoff_date is None:
        cutoff_date = claims_df['ServiceDate'].max()
    
    # Filter claims before cutoff date
    historical_claims = claims_df[claims_df['ServiceDate'] <= cutoff_date]
    
    # Calculate features per member
    member_features = historical_claims.groupby('Member_ID').agg({
        'TotPaymentUSD': ['count', 'sum', 'mean', 'std', 'max'],
        'ServiceGroup': 'nunique',
        'ServiceType': 'nunique',
        'ServiceDate': lambda x: (cutoff_date - x.min()).days / 365.25  # history length in years
    }).reset_index()
    
    # Flatten column names
    member_features.columns = ['Member_ID', 'claim_count', 'total_claims', 'avg_claim', 
                             'std_claim', 'max_claim', 'unique_service_groups', 
                             'unique_service_types', 'history_years']
    
    return member_features

def calculate_service_type_features(claims_df, member_id):
    """Calculate features related to service types for a member"""
    member_claims = claims_df[claims_df['Member_ID'] == member_id]
    
    # Calculate total claims by service group
    service_group_totals = member_claims.groupby('ServiceGroup')['TotPaymentUSD'].sum()
    
    # Calculate frequency by service group
    service_group_freq = member_claims.groupby('ServiceGroup').size()
    
    return service_group_totals, service_group_freq

def calculate_temporal_features(claims_df, cutoff_date=None):
    """Calculate temporal features for claims"""
    if cutoff_date is None:
        cutoff_date = claims_df['ServiceDate'].max()
    
    # Calculate days between claims
    claims_df = claims_df.sort_values('ServiceDate')
    claims_df['days_since_last_claim'] = claims_df.groupby('Member_ID')['ServiceDate'].diff().dt.days
    
    # Calculate monthly aggregates
    claims_df['month'] = claims_df['ServiceDate'].dt.strftime('%Y-%m')  # Use string format instead of Period
    monthly_claims = claims_df.groupby(['Member_ID', 'month']).agg({
        'TotPaymentUSD': ['count', 'sum']
    }).reset_index()
    
    # Flatten column names
    monthly_claims.columns = ['Member_ID', 'month', 'monthly_claim_count', 'monthly_claim_sum']
    
    # Sort by member and month
    monthly_claims = monthly_claims.sort_values(['Member_ID', 'month'])
    
    # Calculate rolling averages
    monthly_claims['3m_avg_claims'] = monthly_claims.groupby('Member_ID')['monthly_claim_sum'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    monthly_claims['6m_avg_claims'] = monthly_claims.groupby('Member_ID')['monthly_claim_sum'].transform(lambda x: x.rolling(6, min_periods=1).mean())
    
    # Get the most recent values for each member
    latest_features = monthly_claims.groupby('Member_ID').last().reset_index()
    
    # Drop the month column as it's not needed for modeling
    latest_features = latest_features.drop('month', axis=1)
    
    return latest_features

def create_target_variable(claims_df, cutoff_date, prediction_window=180):
    """Create target variable: total claims in next 6 months"""
    end_date = cutoff_date + timedelta(days=prediction_window)
    
    # Filter claims in prediction window
    future_claims = claims_df[
        (claims_df['ServiceDate'] > cutoff_date) & 
        (claims_df['ServiceDate'] <= end_date)
    ]
    
    # Calculate total claims per member in prediction window
    target = future_claims.groupby('Member_ID')['TotPaymentUSD'].sum().reset_index()
    target.columns = ['Member_ID', 'future_6m_claims']
    
    return target

def combine_features(historical_features, temporal_features, member_df, target):
    """Combine all features and target variable"""
    # Merge features
    features = pd.merge(historical_features, temporal_features, on='Member_ID', how='left')
    features = pd.merge(features, member_df, on='Member_ID', how='left')
    
    # Add target variable
    features = pd.merge(features, target, on='Member_ID', how='left')
    
    # Fill missing target values with 0 (assuming no claims)
    features['future_6m_claims'].fillna(0, inplace=True)
    
    # Fill missing temporal features with 0
    temporal_cols = ['monthly_claim_count', 'monthly_claim_sum', '3m_avg_claims', '6m_avg_claims']
    features[temporal_cols] = features[temporal_cols].fillna(0)
    
    return features

def prepare_features_for_modeling(claims_df, members_df, cutoff_date=None):
    """Main function to prepare features for modeling"""
    if cutoff_date is None:
        cutoff_date = claims_df['ServiceDate'].max() - timedelta(days=180)
    
    # Calculate historical features
    historical_features = calculate_historical_features(claims_df, cutoff_date)
    
    # Calculate temporal features
    temporal_features = calculate_temporal_features(claims_df, cutoff_date)
    
    # Create target variable
    target = create_target_variable(claims_df, cutoff_date)
    
    # Combine all features
    features = combine_features(historical_features, temporal_features, members_df, target)
    
    return features