import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import os

def create_date_features(df, date_column):
    """
    Create comprehensive date-based features from a date column
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataframe to process
    date_column : str
        Name of the date column to extract features from
        
    Returns:
    --------
    pandas DataFrame
        Dataframe with additional date-based features
    """
    # Make a copy to avoid modifying the original
    df_processed = df.copy()
    
    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df_processed[date_column]):
        df_processed[date_column] = pd.to_datetime(df_processed[date_column])
    
    # Extract basic date components
    df_processed[f'{date_column}_year'] = df_processed[date_column].dt.year
    df_processed[f'{date_column}_month'] = df_processed[date_column].dt.month
    df_processed[f'{date_column}_day'] = df_processed[date_column].dt.day
    
    # Extract day of week and weekend flag
    df_processed[f'{date_column}_dayofweek'] = df_processed[date_column].dt.dayofweek
    df_processed[f'{date_column}_is_weekend'] = df_processed[f'{date_column}_dayofweek'].isin([5, 6]).astype(int)
    
    # Extract week of year and quarter
    df_processed[f'{date_column}_weekofyear'] = df_processed[date_column].dt.isocalendar().week
    df_processed[f'{date_column}_quarter'] = df_processed[date_column].dt.quarter
    
    # Extract month start/end and quarter start/end
    df_processed[f'{date_column}_is_month_start'] = df_processed[date_column].dt.is_month_start.astype(int)
    df_processed[f'{date_column}_is_month_end'] = df_processed[date_column].dt.is_month_end.astype(int)
    df_processed[f'{date_column}_is_quarter_start'] = df_processed[date_column].dt.is_quarter_start.astype(int)
    df_processed[f'{date_column}_is_quarter_end'] = df_processed[date_column].dt.is_quarter_end.astype(int)
    
    # Create "days since" feature to capture time progression
    reference_date = df_processed[date_column].min()
    df_processed[f'days_since_{date_column}_start'] = (df_processed[date_column] - reference_date).dt.days
    
    print(f"Created {len(df_processed.columns) - len(df.columns)} new date features from {date_column}")
    
    return df_processed

def create_cyclical_features(df, column, period):
    """
    Create cyclical features using sine and cosine transformations
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataframe to process
    column : str
        Name of the column to transform (e.g., 'month', 'day', 'hour')
    period : int
        The period of the feature (e.g., 12 for month, 7 for day of week)
        
    Returns:
    --------
    pandas DataFrame
        Dataframe with additional cyclical features
    """
    # Make a copy to avoid modifying the original
    df_processed = df.copy()
    
    # Create sine and cosine features
    df_processed[f'{column}_sin'] = np.sin(2 * np.pi * df_processed[column] / period)
    df_processed[f'{column}_cos'] = np.cos(2 * np.pi * df_processed[column] / period)
    
    return df_processed

def add_all_cyclical_features(df):
    """
    Add cyclical encoding for all relevant date components
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataframe with date features already extracted
        
    Returns:
    --------
    pandas DataFrame
        Dataframe with cyclical features
    """
    # Make a copy to avoid modifying the original
    df_processed = df.copy()
    
    # Find columns that could benefit from cyclical encoding
    month_cols = [col for col in df_processed.columns if '_month' in col]
    day_cols = [col for col in df_processed.columns if '_day' in col]
    dayofweek_cols = [col for col in df_processed.columns if '_dayofweek' in col]
    quarter_cols = [col for col in df_processed.columns if '_quarter' in col]
    weekofyear_cols = [col for col in df_processed.columns if '_weekofyear' in col]
    
    # Add cyclical encoding for each feature
    for col in month_cols:
        df_processed = create_cyclical_features(df_processed, col, 12)
    
    for col in day_cols:
        df_processed = create_cyclical_features(df_processed, col, 31)  # Maximum days in a month
    
    for col in dayofweek_cols:
        df_processed = create_cyclical_features(df_processed, col, 7)
    
    for col in quarter_cols:
        df_processed = create_cyclical_features(df_processed, col, 4)
    
    for col in weekofyear_cols:
        df_processed = create_cyclical_features(df_processed, col, 52)  # Weeks in a year
    
    print(f"Created {len(df_processed.columns) - len(df.columns)} new cyclical features")
    
    return df_processed

def create_customer_behavior_features(claims_df, member_id_col='Member_ID', date_col='ServiceDate', amount_col='TotPaymentUSD'):
    """
    Create advanced customer behavior features from claims data
    
    Parameters:
    -----------
    claims_df : pandas DataFrame
        The claims dataframe
    member_id_col : str
        Column name containing member IDs
    date_col : str
        Column name containing service dates
    amount_col : str
        Column name containing claim amounts
        
    Returns:
    --------
    pandas DataFrame
        Dataframe with customer behavior features aggregated by member
    """
    # Ensure the date column is datetime
    claims_df = claims_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(claims_df[date_col]):
        claims_df[date_col] = pd.to_datetime(claims_df[date_col])
    
    # Sort by member and date
    claims_df = claims_df.sort_values([member_id_col, date_col])
    
    # Calculate time between claims for each member
    claims_df['days_since_prev_claim'] = claims_df.groupby(member_id_col)[date_col].diff().dt.days
    
    # Calculate rate of change in claim amounts
    claims_df['claim_amount_diff'] = claims_df.groupby(member_id_col)[amount_col].diff()
    claims_df['claim_amount_pct_change'] = claims_df.groupby(member_id_col)[amount_col].pct_change() * 100
    
    # Calculate time windows for aggregation
    max_date = claims_df[date_col].max()
    claims_df['days_to_last_date'] = (max_date - claims_df[date_col]).dt.days
    
    # Calculate claim frequency and recency
    member_stats = claims_df.groupby(member_id_col).agg({
        date_col: [
            'count',  # Total number of claims
            ('max', lambda x: (max_date - x.max()).days),  # Recency (days since last claim)
            ('min', lambda x: (max_date - x.min()).days),  # Days since first claim (tenure)
            ('nunique', lambda x: x.dt.strftime('%Y-%m').nunique())  # Number of unique months with claims
        ],
        'days_since_prev_claim': ['mean', 'median', 'std', 'min', 'max'],  # Claim frequency metrics
        amount_col: ['sum', 'mean', 'median', 'std', 'min', 'max'],  # Claim amount metrics
        'claim_amount_diff': ['mean', 'std'],  # Trend in claim amounts
        'claim_amount_pct_change': ['mean', 'std']  # Volatility in claim amounts
    })
    
    # Flatten column names
    member_stats.columns = [f"{col[0]}_{col[1]}" if col[1] != '' else col[0] for col in member_stats.columns]
    member_stats = member_stats.rename(columns={
        f'{date_col}_count': 'total_claims',
        f'{date_col}_max': 'days_since_last_claim',
        f'{date_col}_min': 'days_since_first_claim',
        f'{date_col}_nunique': 'unique_months_with_claims'
    })
    
    # Calculate additional behavioral metrics
    member_stats['avg_claim_per_month'] = member_stats['total_claims'] / (member_stats['unique_months_with_claims'] + 1)
    member_stats['claim_frequency'] = member_stats['total_claims'] / (member_stats['days_since_first_claim'] + 1) * 30  # Claims per month
    
    # Calculate claim regularity (coefficient of variation of days between claims)
    # Lower values indicate more regular claiming behavior
    member_stats['claim_regularity'] = member_stats['days_since_prev_claim_std'] / (member_stats['days_since_prev_claim_mean'] + 1)
    
    # Calculate claim amount volatility (coefficient of variation of claim amounts)
    member_stats['claim_amount_volatility'] = member_stats[f'{amount_col}_std'] / (member_stats[f'{amount_col}_mean'] + 1)
    
    # Reset index to get member_id as a column
    member_stats = member_stats.reset_index()
    
    # Create visualizations
    if not os.path.exists('visualizations/customer_behavior'):
        os.makedirs('visualizations/customer_behavior', exist_ok=True)
    
    # Plot distribution of claim frequency
    plt.figure(figsize=(10, 6))
    sns.histplot(member_stats['claim_frequency'].dropna(), kde=True)
    plt.title('Distribution of Claim Frequency (Claims per Month)')
    plt.savefig('visualizations/customer_behavior/claim_frequency_distribution.png')
    plt.close()
    
    # Plot distribution of claim regularity
    plt.figure(figsize=(10, 6))
    sns.histplot(member_stats['claim_regularity'].dropna(), kde=True)
    plt.title('Distribution of Claim Regularity (Lower = More Regular)')
    plt.savefig('visualizations/customer_behavior/claim_regularity_distribution.png')
    plt.close()
    
    # Plot distribution of claim amount volatility
    plt.figure(figsize=(10, 6))
    sns.histplot(member_stats['claim_amount_volatility'].dropna(), kde=True)
    plt.title('Distribution of Claim Amount Volatility (Higher = More Volatile)')
    plt.savefig('visualizations/customer_behavior/claim_amount_volatility_distribution.png')
    plt.close()
    
    print(f"Created {len(member_stats.columns) - 1} customer behavior features for {len(member_stats)} members")
    
    return member_stats

def create_service_distribution_features(claims_df, member_id_col='Member_ID', service_type_col='ServiceType', 
                                        service_group_col='ServiceGroup', amount_col='TotPaymentUSD'):
    """
    Create features related to the distribution of services used by each member
    
    Parameters:
    -----------
    claims_df : pandas DataFrame
        The claims dataframe
    member_id_col : str
        Column name containing member IDs
    service_type_col : str
        Column name containing service types
    service_group_col : str
        Column name containing service groups
    amount_col : str
        Column name containing claim amounts
        
    Returns:
    --------
    pandas DataFrame
        Dataframe with service distribution features aggregated by member
    """
    # Group by member and service type
    service_type_stats = claims_df.groupby([member_id_col, service_type_col]).agg({
        amount_col: ['count', 'sum']
    }).reset_index()
    
    # Flatten column names
    service_type_stats.columns = [
        f"{col[0]}_{col[1]}" if col[1] != '' else col[0] 
        for col in service_type_stats.columns
    ]
    
    # Calculate total claims and amounts per member
    member_totals = claims_df.groupby(member_id_col).agg({
        amount_col: ['count', 'sum']
    }).reset_index()
    
    # Flatten column names
    member_totals.columns = [
        f"{col[0]}_total_{col[1]}" if col[1] != '' else col[0] 
        for col in member_totals.columns
    ]
    
    # Merge with service type stats
    service_type_stats = pd.merge(service_type_stats, member_totals, on=member_id_col)
    
    # Calculate percentages
    service_type_stats[f'{service_type_col}_pct_count'] = (
        service_type_stats[f'{amount_col}_count'] / service_type_stats[f'{amount_col}_total_count'] * 100
    )
    service_type_stats[f'{service_type_col}_pct_amount'] = (
        service_type_stats[f'{amount_col}_sum'] / service_type_stats[f'{amount_col}_total_sum'] * 100
    )
    
    # Calculate service diversity metrics
    service_diversity = claims_df.groupby(member_id_col).agg({
        service_type_col: 'nunique',
        service_group_col: 'nunique'
    }).reset_index()
    
    service_diversity.columns = [
        member_id_col, 
        'service_type_diversity', 
        'service_group_diversity'
    ]
    
    # Calculate service concentration (Herfindahl-Hirschman Index)
    def calc_hhi(group):
        """Calculate Herfindahl-Hirschman Index for a group"""
        totals = group[amount_col].sum()
        if totals == 0:
            return 0
        shares = group.groupby(service_type_col)[amount_col].sum() / totals
        return (shares ** 2).sum() * 10000  # Scale to 0-10000
    
    service_concentration = claims_df.groupby(member_id_col).apply(calc_hhi).reset_index()
    service_concentration.columns = [member_id_col, 'service_concentration_hhi']
    
    # Pivot the service type percentages to create features per service type
    service_type_pivot = service_type_stats.pivot_table(
        index=member_id_col, 
        columns=service_type_col, 
        values=[f'{service_type_col}_pct_amount', f'{service_type_col}_pct_count'],
        fill_value=0
    )
    
    # Flatten multi-index columns
    service_type_pivot.columns = [
        f"{col[0]}_{col[1]}" 
        for col in service_type_pivot.columns
    ]
    
    # Reset index
    service_type_pivot = service_type_pivot.reset_index()
    
    # Merge diversity and concentration metrics
    result = pd.merge(service_diversity, service_concentration, on=member_id_col)
    
    # Merge with pivoted service type data
    result = pd.merge(result, service_type_pivot, on=member_id_col)
    
    # Create visualizations
    if not os.path.exists('visualizations/service_distribution'):
        os.makedirs('visualizations/service_distribution', exist_ok=True)
    
    # Plot distribution of service type diversity
    plt.figure(figsize=(10, 6))
    sns.histplot(service_diversity['service_type_diversity'].dropna(), kde=True)
    plt.title('Distribution of Service Type Diversity')
    plt.savefig('visualizations/service_distribution/service_type_diversity_distribution.png')
    plt.close()
    
    # Plot distribution of service concentration
    plt.figure(figsize=(10, 6))
    sns.histplot(service_concentration['service_concentration_hhi'].dropna(), kde=True)
    plt.title('Distribution of Service Concentration (HHI)')
    plt.savefig('visualizations/service_distribution/service_concentration_distribution.png')
    plt.close()
    
    print(f"Created {len(result.columns) - 1} service distribution features for {len(result)} members")
    
    return result

def enhanced_feature_engineering(claims_df, members_df, date_columns=None):
    """
    Complete pipeline for enhanced feature engineering
    
    Parameters:
    -----------
    claims_df : pandas DataFrame
        The claims dataframe
    members_df : pandas DataFrame
        The members dataframe
    date_columns : list, optional
        List of date columns to create features from
        
    Returns:
    --------
    pandas DataFrame
        Dataframe with all enhanced features
    """
    print("Starting enhanced feature engineering pipeline...")
    
    # If no date columns provided, try to identify them
    if date_columns is None:
        date_columns = []
        for col in claims_df.columns:
            if pd.api.types.is_datetime64_any_dtype(claims_df[col]):
                date_columns.append(col)
        
        for col in members_df.columns:
            if pd.api.types.is_datetime64_any_dtype(members_df[col]):
                date_columns.append(col)
    
    # Create date features for claims dataframe
    claims_enhanced = claims_df.copy()
    for date_col in [col for col in date_columns if col in claims_df.columns]:
        claims_enhanced = create_date_features(claims_enhanced, date_col)
    
    # Create date features for members dataframe
    members_enhanced = members_df.copy()
    for date_col in [col for col in date_columns if col in members_df.columns]:
        members_enhanced = create_date_features(members_enhanced, date_col)
    
    # Add cyclical encoding for date features
    claims_enhanced = add_all_cyclical_features(claims_enhanced)
    members_enhanced = add_all_cyclical_features(members_enhanced)
    
    # Create customer behavior features
    customer_behavior = create_customer_behavior_features(
        claims_enhanced, 
        member_id_col='Member_ID', 
        date_col='ServiceDate', 
        amount_col='TotPaymentUSD'
    )
    
    # Create service distribution features
    service_distribution = create_service_distribution_features(
        claims_enhanced,
        member_id_col='Member_ID',
        service_type_col='ServiceType',
        service_group_col='ServiceGroup',
        amount_col='TotPaymentUSD'
    )
    
    # Combine all features at the member level
    # Start with the enhanced member dataframe
    combined_features = members_enhanced
    
    # Add customer behavior features
    combined_features = pd.merge(
        combined_features, 
        customer_behavior, 
        on='Member_ID', 
        how='left'
    )
    
    # Add service distribution features
    combined_features = pd.merge(
        combined_features, 
        service_distribution, 
        on='Member_ID', 
        how='left'
    )
    
    # Fill NaN values for members without claims
    numeric_cols = combined_features.select_dtypes(include=['int', 'float']).columns
    combined_features[numeric_cols] = combined_features[numeric_cols].fillna(0)
    
    print(f"Enhanced feature engineering completed with {combined_features.shape[1]} features for {combined_features.shape[0]} members")
    
    return combined_features

if __name__ == "__main__":
    # Example usage
    print("This is a module for enhanced feature engineering. Import and use in your main script.") 