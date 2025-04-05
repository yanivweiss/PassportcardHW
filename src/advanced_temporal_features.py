import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')

def create_time_window_features(claims_df, cutoff_date=None):
    """
    Create advanced time window features that capture temporal patterns in different windows
    """
    if cutoff_date is None:
        cutoff_date = claims_df['ServiceDate'].max()
    
    # Get all unique member IDs from the original dataframe
    all_member_ids = claims_df['Member_ID'].unique()
    
    # Filter claims before cutoff date
    historical_claims = claims_df[claims_df['ServiceDate'] <= cutoff_date].copy()
    
    # Sort by member and date
    historical_claims = historical_claims.sort_values(['Member_ID', 'ServiceDate'])
    
    # Create multiple time windows for analysis
    windows = {
        '30d': 30,
        '60d': 60,
        '90d': 90,
        '180d': 180,
        '365d': 365
    }
    
    results = []
    
    for member_id in all_member_ids:
        member_claims = historical_claims[historical_claims['Member_ID'] == member_id]
        
        feature_dict = {'Member_ID': member_id}
        
        # For each time window, calculate features
        for window_name, days in windows.items():
            window_start = cutoff_date - timedelta(days=days)
            window_claims = member_claims[member_claims['ServiceDate'] >= window_start]
            
            # Skip if no claims in window
            if len(window_claims) == 0:
                feature_dict[f'claim_count_{window_name}'] = 0
                feature_dict[f'claim_amount_{window_name}'] = 0
                feature_dict[f'claim_frequency_{window_name}'] = 0
                feature_dict[f'avg_claim_{window_name}'] = 0
                feature_dict[f'max_claim_{window_name}'] = 0
                feature_dict[f'service_types_{window_name}'] = 0
                continue
            
            # Basic aggregates
            feature_dict[f'claim_count_{window_name}'] = len(window_claims)
            feature_dict[f'claim_amount_{window_name}'] = window_claims['TotPaymentUSD'].sum()
            feature_dict[f'claim_frequency_{window_name}'] = len(window_claims) / (days / 30)  # monthly rate
            feature_dict[f'avg_claim_{window_name}'] = window_claims['TotPaymentUSD'].mean()
            feature_dict[f'max_claim_{window_name}'] = window_claims['TotPaymentUSD'].max()
            feature_dict[f'service_types_{window_name}'] = window_claims['ServiceType'].nunique()
            
            # Service group distributions
            service_groups = window_claims.groupby('ServiceGroup')['TotPaymentUSD'].sum()
            for group in service_groups.index:
                feature_dict[f'{group}_amount_{window_name}'] = service_groups[group]
        
        # Add rolling trend indicators (acceleration/deceleration)
        if '180d' in windows and '365d' in windows and feature_dict.get('claim_count_365d', 0) > 0:
            # Trend in frequency
            first_half_freq = feature_dict.get('claim_frequency_180d', 0) if feature_dict.get('claim_count_180d', 0) > 0 else 0
            second_half_freq = feature_dict.get('claim_frequency_90d', 0) if feature_dict.get('claim_count_90d', 0) > 0 else 0
            feature_dict['claim_freq_trend'] = second_half_freq - first_half_freq
            
            # Trend in average claim amount
            first_half_avg = feature_dict.get('avg_claim_180d', 0) if feature_dict.get('claim_count_180d', 0) > 0 else 0
            second_half_avg = feature_dict.get('avg_claim_90d', 0) if feature_dict.get('claim_count_90d', 0) > 0 else 0
            feature_dict['avg_claim_trend'] = second_half_avg - first_half_avg
        else:
            feature_dict['claim_freq_trend'] = 0
            feature_dict['avg_claim_trend'] = 0
        
        results.append(feature_dict)
    
    # Convert to DataFrame
    window_features = pd.DataFrame(results)
    
    return window_features

def create_seasonality_features(claims_df, cutoff_date=None):
    """
    Extract seasonality components from the time series data
    """
    if cutoff_date is None:
        cutoff_date = claims_df['ServiceDate'].max()
    
    # Get all unique member IDs from the original dataframe
    all_member_ids = claims_df['Member_ID'].unique()
    
    # Filter claims before cutoff date
    historical_claims = claims_df[claims_df['ServiceDate'] <= cutoff_date].copy()
    
    # Create monthly time series for each member
    historical_claims['year_month'] = historical_claims['ServiceDate'].dt.to_period('M')
    
    # Aggregate claims by member and month
    monthly_claims = historical_claims.groupby(['Member_ID', 'year_month'])['TotPaymentUSD'].agg(['sum', 'count']).reset_index()
    monthly_claims['year_month'] = monthly_claims['year_month'].astype(str)
    
    # Convert to proper datetime for time series analysis
    monthly_claims['date'] = pd.to_datetime(monthly_claims['year_month'])
    
    seasonality_features = []
    
    # Process each member ID
    for member_id in all_member_ids:
        member_monthly = monthly_claims[monthly_claims['Member_ID'] == member_id].sort_values('date')
        
        # Need at least 12 months for meaningful seasonality analysis
        if len(member_monthly) >= 12:
            # For claim amounts
            try:
                # Create regular time series (fill missing months with 0)
                date_range = pd.date_range(member_monthly['date'].min(), member_monthly['date'].max(), freq='MS')
                ts = pd.Series(index=date_range, data=np.nan)
                
                for _, row in member_monthly.iterrows():
                    ts[row['date']] = row['sum']
                
                ts = ts.fillna(0)
                
                # Decompose time series
                result = seasonal_decompose(ts, model='additive', period=12)
                
                # Get seasonality strength
                seasonality_strength = np.std(result.seasonal) / (np.std(result.trend) + np.std(result.seasonal) + 1e-10)
                
                # Get month with highest seasonal component
                seasonal_by_month = {}
                for i, val in enumerate(result.seasonal):
                    month = ts.index[i].month
                    if month not in seasonal_by_month:
                        seasonal_by_month[month] = []
                    seasonal_by_month[month].append(val)
                
                avg_seasonal_by_month = {m: np.mean(vals) for m, vals in seasonal_by_month.items()}
                peak_month = max(avg_seasonal_by_month, key=avg_seasonal_by_month.get)
                peak_value = avg_seasonal_by_month[peak_month]
                
                # Create feature dictionary
                feature_dict = {
                    'Member_ID': member_id,
                    'seasonality_strength': seasonality_strength,
                    'peak_month': peak_month,
                    'peak_value': peak_value,
                    'has_seasonality': 1 if seasonality_strength > 0.3 else 0  # Threshold for meaningful seasonality
                }
                
                # Add seasonal index for each month
                for month, value in avg_seasonal_by_month.items():
                    feature_dict[f'seasonal_idx_month_{month}'] = value
                
                seasonality_features.append(feature_dict)
                
            except Exception as e:
                # If decomposition fails, add basic feature dictionary
                seasonality_features.append({
                    'Member_ID': member_id,
                    'seasonality_strength': 0,
                    'peak_month': 0,
                    'peak_value': 0,
                    'has_seasonality': 0
                })
        else:
            # Not enough data for seasonality analysis
            seasonality_features.append({
                'Member_ID': member_id,
                'seasonality_strength': 0,
                'peak_month': 0,
                'peak_value': 0,
                'has_seasonality': 0
            })
    
    # Convert to DataFrame
    seasonality_df = pd.DataFrame(seasonality_features)
    
    return seasonality_df

def create_volatility_features(claims_df, cutoff_date=None):
    """
    Create features that capture the volatility of claims over time
    """
    if cutoff_date is None:
        cutoff_date = claims_df['ServiceDate'].max()
    
    # Get all unique member IDs from the original dataframe
    all_member_ids = claims_df['Member_ID'].unique()
    
    # Filter claims before cutoff date
    historical_claims = claims_df[claims_df['ServiceDate'] <= cutoff_date].copy()
    
    # Create monthly time series for each member
    historical_claims['year_month'] = historical_claims['ServiceDate'].dt.to_period('M')
    
    # Aggregate claims by member and month
    monthly_claims = historical_claims.groupby(['Member_ID', 'year_month'])['TotPaymentUSD'].agg(['sum', 'count']).reset_index()
    
    volatility_features = []
    
    # Process each member ID
    for member_id in all_member_ids:
        member_monthly = monthly_claims[monthly_claims['Member_ID'] == member_id]
        
        # Calculate volatility metrics
        if len(member_monthly) >= 3:  # Need at least 3 months for meaningful volatility
            monthly_amounts = member_monthly['sum'].values
            monthly_counts = member_monthly['count'].values
            
            # Coefficient of variation (higher = more volatile)
            cv_amount = np.std(monthly_amounts) / (np.mean(monthly_amounts) + 1e-10)
            cv_count = np.std(monthly_counts) / (np.mean(monthly_counts) + 1e-10)
            
            # Calculate month-to-month changes
            amount_changes = np.diff(monthly_amounts)
            count_changes = np.diff(monthly_counts)
            
            # Average absolute change
            avg_abs_change_amount = np.mean(np.abs(amount_changes))
            avg_abs_change_count = np.mean(np.abs(count_changes))
            
            # Maximum spike (maximum month-to-month increase)
            max_spike_amount = np.max(amount_changes) if len(amount_changes) > 0 else 0
            max_spike_count = np.max(count_changes) if len(count_changes) > 0 else 0
            
            # Direction changes (how often the trend changes direction)
            direction_changes_amount = sum(1 for i in range(1, len(amount_changes)) if amount_changes[i] * amount_changes[i-1] < 0)
            direction_changes_count = sum(1 for i in range(1, len(count_changes)) if count_changes[i] * count_changes[i-1] < 0)
            
            # Normalize by number of months
            direction_change_rate_amount = direction_changes_amount / (len(amount_changes) - 1) if len(amount_changes) > 1 else 0
            direction_change_rate_count = direction_changes_count / (len(count_changes) - 1) if len(count_changes) > 1 else 0
            
            feature_dict = {
                'Member_ID': member_id,
                'cv_amount': cv_amount,
                'cv_count': cv_count,
                'avg_abs_change_amount': avg_abs_change_amount,
                'avg_abs_change_count': avg_abs_change_count,
                'max_spike_amount': max_spike_amount,
                'max_spike_count': max_spike_count,
                'direction_change_rate_amount': direction_change_rate_amount,
                'direction_change_rate_count': direction_change_rate_count,
                'volatility_score': (cv_amount + cv_count + direction_change_rate_amount + direction_change_rate_count) / 4
            }
        else:
            # Not enough data for volatility analysis
            feature_dict = {
                'Member_ID': member_id,
                'cv_amount': 0,
                'cv_count': 0,
                'avg_abs_change_amount': 0,
                'avg_abs_change_count': 0,
                'max_spike_amount': 0,
                'max_spike_count': 0,
                'direction_change_rate_amount': 0,
                'direction_change_rate_count': 0,
                'volatility_score': 0
            }
        
        volatility_features.append(feature_dict)
    
    # Convert to DataFrame
    volatility_df = pd.DataFrame(volatility_features)
    
    return volatility_df

def create_advanced_temporal_features(claims_df, cutoff_date=None):
    """
    Create all advanced temporal features
    """
    if cutoff_date is None:
        cutoff_date = claims_df['ServiceDate'].max()
    
    # Generate different types of temporal features
    window_features = create_time_window_features(claims_df, cutoff_date)
    seasonality_features = create_seasonality_features(claims_df, cutoff_date)
    volatility_features = create_volatility_features(claims_df, cutoff_date)
    
    # Merge all features
    temporal_features = pd.merge(window_features, seasonality_features, on='Member_ID', how='outer')
    temporal_features = pd.merge(temporal_features, volatility_features, on='Member_ID', how='outer')
    
    # Fill missing values
    temporal_features = temporal_features.fillna(0)
    
    return temporal_features 