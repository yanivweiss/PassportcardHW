import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew

def create_advanced_temporal_features(claims_df, cutoff_date=None):
    """Create advanced temporal features from claims data"""
    if cutoff_date is None:
        cutoff_date = claims_df['ServiceDate'].max()
    
    # Filter claims before cutoff date
    historical_claims = claims_df[claims_df['ServiceDate'] <= cutoff_date]
    
    # Sort by member and date
    historical_claims = historical_claims.sort_values(['Member_ID', 'ServiceDate'])
    
    # Calculate days since last claim
    historical_claims['days_since_last_claim'] = historical_claims.groupby('Member_ID')['ServiceDate'].diff().dt.days
    
    # Calculate time-based aggregates (monthly, quarterly, yearly)
    historical_claims['month'] = historical_claims['ServiceDate'].dt.to_period('M').astype(str)
    historical_claims['quarter'] = historical_claims['ServiceDate'].dt.to_period('Q').astype(str)
    historical_claims['year'] = historical_claims['ServiceDate'].dt.year
    
    # Add recency features
    historical_claims['recency'] = (cutoff_date - historical_claims['ServiceDate']).dt.days
    
    # Calculate seasonality features
    historical_claims['month_of_year'] = historical_claims['ServiceDate'].dt.month
    historical_claims['day_of_week'] = historical_claims['ServiceDate'].dt.dayofweek
    historical_claims['is_weekend'] = historical_claims['day_of_week'].isin([5, 6]).astype(int)
    
    # Calculate time-based statistics
    time_stats = historical_claims.groupby('Member_ID').agg({
        # Basic count and sum
        'TotPaymentUSD': ['count', 'sum', 'mean', 'std', 'max', 'min'],
        
        # Recency metrics
        'recency': ['min', 'mean'],
        
        # Time between claims
        'days_since_last_claim': ['mean', 'std', 'max'],
        
        # First and last claim dates
        'ServiceDate': ['min', 'max']
    })
    
    # Flatten column names
    time_stats.columns = ['_'.join(col).strip() for col in time_stats.columns.values]
    time_stats = time_stats.reset_index()
    
    # Calculate duration of history
    time_stats['history_days'] = (time_stats['ServiceDate_max'] - time_stats['ServiceDate_min']).dt.days
    time_stats['history_years'] = time_stats['history_days'] / 365.25
    
    # Add frequency features
    time_stats['claim_frequency_monthly'] = time_stats['TotPaymentUSD_count'] / (time_stats['history_days'] / 30)
    time_stats['claim_frequency_yearly'] = time_stats['TotPaymentUSD_count'] / time_stats['history_years']
    
    # Calculate acceleration (change in frequency)
    # Split history into two halves and compare claim frequencies
    member_features = []
    
    for member_id in time_stats['Member_ID'].unique():
        member_claims = historical_claims[historical_claims['Member_ID'] == member_id]
        
        if len(member_claims) > 1:
            # Find midpoint date
            start_date = member_claims['ServiceDate'].min()
            end_date = member_claims['ServiceDate'].max()
            mid_date = start_date + (end_date - start_date) / 2
            
            # Split claims into two periods
            first_half = member_claims[member_claims['ServiceDate'] <= mid_date]
            second_half = member_claims[member_claims['ServiceDate'] > mid_date]
            
            # Calculate frequencies in each half
            first_half_days = (mid_date - start_date).days or 1  # Avoid division by zero
            second_half_days = (end_date - mid_date).days or 1   # Avoid division by zero
            
            first_half_freq = len(first_half) / (first_half_days / 30)  # Monthly
            second_half_freq = len(second_half) / (second_half_days / 30)  # Monthly
            
            # Calculate acceleration (change in frequency)
            claim_acceleration = second_half_freq - first_half_freq
            
            # Calculate average claim amount trends
            first_half_avg = first_half['TotPaymentUSD'].mean() if len(first_half) > 0 else 0
            second_half_avg = second_half['TotPaymentUSD'].mean() if len(second_half) > 0 else 0
            avg_claim_trend = second_half_avg - first_half_avg
            
            # Create quarterly and yearly aggregates
            quarterly_claims = member_claims.groupby('quarter')['TotPaymentUSD'].agg(['count', 'sum', 'mean'])
            yearly_claims = member_claims.groupby('year')['TotPaymentUSD'].agg(['count', 'sum', 'mean'])
            
            # Calculate quarter-over-quarter and year-over-year growth
            if len(quarterly_claims) > 1:
                qoq_growth = quarterly_claims['sum'].pct_change().mean()
                qoq_count_growth = quarterly_claims['count'].pct_change().mean()
            else:
                qoq_growth = 0
                qoq_count_growth = 0
                
            if len(yearly_claims) > 1:
                yoy_growth = yearly_claims['sum'].pct_change().mean()
                yoy_count_growth = yearly_claims['count'].pct_change().mean()
            else:
                yoy_growth = 0
                yoy_count_growth = 0
        else:
            # Default values for members with only one claim
            claim_acceleration = 0
            avg_claim_trend = 0
            qoq_growth = 0
            qoq_count_growth = 0
            yoy_growth = 0
            yoy_count_growth = 0
        
        # Get recent claims (last 90 days)
        recent_claims = member_claims[member_claims['recency'] <= 90]
        recent_claim_count = len(recent_claims)
        recent_claim_amount = recent_claims['TotPaymentUSD'].sum() if len(recent_claims) > 0 else 0
        
        # Get seasonal patterns
        monthly_seasonality = member_claims.groupby('month_of_year')['TotPaymentUSD'].agg(['count', 'sum', 'mean'])
        if len(monthly_seasonality) > 0:
            high_season_month = monthly_seasonality['sum'].idxmax()
            high_season_amount = monthly_seasonality['sum'].max()
            seasonal_variation = monthly_seasonality['sum'].std() / monthly_seasonality['sum'].mean() if monthly_seasonality['sum'].mean() > 0 else 0
        else:
            high_season_month = 0
            high_season_amount = 0
            seasonal_variation = 0
        
        # Create feature dictionary
        member_feature = {
            'Member_ID': member_id,
            'claim_acceleration': claim_acceleration,
            'avg_claim_trend': avg_claim_trend,
            'qoq_growth': qoq_growth,
            'qoq_count_growth': qoq_count_growth,
            'yoy_growth': yoy_growth,
            'yoy_count_growth': yoy_count_growth,
            'recent_claim_count': recent_claim_count,
            'recent_claim_amount': recent_claim_amount,
            'high_season_month': high_season_month,
            'high_season_amount': high_season_amount,
            'seasonal_variation': seasonal_variation
        }
        
        member_features.append(member_feature)
    
    # Convert to DataFrame
    advanced_temporal_features = pd.DataFrame(member_features)
    
    # Merge with time stats
    result = pd.merge(time_stats, advanced_temporal_features, on='Member_ID', how='left')
    
    # Fill NaN values with 0
    result = result.fillna(0)
    
    return result

def create_service_type_profiles(claims_df, cutoff_date=None):
    """Create feature profiles based on service types"""
    if cutoff_date is None:
        cutoff_date = claims_df['ServiceDate'].max()
    
    # Filter claims before cutoff date
    historical_claims = claims_df[claims_df['ServiceDate'] <= cutoff_date]
    
    # Pivot data to get service type profiles
    service_pivot = pd.pivot_table(
        historical_claims, 
        values='TotPaymentUSD', 
        index='Member_ID',
        columns='ServiceGroup', 
        aggfunc=['sum', 'count', 'mean'],
        fill_value=0
    )
    
    # Flatten column names
    service_pivot.columns = [f"{col[0]}_{col[1]}".replace(' ', '_').lower() for col in service_pivot.columns]
    service_pivot = service_pivot.reset_index()
    
    # Calculate service type diversity (entropy)
    member_service_entropy = []
    
    for member_id in service_pivot['Member_ID'].unique():
        member_claims = historical_claims[historical_claims['Member_ID'] == member_id]
        service_counts = member_claims['ServiceGroup'].value_counts(normalize=True)
        
        # Calculate entropy if there are claims
        if len(service_counts) > 0:
            entropy = -sum(p * np.log(p) for p in service_counts if p > 0)
        else:
            entropy = 0
            
        member_service_entropy.append({
            'Member_ID': member_id,
            'service_entropy': entropy
        })
    
    # Convert to DataFrame
    service_entropy = pd.DataFrame(member_service_entropy)
    
    # Merge with service profiles
    result = pd.merge(service_pivot, service_entropy, on='Member_ID', how='left')
    
    return result

def create_risk_scores(members_df):
    """Create sophisticated risk scores based on questionnaire responses and demographics"""
    # Create a copy to avoid modifying the original
    members = members_df.copy()
    
    # Get all questionnaire columns
    questionnaire_cols = [col for col in members.columns if col.startswith('Questionnaire_')]
    
    # Calculate basic risk score (sum of all questionnaire responses)
    members['basic_risk_score'] = members[questionnaire_cols].sum(axis=1)
    
    # Create weighted risk scores for different categories
    
    # 1. Chronic condition risk
    chronic_cols = [
        'Questionnaire_diabetes', 'Questionnaire_heart', 'Questionnaire_respiratory',
        'Questionnaire_thyroid', 'Questionnaire_liver', 'Questionnaire_immune'
    ]
    members['chronic_risk_score'] = members[chronic_cols].sum(axis=1) * 2  # Higher weight
    
    # 2. Cancer and serious illness risk
    cancer_cols = [
        'Questionnaire_cancer', 'Questionnaire_tumor', 'Questionnaire_relatives'
    ]
    members['cancer_risk_score'] = members[cancer_cols].sum(axis=1) * 3  # Higher weight
    
    # 3. Lifestyle risk
    lifestyle_cols = [
        'Questionnaire_smoke', 'Questionnaire_alcoholism', 'Questionnaire_drink'
    ]
    members['lifestyle_risk_score'] = members[lifestyle_cols].sum(axis=1) * 1.5
    
    # 4. Age-related risk (using Age column created in data preparation)
    members['age_risk'] = 0
    if 'Age' in members.columns:
        members.loc[members['Age'] >= 65, 'age_risk'] = 3
        members.loc[(members['Age'] >= 50) & (members['Age'] < 65), 'age_risk'] = 2
        members.loc[(members['Age'] >= 35) & (members['Age'] < 50), 'age_risk'] = 1
    
    # 5. BMI-related risk
    members['bmi_risk'] = 0
    if 'BMI' in members.columns:
        members.loc[members['BMI'] >= 30, 'bmi_risk'] = 2  # Obesity
        members.loc[members['BMI'] < 18.5, 'bmi_risk'] = 1  # Underweight
    
    # Combined weighted risk score
    members['weighted_risk_score'] = (
        members['basic_risk_score'] + 
        members['chronic_risk_score'] + 
        members['cancer_risk_score'] + 
        members['lifestyle_risk_score'] +
        members['age_risk'] * 2 +
        members['bmi_risk'] * 1.5
    )
    
    # If we have enough data, try a PCA-based risk score
    if len(members) > 50:  # Reasonable sample size for PCA
        try:
            # Standardize questionnaire data
            scaler = StandardScaler()
            questionnaire_data = scaler.fit_transform(members[questionnaire_cols])
            
            # Apply PCA to create risk components
            pca = PCA(n_components=3)  # Extract 3 main risk components
            risk_components = pca.fit_transform(questionnaire_data)
            
            # Add PCA components as features
            members['risk_component_1'] = risk_components[:, 0]
            members['risk_component_2'] = risk_components[:, 1]
            members['risk_component_3'] = risk_components[:, 2]
            
            # Create PCA-based risk score (using absolute values since components can be negative)
            members['pca_risk_score'] = np.abs(risk_components[:, 0]) + np.abs(risk_components[:, 1]) + np.abs(risk_components[:, 2])
        except:
            # If PCA fails, fallback to weighted score
            members['pca_risk_score'] = members['weighted_risk_score']
    else:
        members['pca_risk_score'] = members['weighted_risk_score']
    
    # Select only risk-related columns
    risk_cols = [
        'Member_ID', 'basic_risk_score', 'chronic_risk_score', 'cancer_risk_score',
        'lifestyle_risk_score', 'age_risk', 'bmi_risk', 'weighted_risk_score',
        'pca_risk_score'
    ]
    
    return members[risk_cols]

def create_interaction_features(features_df):
    """Create interaction terms between key features"""
    # Create a copy to avoid modifying the original
    df = features_df.copy()
    
    # Identify key numerical features for interactions
    key_features = [
        'claim_count', 'total_claims', 'avg_claim', 
        'history_years', 'monthly_claim_count', 'monthly_claim_sum',
        '3m_avg_claims', '6m_avg_claims'
    ]
    
    risk_features = [
        'weighted_risk_score', 'basic_risk_score', 'chronic_risk_score',
        'cancer_risk_score', 'lifestyle_risk_score'
    ]
    
    # Only use features that exist in the dataframe
    key_features = [f for f in key_features if f in df.columns]
    risk_features = [f for f in risk_features if f in df.columns]
    
    # Create risk-related interactions
    for risk_feature in risk_features:
        for key_feature in key_features:
            interaction_name = f"{risk_feature}_x_{key_feature}"
            df[interaction_name] = df[risk_feature] * df[key_feature]
    
    # Create ratio features
    if 'total_claims' in df.columns and 'claim_count' in df.columns and df['claim_count'].max() > 0:
        df['avg_claim_amount'] = df['total_claims'] / df['claim_count'].replace(0, 1)
    
    if 'history_years' in df.columns and 'claim_count' in df.columns:
        df['claims_per_year'] = df['claim_count'] / df['history_years'].replace(0, 1)
    
    if 'history_years' in df.columns and 'total_claims' in df.columns:
        df['spend_per_year'] = df['total_claims'] / df['history_years'].replace(0, 1)
    
    # Add non-linearity: square and sqrt of important features
    for feature in key_features:
        if feature in df.columns:
            df[f"{feature}_squared"] = df[feature] ** 2
            df[f"{feature}_sqrt"] = np.sqrt(np.abs(df[feature]))
    
    # Calculate skewness of numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Fix highly skewed features
    for col in numeric_cols:
        # Skip columns that would cause issues
        if col.endswith('_sqrt') or col.endswith('_squared') or df[col].min() < 0:
            continue
            
        # Calculate skewness
        col_skew = skew(df[col].dropna())
        
        # Apply log transformation to highly skewed features
        if col_skew > 1.0 and df[col].min() >= 0:
            df[f"{col}_log"] = np.log1p(df[col])
    
    return df

def enhance_features(features_df, claims_df, members_df, cutoff_date=None):
    """Main function to enhance features with advanced techniques"""
    if cutoff_date is None:
        cutoff_date = claims_df['ServiceDate'].max() - timedelta(days=180)
    
    # Create advanced temporal features
    print("Creating advanced temporal features...")
    temporal_features = create_advanced_temporal_features(claims_df, cutoff_date)
    
    # Create service type profiles
    print("Creating service type profiles...")
    service_profiles = create_service_type_profiles(claims_df, cutoff_date)
    
    # Create risk scores
    print("Creating enhanced risk scores...")
    risk_scores = create_risk_scores(members_df)
    
    # Merge all feature sets
    print("Merging feature sets...")
    enhanced_features = pd.merge(features_df, temporal_features, on='Member_ID', how='left')
    enhanced_features = pd.merge(enhanced_features, service_profiles, on='Member_ID', how='left')
    enhanced_features = pd.merge(enhanced_features, risk_scores, on='Member_ID', how='left')
    
    # Create interaction features
    print("Creating interaction features...")
    final_features = create_interaction_features(enhanced_features)
    
    # Fill NaN values
    final_features = final_features.fillna(0)
    
    return final_features 