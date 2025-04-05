import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler

def extract_member_features(members_df):
    """
    Extract features from member data only, avoiding using claims data for features
    
    Parameters:
    -----------
    members_df : pandas DataFrame
        Member profile data
        
    Returns:
    --------
    pandas DataFrame
        DataFrame containing member-based features only
    """
    # Make a copy to avoid modifying the original
    members = members_df.copy()
    
    # Ensure date columns are datetime
    date_cols = ['PolicyStartDate', 'PolicyEndDate', 'DateOfBirth']
    for col in date_cols:
        if col in members.columns and not pd.api.types.is_datetime64_any_dtype(members[col]):
            members[col] = pd.to_datetime(members[col], errors='coerce')
    
    # 1. Demographic Features
    
    # Age feature
    if 'DateOfBirth' in members.columns:
        members['Age'] = (pd.Timestamp.now() - members['DateOfBirth']).dt.days / 365.25
    
    # Age groups (categorical)
    if 'Age' in members.columns:
        members['AgeGroup'] = pd.cut(
            members['Age'], 
            bins=[0, 18, 35, 50, 65, 100], 
            labels=['0-18', '19-35', '36-50', '51-65', '65+']
        )
        
        # One-hot encode age groups
        age_dummies = pd.get_dummies(members['AgeGroup'], prefix='age_group')
        members = pd.concat([members, age_dummies], axis=1)
    
    # 2. Policy Features
    
    # Policy duration in years
    if 'PolicyStartDate' in members.columns and 'PolicyEndDate' in members.columns:
        members['PolicyDuration'] = (members['PolicyEndDate'] - members['PolicyStartDate']).dt.days / 365.25
        
        # Create policy duration groups
        members['PolicyDurationGroup'] = pd.cut(
            members['PolicyDuration'], 
            bins=[-1, 1, 3, 5, 100], 
            labels=['<1yr', '1-3yr', '3-5yr', '5yr+']
        )
        
        # One-hot encode policy duration groups
        duration_dummies = pd.get_dummies(members['PolicyDurationGroup'], prefix='policy_duration')
        members = pd.concat([members, duration_dummies], axis=1)
    
    # 3. Health Indicators
    
    # BMI features
    if 'BMI' in members.columns:
        members['BMI_category'] = pd.cut(
            members['BMI'], 
            bins=[0, 18.5, 25, 30, 35, 100], 
            labels=['Underweight', 'Normal', 'Overweight', 'Obese', 'Severely_Obese']
        )
        
        # One-hot encode BMI categories
        bmi_dummies = pd.get_dummies(members['BMI_category'], prefix='bmi')
        members = pd.concat([members, bmi_dummies], axis=1)
    
    # 4. Questionnaire Features
    
    # Get all questionnaire columns
    questionnaire_cols = [col for col in members.columns if col.startswith('Questionnaire_')]
    
    if questionnaire_cols:
        # Create domain-specific health scores
        
        # Chronic condition score
        chronic_cols = [col for col in questionnaire_cols if any(cond in col.lower() for cond in 
                        ['diabetes', 'heart', 'respiratory', 'thyroid', 'liver', 'immune', 'highblood'])]
        if chronic_cols:
            members['chronic_condition_score'] = members[chronic_cols].sum(axis=1)
        
        # Cancer risk score
        cancer_cols = [col for col in questionnaire_cols if any(cond in col.lower() for cond in 
                        ['cancer', 'tumor', 'relatives'])]
        if cancer_cols:
            members['cancer_risk_score'] = members[cancer_cols].sum(axis=1)
        
        # Lifestyle risk score
        lifestyle_cols = [col for col in questionnaire_cols if any(cond in col.lower() for cond in 
                        ['smoke', 'alcoholism', 'drink', 'exercise', 'diet'])]
        if lifestyle_cols:
            members['lifestyle_risk_score'] = members[lifestyle_cols].sum(axis=1)
        
        # Overall health risk score (weighted combination)
        health_scores = []
        if 'chronic_condition_score' in members.columns:
            health_scores.append(members['chronic_condition_score'] * 0.4)
        if 'cancer_risk_score' in members.columns:
            health_scores.append(members['cancer_risk_score'] * 0.4)
        if 'lifestyle_risk_score' in members.columns:
            health_scores.append(members['lifestyle_risk_score'] * 0.2)
        
        if health_scores:
            members['health_risk_score'] = sum(health_scores)
        
        # Sum of all questionnaire responses
        members['total_conditions'] = members[questionnaire_cols].sum(axis=1)
    
    # 5. Geographic Features
    
    # Country of origin features
    if 'CountryOfOrigin' in members.columns:
        # Get top countries and group others
        top_countries = members['CountryOfOrigin'].value_counts().nlargest(5).index.tolist()
        members['CountryGrouped'] = members['CountryOfOrigin'].apply(
            lambda x: x if x in top_countries else 'Other'
        )
        
        # One-hot encode countries
        country_dummies = pd.get_dummies(members['CountryGrouped'], prefix='country')
        members = pd.concat([members, country_dummies], axis=1)
    
    # 6. Gender Features
    if 'Gender' in members.columns:
        # One-hot encode gender
        gender_dummies = pd.get_dummies(members['Gender'], prefix='gender')
        members = pd.concat([members, gender_dummies], axis=1)
    
    # 7. Calculate interaction terms
    
    # Age and BMI interaction
    if 'Age' in members.columns and 'BMI' in members.columns:
        members['Age_BMI_interaction'] = members['Age'] * members['BMI']
    
    # Age and chronic conditions interaction
    if 'Age' in members.columns and 'chronic_condition_score' in members.columns:
        members['Age_chronic_interaction'] = members['Age'] * members['chronic_condition_score']
    
    # Filter out non-feature columns
    feature_cols = [
        'Member_ID',  # Keep ID for merging
        'Age', 'PolicyDuration', 'BMI',
        'chronic_condition_score', 'cancer_risk_score', 'lifestyle_risk_score',
        'health_risk_score', 'total_conditions',
        'Age_BMI_interaction', 'Age_chronic_interaction'
    ]
    
    # Add categorical dummies
    feature_cols.extend([col for col in members.columns if 'age_group_' in col])
    feature_cols.extend([col for col in members.columns if 'policy_duration_' in col])
    feature_cols.extend([col for col in members.columns if 'bmi_' in col])
    feature_cols.extend([col for col in members.columns if 'country_' in col])
    feature_cols.extend([col for col in members.columns if 'gender_' in col])
    
    # Only keep columns that exist
    existing_cols = [col for col in feature_cols if col in members.columns]
    
    return members[existing_cols]

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

def prepare_member_features(members_df, claims_df=None, cutoff_date=None):
    """
    Main function to prepare member-based features for modeling
    
    Parameters:
    -----------
    members_df : pandas DataFrame
        Member profile data
    claims_df : pandas DataFrame, optional
        Claims data (only used for target variable)
    cutoff_date : datetime, optional
        Cutoff date for creating the target variable
    
    Returns:
    --------
    pandas DataFrame
        DataFrame with member features and target variable (if claims_df provided)
    """
    # Extract member features
    features = extract_member_features(members_df)
    
    # If claims data is provided, create and add target variable
    if claims_df is not None:
        if cutoff_date is None:
            cutoff_date = claims_df['ServiceDate'].max() - timedelta(days=180)
        
        # Create target variable from claims after cutoff
        target = create_target_variable(claims_df, cutoff_date)
        
        # Merge features with target
        features = pd.merge(features, target, on='Member_ID', how='left')
        
        # Fill missing target values with 0 (assuming no claims)
        features['future_6m_claims'].fillna(0, inplace=True)
    
    return features 