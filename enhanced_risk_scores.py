import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def create_enhanced_risk_scores(members_df, claims_df=None):
    """
    Create enhanced risk scores based on questionnaire responses, demographics,
    and claim patterns (if available)
    """
    # Create a copy to avoid modifying the original
    members = members_df.copy()
    
    # Get all questionnaire columns
    questionnaire_cols = [col for col in members.columns if col.startswith('Questionnaire_')]
    
    if len(questionnaire_cols) == 0:
        raise ValueError("No questionnaire columns found. Expected columns starting with 'Questionnaire_'")
    
    # 1. Create domain-specific risk scores with medical weights
    
    # Chronic condition risk - higher weights for diabetes and heart conditions
    chronic_cols = {
        'Questionnaire_diabetes': 3.0,  # Higher weight
        'Questionnaire_heart': 2.5,     # Higher weight
        'Questionnaire_respiratory': 2.0,
        'Questionnaire_thyroid': 1.5,
        'Questionnaire_liver': 2.0,
        'Questionnaire_immune': 2.0
    }
    
    # Only include columns that exist in the dataframe
    chronic_cols = {k: v for k, v in chronic_cols.items() if k in members.columns}
    
    if chronic_cols:
        members['chronic_risk_score'] = sum(members[col] * weight for col, weight in chronic_cols.items())
    else:
        members['chronic_risk_score'] = 0
    
    # Cancer and serious illness risk - very high weights
    cancer_cols = {
        'Questionnaire_cancer': 4.0,
        'Questionnaire_tumor': 3.0,
        'Questionnaire_relatives': 1.5  # Family history
    }
    
    cancer_cols = {k: v for k, v in cancer_cols.items() if k in members.columns}
    
    if cancer_cols:
        members['cancer_risk_score'] = sum(members[col] * weight for col, weight in cancer_cols.items())
    else:
        members['cancer_risk_score'] = 0
    
    # Lifestyle risk
    lifestyle_cols = {
        'Questionnaire_smoke': 2.0,
        'Questionnaire_alcoholism': 2.5,
        'Questionnaire_drink': 1.0  # Occasional alcohol is less risk
    }
    
    lifestyle_cols = {k: v for k, v in lifestyle_cols.items() if k in members.columns}
    
    if lifestyle_cols:
        members['lifestyle_risk_score'] = sum(members[col] * weight for col, weight in lifestyle_cols.items())
    else:
        members['lifestyle_risk_score'] = 0
    
    # 2. Age-related risk (using birthdate)
    if 'DateOfBirth' in members.columns:
        # Calculate age
        current_date = pd.Timestamp.now()
        members['Age'] = (current_date - pd.to_datetime(members['DateOfBirth'])).dt.days / 365.25
        
        # Assign age risk based on age brackets
        members['age_risk'] = 0
        members.loc[members['Age'] >= 75, 'age_risk'] = 4  # Very high risk for elderly
        members.loc[(members['Age'] >= 65) & (members['Age'] < 75), 'age_risk'] = 3
        members.loc[(members['Age'] >= 50) & (members['Age'] < 65), 'age_risk'] = 2
        members.loc[(members['Age'] >= 35) & (members['Age'] < 50), 'age_risk'] = 1
    elif 'Age' in members.columns:
        # Age is already calculated
        members['age_risk'] = 0
        members.loc[members['Age'] >= 75, 'age_risk'] = 4
        members.loc[(members['Age'] >= 65) & (members['Age'] < 75), 'age_risk'] = 3
        members.loc[(members['Age'] >= 50) & (members['Age'] < 65), 'age_risk'] = 2
        members.loc[(members['Age'] >= 35) & (members['Age'] < 50), 'age_risk'] = 1
    else:
        members['age_risk'] = 0
    
    # 3. BMI-related risk
    if 'BMI' in members.columns:
        members['bmi_risk'] = 0
        members.loc[members['BMI'] >= 35, 'bmi_risk'] = 3  # Severe obesity
        members.loc[(members['BMI'] >= 30) & (members['BMI'] < 35), 'bmi_risk'] = 2  # Obesity
        members.loc[(members['BMI'] < 18.5), 'bmi_risk'] = 1  # Underweight
    else:
        members['bmi_risk'] = 0
    
    # 4. Gender-based risk adjustment
    if 'Gender' in members.columns:
        # Adjust for gender-specific risks (simplified)
        # In a real model, this would be based on medical statistics
        gender_risk = np.zeros(len(members))
        members['gender_risk'] = gender_risk
    else:
        members['gender_risk'] = 0
    
    # 5. Combined weighted risk score with medical expertise weighting
    members['medical_risk_score'] = (
        members['chronic_risk_score'] * 0.3 +  # 30% weight
        members['cancer_risk_score'] * 0.3 +   # 30% weight
        members['lifestyle_risk_score'] * 0.2 + # 20% weight
        members['age_risk'] * 0.15 +           # 15% weight
        members['bmi_risk'] * 0.05             # 5% weight
    )
    
    # 6. If we have claims data, incorporate it
    claims_risk_included = False
    if claims_df is not None:
        try:
            print("Attempting to incorporate claims data into risk scores...")
            
            # Check if claims_df has the required columns
            required_cols = ['Member_ID', 'TotPaymentUSD']
            missing_cols = [col for col in required_cols if col not in claims_df.columns]
            if missing_cols:
                raise ValueError(f"Claims data missing required columns: {missing_cols}")
            
            # Ensure Member_ID uses the same data type in both dataframes
            claims_df = claims_df.copy()
            
            # Convert Member_ID to the same data type in both dataframes
            if 'Member_ID' in members.columns and 'Member_ID' in claims_df.columns:
                # Use string conversion for robust matching
                members['Member_ID'] = members['Member_ID'].astype(str)
                claims_df['Member_ID'] = claims_df['Member_ID'].astype(str)
                
                print(f"Members data has {len(members['Member_ID'].unique())} unique Member_IDs")
                print(f"Claims data has {len(claims_df['Member_ID'].unique())} unique Member_IDs")
                
                # Check for overlap
                member_ids_in_claims = set(claims_df['Member_ID'].unique())
                member_ids_in_members = set(members['Member_ID'].unique())
                overlap = member_ids_in_members.intersection(member_ids_in_claims)
                print(f"Found {len(overlap)} overlapping Member_IDs between datasets")
            
            # Merge claims history summary
            member_claim_summary = claims_df.groupby('Member_ID').agg({
                'TotPaymentUSD': ['count', 'sum', 'mean', 'std']
            }).reset_index()
            
            # Flatten column names
            member_claim_summary.columns = [col[0] if col[1] == '' else f'claim_{col[0]}_{col[1]}' 
                                          for col in member_claim_summary.columns]
            
            # Merge with members
            before_merge = len(members)
            members = pd.merge(members, member_claim_summary, on='Member_ID', how='left')
            after_merge = len(members)
            
            if before_merge != after_merge:
                print(f"Warning: Row count changed after merge ({before_merge} → {after_merge}). Using left join instead.")
                # Revert and try again with more careful joining
                members = members_df.copy()
                members['Member_ID'] = members['Member_ID'].astype(str)
                temp_claim_summary = member_claim_summary.copy()
                temp_claim_summary['Member_ID'] = temp_claim_summary['Member_ID'].astype(str)
                
                members = pd.merge(members, temp_claim_summary, on='Member_ID', how='left')
            
            # Fill NaN values for members with no claims
            claim_cols = [col for col in members.columns if col.startswith('claim_')]
            members[claim_cols] = members[claim_cols].fillna(0)
            
            # Handle NaN values in standard deviation
            if 'claim_TotPaymentUSD_std' in members.columns:
                members['claim_TotPaymentUSD_std'] = members['claim_TotPaymentUSD_std'].fillna(0)
            
            # Create claims-based risk score
            # Standardize claim features
            scaler = StandardScaler()
            if len(members) > 10:  # Need reasonable number of members
                claims_features = members[['claim_TotPaymentUSD_count', 'claim_TotPaymentUSD_sum', 'claim_TotPaymentUSD_mean']]
                
                # Replace infinity with large values
                claims_features = claims_features.replace([np.inf, -np.inf], np.nan)
                claims_features = claims_features.fillna(claims_features.max().max() * 1.5)
                
                # Apply scaling
                members['claims_risk_score'] = scaler.fit_transform(claims_features).sum(axis=1)
            else:
                # Simple formula if too few members
                members['claims_risk_score'] = (
                    members['claim_TotPaymentUSD_count'] * 0.3 + 
                    (members['claim_TotPaymentUSD_sum'] / 100) * 0.3 + 
                    members['claim_TotPaymentUSD_mean'] * 0.4
                )
            
            # Normalize claims risk score to 0-10 scale
            members['claims_risk_score'] = members['claims_risk_score'].replace([np.inf, -np.inf], np.nan)
            members['claims_risk_score'] = members['claims_risk_score'].fillna(0)
            
            if members['claims_risk_score'].max() > 0:
                members['claims_risk_score'] = 10 * members['claims_risk_score'] / members['claims_risk_score'].max()
            
            # Update medical risk with claims information
            members['combined_risk_score'] = members['medical_risk_score'] * 0.7 + members['claims_risk_score'] * 0.3
            claims_risk_included = True
            
            print("Successfully incorporated claims data into risk scores")
        except Exception as e:
            print(f"Could not incorporate claims data into risk scores: {e}")
            members['combined_risk_score'] = members['medical_risk_score']
    else:
        members['combined_risk_score'] = members['medical_risk_score']
    
    # 7. Use PCA for dimension reduction if we have enough data
    if len(members) > 20 and len(questionnaire_cols) >= 3:  # Need reasonable data size
        try:
            # Standardize questionnaire data
            scaler = StandardScaler()
            questionnaire_data = scaler.fit_transform(members[questionnaire_cols])
            
            # Apply PCA to create risk components
            pca = PCA(n_components=min(3, len(questionnaire_cols)))
            risk_components = pca.fit_transform(questionnaire_data)
            
            # Add PCA components as features
            for i in range(risk_components.shape[1]):
                members[f'risk_component_{i+1}'] = risk_components[:, i]
            
            # Create PCA-based risk score (using absolute values since components can be negative)
            members['pca_risk_score'] = np.sum(np.abs(risk_components), axis=1)
            
            # Normalize to 0-10 scale
            if members['pca_risk_score'].max() > 0:
                members['pca_risk_score'] = 10 * members['pca_risk_score'] / members['pca_risk_score'].max()
        except Exception as e:
            print(f"Could not create PCA-based risk score: {e}")
            members['pca_risk_score'] = members['combined_risk_score']
    else:
        members['pca_risk_score'] = members['combined_risk_score']
    
    # 8. Cluster members into risk groups
    if len(members) >= 20:  # Need reasonable sample size for clustering
        try:
            # Select features for clustering
            cluster_features = ['medical_risk_score']
            if 'claims_risk_score' in members.columns:
                cluster_features.append('claims_risk_score')
            if 'pca_risk_score' in members.columns:
                cluster_features.append('pca_risk_score')
            
            # Standardize data
            X = StandardScaler().fit_transform(members[cluster_features])
            
            # Determine optimal number of clusters (2-5)
            inertia = []
            for k in range(2, min(6, len(members) // 4 + 1)):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(X)
                inertia.append(kmeans.inertia_)
            
            # Find elbow point (simplified)
            k_optimal = 2
            if len(inertia) > 1:
                # Simple elbow detection
                diffs = np.diff(inertia)
                if len(diffs) > 1 and abs(diffs[0]) > 2 * abs(diffs[1]):
                    k_optimal = 3
            
            # Apply KMeans with optimal k
            kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
            members['risk_cluster'] = kmeans.fit_predict(X)
            
            # Map clusters to risk levels based on centroids
            centroids = kmeans.cluster_centers_
            risk_level_map = {}
            
            # Calculate average risk for each cluster
            for cluster in range(k_optimal):
                avg_risk = np.mean(centroids[cluster])
                risk_level_map[cluster] = avg_risk
            
            # Sort clusters by risk level
            sorted_clusters = sorted(risk_level_map.items(), key=lambda x: x[1])
            
            # Assign risk levels
            risk_level_names = ['Low', 'Medium', 'High', 'Very High', 'Extreme']
            cluster_to_risk = {}
            
            for i, (cluster, _) in enumerate(sorted_clusters):
                if i < len(risk_level_names):
                    cluster_to_risk[cluster] = risk_level_names[i]
                else:
                    cluster_to_risk[cluster] = 'Unknown'
            
            # Map risk levels
            members['risk_level'] = members['risk_cluster'].map(cluster_to_risk)
            
        except Exception as e:
            print(f"Could not perform risk clustering: {e}")
            # Assign risk levels based on quartiles
            members['risk_cluster'] = pd.qcut(members['combined_risk_score'], 
                                            q=min(4, len(members) // 5), 
                                            labels=False, 
                                            duplicates='drop')
            
            risk_level_map = {
                0: 'Low',
                1: 'Medium',
                2: 'High',
                3: 'Very High'
            }
            members['risk_level'] = members['risk_cluster'].map(risk_level_map)
    else:
        # For small datasets, use simple quartiles
        members['risk_cluster'] = np.minimum(3, (members['combined_risk_score'] / 2.5).astype(int))
        risk_level_map = {
            0: 'Low',
            1: 'Medium',
            2: 'High',
            3: 'Very High'
        }
        members['risk_level'] = members['risk_cluster'].map(risk_level_map)
    
    # 9. Final risk score (normalized to 0-100 for easier interpretation)
    if claims_risk_included and 'claims_risk_score' in members.columns:
        members['final_risk_score'] = 0.4 * members['medical_risk_score'] + 0.3 * members['pca_risk_score'] + 0.3 * members['claims_risk_score']
    else:
        members['final_risk_score'] = 0.6 * members['medical_risk_score'] + 0.4 * members['pca_risk_score']
    
    # Normalize to 0-100
    if members['final_risk_score'].max() > 0:
        members['final_risk_score'] = 100 * members['final_risk_score'] / members['final_risk_score'].max()
    
    # Select only risk-related columns for output
    risk_cols = [
        'Member_ID', 'chronic_risk_score', 'cancer_risk_score', 'lifestyle_risk_score', 
        'age_risk', 'bmi_risk', 'gender_risk', 'medical_risk_score'
    ]
    
    if claims_risk_included and 'claims_risk_score' in members.columns:
        risk_cols.append('claims_risk_score')
    
    risk_cols.extend([
        'pca_risk_score', 'combined_risk_score', 'risk_cluster', 'risk_level', 
        'final_risk_score'
    ])
    
    # Keep only columns that exist in the DataFrame
    risk_cols = [col for col in risk_cols if col in members.columns]
    
    return members[risk_cols]

def create_risk_interaction_features(features_df, risk_scores_df):
    """
    Create interaction features between risk scores and other features
    """
    # Ensure consistent data types for Member_ID before merging
    features_df = features_df.copy()
    risk_scores_df = risk_scores_df.copy()
    
    # Convert Member_ID to string in both dataframes for consistent joining
    features_df['Member_ID'] = features_df['Member_ID'].astype(str)
    risk_scores_df['Member_ID'] = risk_scores_df['Member_ID'].astype(str)
    
    # Merge risk scores with features
    merged_df = pd.merge(features_df, risk_scores_df, on='Member_ID', how='left')
    
    # Fill missing risk scores with 0
    # Get only columns that exist in both risk_scores_df and merged_df
    risk_cols = [col for col in risk_scores_df.columns if 'risk' in col.lower() and col != 'Member_ID' and col in merged_df.columns]
    if risk_cols:
        merged_df[risk_cols] = merged_df[risk_cols].fillna(0)
    
    # Identify key features for interactions
    claim_features = [col for col in merged_df.columns if any(x in col for x in [
        'claim_count', 'claim_amount', 'claim_frequency', 'avg_claim', 
        'total_claims', 'max_claim'
    ])]
    
    # Create interactions for core risk scores - only if they exist
    key_risk_scores = [col for col in ['final_risk_score', 'medical_risk_score', 'chronic_risk_score'] if col in merged_df.columns]
    
    # Create interactions between risk scores and claim features
    for risk_col in key_risk_scores:
        for feature_col in claim_features:
            if feature_col in merged_df.columns:
                interaction_name = f"{risk_col}_x_{feature_col}"
                merged_df[interaction_name] = merged_df[risk_col] * merged_df[feature_col]
    
    # Create categorized risk indicators
    if 'risk_level' in merged_df.columns:
        # One-hot encode risk levels
        risk_levels = merged_df['risk_level'].dropna().unique()
        for level in risk_levels:
            merged_df[f'is_{level}_risk'] = (merged_df['risk_level'] == level).astype(int)
    
    return merged_df 