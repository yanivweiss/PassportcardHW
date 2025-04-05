import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import KNNImputer
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os

def handle_missing_values_advanced(df, categorical_strategy='mode', numerical_strategy='knn'):
    """
    Advanced missing value handling with multiple strategies
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataframe to process
    categorical_strategy : str
        Strategy for categorical columns: 'mode', 'missing_category', or 'drop'
    numerical_strategy : str
        Strategy for numerical columns: 'mean', 'median', 'knn', or 'zero'
        
    Returns:
    --------
    pandas DataFrame
        Processed dataframe with missing values handled
    """
    print(f"Handling missing values with {categorical_strategy} for categorical and {numerical_strategy} for numerical")
    
    # Make a copy to avoid modifying the original
    df_processed = df.copy()
    
    # Get column types
    categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns
    numerical_cols = df_processed.select_dtypes(include=['int', 'float']).columns
    
    # Handle categorical features
    for col in categorical_cols:
        missing_count = df_processed[col].isna().sum()
        if missing_count > 0:
            print(f"Column {col}: {missing_count} missing values ({missing_count/len(df_processed)*100:.2f}%)")
            
            if categorical_strategy == 'mode':
                # Replace with mode
                mode_value = df_processed[col].mode()[0]
                df_processed[col].fillna(mode_value, inplace=True)
                
            elif categorical_strategy == 'missing_category':
                # Create a new category for missing values
                df_processed[col].fillna('Missing', inplace=True)
                
            elif categorical_strategy == 'drop' and missing_count/len(df_processed) > 0.5:
                # Drop column if too many missing values
                df_processed.drop(col, axis=1, inplace=True)
                print(f"Dropped column {col} due to high missing ratio")
    
    # Handle numerical features
    if numerical_strategy == 'knn':
        # Use KNN imputation for numerical features
        numerical_data = df_processed[numerical_cols]
        
        # Handle infinite values before KNN imputation
        numerical_data = numerical_data.replace([np.inf, -np.inf], np.nan)
        
        # Initialize and fit the KNN imputer
        imputer = KNNImputer(n_neighbors=5)
        imputed_data = imputer.fit_transform(numerical_data)
        
        # Update the dataframe with imputed values
        df_processed[numerical_cols] = imputed_data
        
    else:
        for col in numerical_cols:
            missing_count = df_processed[col].isna().sum()
            if missing_count > 0:
                print(f"Column {col}: {missing_count} missing values ({missing_count/len(df_processed)*100:.2f}%)")
                
                if numerical_strategy == 'mean':
                    # Replace with mean
                    df_processed[col].fillna(df_processed[col].mean(), inplace=True)
                    
                elif numerical_strategy == 'median':
                    # Replace with median
                    df_processed[col].fillna(df_processed[col].median(), inplace=True)
                    
                elif numerical_strategy == 'zero':
                    # Replace with zero
                    df_processed[col].fillna(0, inplace=True)
    
    return df_processed

def detect_and_handle_outliers(df, columns=None, method='iqr', threshold=1.5, visualization=True):
    """
    Detect and handle outliers in numerical features
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataframe to process
    columns : list, optional
        List of columns to check for outliers. If None, all numerical columns will be checked.
    method : str
        Method to detect outliers: 'iqr' or 'zscore'
    threshold : float
        Threshold for outlier detection (1.5 for IQR, 3 for z-score)
    visualization : bool
        Whether to create visualization of outliers
        
    Returns:
    --------
    tuple
        (processed_df, outlier_info) where outlier_info is a dict with info about outliers
    """
    # Make a copy to avoid modifying the original
    df_processed = df.copy()
    
    # If no columns provided, use all numerical columns
    if columns is None:
        columns = df_processed.select_dtypes(include=['int', 'float']).columns
    
    outlier_info = {}
    
    # Create visualizations directory if it doesn't exist and visualization is True
    if visualization:
        os.makedirs('visualizations/outliers', exist_ok=True)
    
    for col in columns:
        if df_processed[col].dtype in ['int64', 'float64']:
            if method == 'iqr':
                # Calculate IQR
                Q1 = df_processed[col].quantile(0.25)
                Q3 = df_processed[col].quantile(0.75)
                IQR = Q3 - Q1
                
                # Define outlier boundaries
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                # Identify outliers
                outliers = df_processed[(df_processed[col] < lower_bound) | (df_processed[col] > upper_bound)]
                
            elif method == 'zscore':
                # Calculate z-scores
                z_scores = np.abs(stats.zscore(df_processed[col], nan_policy='omit'))
                
                # Identify outliers
                outliers = df_processed[z_scores > threshold]
            
            # Handle outliers by capping
            if not outliers.empty:
                outlier_info[col] = {
                    'count': len(outliers),
                    'percentage': len(outliers) / len(df_processed) * 100,
                    'min': outliers[col].min(),
                    'max': outliers[col].max()
                }
                
                print(f"Column {col}: {len(outliers)} outliers detected ({len(outliers)/len(df_processed)*100:.2f}%)")
                
                if method == 'iqr':
                    # Cap outliers at boundaries
                    df_processed[col] = df_processed[col].clip(lower=lower_bound, upper=upper_bound)
                    
                    if visualization:
                        plt.figure(figsize=(10, 6))
                        plt.boxplot(df[col].dropna())
                        plt.title(f'Boxplot of {col} showing outliers')
                        plt.savefig(f'visualizations/outliers/boxplot_{col}.png')
                        plt.close()
                        
                elif method == 'zscore':
                    # Identify indices with outliers
                    outlier_indices = np.where(z_scores > threshold)[0]
                    
                    # Replace outliers with median
                    median_value = df_processed[col].median()
                    df_processed.loc[df_processed.index[outlier_indices], col] = median_value
                    
                    if visualization:
                        plt.figure(figsize=(10, 6))
                        plt.scatter(range(len(z_scores)), z_scores)
                        plt.axhline(y=threshold, color='r', linestyle='--')
                        plt.title(f'Z-scores for {col} with threshold at {threshold}')
                        plt.savefig(f'visualizations/outliers/zscore_{col}.png')
                        plt.close()
    
    return df_processed, outlier_info

def scale_features(df, method='standard', columns=None):
    """
    Scale numerical features using different methods
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataframe to process
    method : str
        Scaling method: 'standard', 'robust', 'minmax'
    columns : list, optional
        List of columns to scale. If None, all numerical columns will be scaled.
        
    Returns:
    --------
    pandas DataFrame
        Dataframe with scaled features and scaler object
    """
    # Make a copy to avoid modifying the original
    df_processed = df.copy()
    
    # If no columns provided, use all numerical columns
    if columns is None:
        columns = df_processed.select_dtypes(include=['int', 'float']).columns.tolist()
        
        # We want to exclude any ID columns or target variables
        exclude = [col for col in columns if 'ID' in col or 'id' in col or 'future' in col]
        columns = [col for col in columns if col not in exclude]
    
    # Choose scaler based on method
    if method == 'standard':
        scaler = StandardScaler()
        print("Using StandardScaler: transforms to zero mean and unit variance")
    elif method == 'robust':
        scaler = RobustScaler()
        print("Using RobustScaler: robust to outliers using median and IQR")
    elif method == 'minmax':
        scaler = MinMaxScaler()
        print("Using MinMaxScaler: transforms to a range between 0 and 1")
    else:
        raise ValueError(f"Unknown scaling method: {method}")
    
    # Scale selected columns
    if columns:
        df_processed[columns] = scaler.fit_transform(df_processed[columns])
    
    return df_processed, scaler

def enhanced_data_preparation(df, missing_strategy='knn', outlier_method='iqr', 
                             scaling_method='robust', visualization=True):
    """
    Complete pipeline for enhanced data preparation
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataframe to process
    missing_strategy : str
        Strategy for numerical missing values
    outlier_method : str
        Method for outlier detection and handling
    scaling_method : str
        Method for feature scaling
    visualization : bool
        Whether to create visualizations
        
    Returns:
    --------
    pandas DataFrame
        Fully processed dataframe ready for modeling
    """
    print("Starting enhanced data preparation pipeline...")
    
    # 1. Handle missing values
    df_processed = handle_missing_values_advanced(
        df, 
        categorical_strategy='mode', 
        numerical_strategy=missing_strategy
    )
    
    # 2. Detect and handle outliers
    df_processed, outlier_info = detect_and_handle_outliers(
        df_processed, 
        method=outlier_method, 
        visualization=visualization
    )
    
    # 3. Scale features
    df_scaled, scaler = scale_features(
        df_processed,
        method=scaling_method
    )
    
    print("Enhanced data preparation completed successfully!")
    
    # Optional: Create a correlation heatmap
    if visualization:
        os.makedirs('visualizations', exist_ok=True)
        
        # Select only numeric columns for correlation
        numeric_cols = df_scaled.select_dtypes(include=['int', 'float']).columns
        
        # Create a correlation matrix visualization
        plt.figure(figsize=(12, 10))
        corr_matrix = df_scaled[numeric_cols].corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', annot=False, 
                    center=0, linewidths=0.5, vmin=-1, vmax=1)
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig('visualizations/correlation_heatmap.png')
        plt.close()
        
    return df_scaled, scaler, outlier_info

if __name__ == "__main__":
    # Example usage
    print("This is a module for enhanced data preparation. Import and use in your main script.") 