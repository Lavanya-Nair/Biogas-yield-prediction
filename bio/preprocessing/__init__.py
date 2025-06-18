import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(df):
    """
    Preprocess the biogas dataset with improved handling of missing values and features
    """
    # Create a copy
    df_processed = df.copy()
    
    # Select relevant features
    features = ['Cattle', 'Dairy', 'Poultry', 'Swine', 
               'Year Operational', 'Digester Type', 'Co-Digestion']
    target = 'Biogas Generation Estimate (cu-ft/day)'
    
    # Remove rows where target variable is NaN
    df_processed = df_processed.dropna(subset=[target])
    
    # Handle Digester Type more intelligently
    digester_type_counts = df_processed['Digester Type'].value_counts()
    common_types = digester_type_counts[digester_type_counts > 5].index
    df_processed['Digester Type'] = df_processed['Digester Type'].apply(
        lambda x: x if pd.notna(x) and x in common_types else 'Other'
    )
    
    # One-hot encode Digester Type instead of label encoding
    digester_dummies = pd.get_dummies(df_processed['Digester Type'], prefix='Digester')
    df_processed = pd.concat([df_processed, digester_dummies], axis=1)
    
    # Handle missing values in features
    df_processed['Year Operational'].fillna(df_processed['Year Operational'].median(), inplace=True)
    
    # More intelligent handling of animal counts
    animal_cols = ['Cattle', 'Dairy', 'Poultry', 'Swine']
    for col in animal_cols:
        # Fill missing values with 0 only if other animal columns are not null
        mask = df_processed[animal_cols].notna().any(axis=1)
        df_processed.loc[mask, col] = df_processed.loc[mask, col].fillna(0)
    
    # Drop rows where all animal counts are missing
    df_processed = df_processed.dropna(subset=animal_cols, how='all')
    
    # Improved Co-Digestion handling
    df_processed['Co-Digestion'] = df_processed['Co-Digestion'].map({'Yes': 1, 'No': 0, np.nan: 0})
    
    # Log transform the target variable (since biogas production often follows log-normal distribution)
    df_processed[target] = np.log1p(df_processed[target])
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_features = ['Cattle', 'Dairy', 'Poultry', 'Swine', 'Year Operational']
    df_processed[numerical_features] = scaler.fit_transform(df_processed[numerical_features])
    
    # Update features list to include one-hot encoded columns
    features = ['Cattle', 'Dairy', 'Poultry', 'Swine', 
               'Year Operational', 'Co-Digestion'] + list(digester_dummies.columns)
    
    # Select final features and target
    X = df_processed[features]
    y = df_processed[target]
    
    return X, y, scaler
