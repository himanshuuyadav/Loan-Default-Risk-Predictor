import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

def load_data(filepath, nrows=None):
    """
    Load the dataset from a CSV file.
    """
    return pd.read_csv(filepath, low_memory=False, nrows=nrows)

def filter_valid_loans(df):
    """
    Filter for completed loans only (Fully Paid, Charged Off, Default).
    """
    valid_statuses = ['Fully Paid', 'Charged Off', 'Default']
    return df[df['loan_status'].isin(valid_statuses)].copy()

def create_binary_target(df):
    """
    Create a binary target variable 'default' (1 for default/charged off, 0 for fully paid).
    """
    df['default'] = df['loan_status'].map({
        'Fully Paid': 0,
        'Charged Off': 1,
        'Default': 1
    })
    return df

def handle_missing_values(df, numeric_cols=None, cat_cols=None):
    """
    Fill missing values for numeric and categorical columns.
    """
    # Fill numeric nulls with median
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include='number').columns.drop('default', errors='ignore')
    
    for col in numeric_cols:
        df[col].fillna(df[col].median(), inplace=True)

    # Fill categorical nulls with mode
    if cat_cols is None:
        cat_cols = df.select_dtypes(include='object').columns
        
    for col in cat_cols:
        if not df[col].empty:
            df[col].fillna(df[col].mode()[0], inplace=True)
            
    return df

def clean_columns(df):
    """
    Clean specific columns like emp_length and int_rate.
    """
    if 'emp_length' in df.columns:
        df['emp_length'] = df['emp_length'].str.extract(r'(\d+)').astype(float)
        df['emp_length'].fillna(df['emp_length'].median(), inplace=True)
        
    if 'int_rate' in df.columns and df['int_rate'].dtype == object:
        df['int_rate'] = df['int_rate'].str.replace('%', '').astype(float)
        
    return df

def encode_categorical(df, ordinal_cols=None, nominal_cols=None):
    """
    Perform encoding for categorical variables.
    """
    # Example ordinal encoding for grade
    if 'grade' in df.columns:
        grade_order = [['A', 'B', 'C', 'D', 'E', 'F', 'G']]
        oe = OrdinalEncoder(categories=grade_order)
        df['grade_encoded'] = oe.fit_transform(df[['grade']])
        df.drop('grade', axis=1, inplace=True)
        
    # Example one-hot encoding
    if nominal_cols:
        df = pd.get_dummies(df, columns=nominal_cols, drop_first=True)
        
    return df
