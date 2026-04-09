import numpy as np
import pandas as pd

def engineer_features(df):
    """
    Create engineered features for the loan default model.
    """
    # 1. Average FICO Score
    if 'fico_range_low' in df.columns and 'fico_range_high' in df.columns:
        df['fico_avg'] = (df['fico_range_low'] + df['fico_range_high']) / 2
        df.drop(['fico_range_low', 'fico_range_high'], axis=1, inplace=True)
        
    # 2. Loan-to-Income Ratio
    if 'loan_amnt' in df.columns and 'annual_inc' in df.columns:
        df['loan_to_income'] = df['loan_amnt'] / df['annual_inc']
        
    # 3. Installment-to-Income Ratio
    if 'installment' in df.columns and 'annual_inc' in df.columns:
        df['installment_to_income'] = df['installment'] / df['annual_inc']
        
    # 4. Use log transforms for skewed variables (during preprocessing or here)
    if 'annual_inc' in df.columns:
        df['annual_inc_log'] = np.log1p(df['annual_inc'])
        
    # 5. Credit History Flag
    if 'pub_rec' in df.columns or 'delinq_2yrs' in df.columns:
        df['has_derog_record'] = ((df.get('pub_rec', 0) > 0) | (df.get('delinq_2yrs', 0) > 0)).astype(int)
        
    return df
