import pandas as pd
import numpy as np

class DataPreprocessor:
    @staticmethod
    def clean_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy() # avoid modifying raw input inplace
        
        # Normalize casing variations
        df['Gender'] = df['Gender'].str.upper()
        df['SubscriptionType'] = df['SubscriptionType'].str.capitalize()
        
        # Clean currency characters and cast to numeric
        if df['MonthlyCharges'].dtype == 'object':
            df['MonthlyCharges'] = df['MonthlyCharges'].str.replace('$', '', regex=False).astype(float)
            
        # Clean string typos to nulls and convert numeric text
        df['TotalSpent'] = df['TotalSpent'].replace('Missing', np.nan)
        if df['TotalSpent'].dtype == 'object':
            df['TotalSpent'] = df['TotalSpent'].str.replace('$', '', regex=False).astype(float)
            
        # Median imputation prevents outlier skew
        median_spent = df['TotalSpent'].median()
        df['TotalSpent'] = df['TotalSpent'].fillna(median_spent)
        
        df['ChurnStatus'] = df['ChurnStatus'].map({'No': 0, 'Yes': 1})
        
        # One-hot encode categories (drop_first avoids the dummy variable trap)
        df = pd.get_dummies(df, columns=['Gender', 'SubscriptionType'], drop_first=True)
        
        # Force pandas boolean flags to ints for scikit-learn stability
        for col in df.columns:
            if df[col].dtype == 'bool':
                df[col] = df[col].astype(int)
                
        return df