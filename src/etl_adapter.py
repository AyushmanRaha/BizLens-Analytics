import pandas as pd
import numpy as np

class DataStandardizer:
    """
    The ETL Layer: Converts 'Client Data' into 'BizLens Standard Data'.
    """
    def __init__(self):
        # The strict schema BizLens expects
        self.REQUIRED_COLUMNS = [
            'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
            'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
            'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
            'MonthlyCharges', 'TotalCharges'
        ]

    def check_schema(self, df: pd.DataFrame) -> list:
        """
        Returns a list of columns that are MISSING from the uploaded file.
        """
        missing = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        return missing

    def transform(self, df: pd.DataFrame, column_mapping: dict) -> pd.DataFrame:
        """
        Applies the mapping to rename client columns to BizLens standard columns.
        """
        # 1. Rename Columns
        df_clean = df.rename(columns=column_mapping).copy()
        
        # 2. Filter only necessary columns (ignore extra client junk data)
        # Only keep columns that are now valid
        valid_cols = [c for c in self.REQUIRED_COLUMNS if c in df_clean.columns]
        df_clean = df_clean[valid_cols]

        # 3. Basic Type Cleaning (Robustness)
        if 'TotalCharges' in df_clean.columns:
            # Force numeric, turning errors to 0 or mean
            df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce').fillna(0)
            
        if 'tenure' in df_clean.columns:
            df_clean['tenure'] = pd.to_numeric(df_clean['tenure'], errors='coerce').fillna(0)

        # 4. Value Standardization (Optional: Lowercase simple text)
        # This ensures 'male' becomes 'Male' if your encoder is sensitive
        # (For this MVP, we rely on the pipeline to handle unknown categories gracefully)
        
        return df_clean