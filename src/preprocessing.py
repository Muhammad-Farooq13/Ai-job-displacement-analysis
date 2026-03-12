"""
Data Preprocessing Module

Handles data cleaning, missing value imputation, encoding, scaling, and 
train-test splitting. Applies transformations consistently across train/test sets.
"""

import logging
from typing import Tuple, Dict, Optional, Any
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
from pathlib import Path

from src.config import Config

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handles all data preprocessing operations."""
    
    def __init__(self):
        """Initialize preprocessor with scalers and encoders."""
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = None
        self.categorical_features = Config.CATEGORICAL_FEATURES.copy()
        self.numeric_features = Config.NUMERIC_FEATURES.copy()
        self.is_fitted = False
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: DataFrame to process
            strategy: 'median' for numeric, 'mode' for categorical
            
        Returns:
            pd.DataFrame: DataFrame with missing values handled
        """
        df = df.copy()
        
        missing_counts = df.isnull().sum()
        if missing_counts.sum() == 0:
            logger.info("✓ No missing values detected")
            return df
        
        # Handle numeric columns
        numeric_missing = df[self.numeric_features].isnull().sum()
        for col in numeric_missing[numeric_missing > 0].index:
            if strategy == "median":
                fill_value = df[col].median()
                df[col].fillna(fill_value, inplace=True)
                logger.info(f"  Filled {col} with median: {fill_value:.2f}")
            elif strategy == "mean":
                fill_value = df[col].mean()
                df[col].fillna(fill_value, inplace=True)
                logger.info(f"  Filled {col} with mean: {fill_value:.2f}")
        
        # Handle categorical columns
        categorical_missing = df[self.categorical_features].isnull().sum()
        for col in categorical_missing[categorical_missing > 0].index:
            fill_value = df[col].mode()[0]
            df[col].fillna(fill_value, inplace=True)
            logger.info(f"  Filled {col} with mode: {fill_value}")
        
        logger.info(f"✓ Missing values handled using strategy: {strategy}")
        return df
    
    def handle_outliers(self, df: pd.DataFrame, method: str = "iqr", threshold: float = 1.5) -> pd.DataFrame:
        """
        Detect and cap outliers.
        
        Args:
            df: DataFrame to process
            method: 'iqr' for Interquartile Range, 'zscore' for z-score
            threshold: IQR multiplier (1.5) or z-score threshold (3)
            
        Returns:
            pd.DataFrame: DataFrame with outliers capped
        """
        df = df.copy()
        outlier_cols = [
            'automation_risk_percent',
            'salary_change_percent',
            'skill_demand_growth_percent'
        ]
        
        for col in outlier_cols:
            if col not in df.columns:
                continue
            
            if method == "iqr":
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outliers_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                df[col] = df[col].clip(lower_bound, upper_bound)
                logger.info(f"  {col}: Capped {outliers_count} outliers [{lower_bound:.2f}, {upper_bound:.2f}]")
        
        logger.info(f"✓ Outliers handled using method: {method}")
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical features using Label Encoding.
        
        Args:
            df: DataFrame to encode
            fit: If True, fit encoders on this data; if False, use fitted encoders
            
        Returns:
            pd.DataFrame: DataFrame with encoded features
        """
        df = df.copy()
        
        for col in self.categorical_features:
            if col not in df.columns:
                continue
            
            if fit:
                # Fit encoder on this data
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
                logger.info(f"  Fitted encoder for {col}: {len(le.classes_)} classes")
            else:
                # Use fitted encoder
                if col not in self.label_encoders:
                    raise ValueError(f"Encoder for {col} not fitted. Call with fit=True first.")
                le = self.label_encoders[col]
                df[col] = le.transform(df[col].astype(str))
        
        logger.info(f"✓ Categorical features encoded" if fit else "✓ Encoding applied")
        return df
    
    def scale_numeric_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Scale numeric features using StandardScaler or MinMaxScaler.
        
        Args:
            df: DataFrame to scale
            fit: If True, fit scaler on this data; if False, use fitted scaler
            
        Returns:
            pd.DataFrame: DataFrame with scaled features
        """
        df = df.copy()
        
        if fit:
            if Config.SCALING_METHOD == "standard":
                self.scaler = StandardScaler()
            else:
                self.scaler = MinMaxScaler()
            
            df[self.numeric_features] = self.scaler.fit_transform(df[self.numeric_features])
            logger.info(f"✓ Scaler fitted and applied: {Config.SCALING_METHOD}")
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Call with fit=True first.")
            df[self.numeric_features] = self.scaler.transform(df[self.numeric_features])
            logger.info("✓ Scaling applied using fitted scaler")
        
        return df
    
    def remove_duplicates(self, df: pd.DataFrame, subset: Optional[list] = None) -> pd.DataFrame:
        """
        Remove duplicate rows.
        
        Args:
            df: DataFrame to process
            subset: Columns to consider for duplicates (default: job_id)
            
        Returns:
            pd.DataFrame: DataFrame with duplicates removed
        """
        df = df.copy()
        subset = subset or ['job_id']
        
        initial_len = len(df)
        df = df.drop_duplicates(subset=subset, keep='first')
        removed = initial_len - len(df)
        
        if removed > 0:
            logger.info(f"✓ Removed {removed} duplicate rows")
        else:
            logger.info("✓ No duplicates found")
        
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete preprocessing pipeline (fit phase).
        Apply all transformations and fit scalers/encoders.
        
        Args:
            df: Raw training DataFrame
            
        Returns:
            pd.DataFrame: Processed DataFrame
        """
        logger.info("Starting preprocessing pipeline (FIT)...")
        
        df = df.copy()
        
        # 1. Remove duplicates
        df = self.remove_duplicates(df)
        
        # 2. Handle missing values
        df = self.handle_missing_values(df, strategy="median")
        
        # 3. Handle outliers
        df = self.handle_outliers(df, method="iqr")
        
        # 4. Encode categorical features
        df = self.encode_categorical_features(df, fit=True)
        
        # 5. Scale numeric features
        df = self.scale_numeric_features(df, fit=True)
        
        # Store feature names
        self.feature_names = [col for col in df.columns if col != Config.TARGET_VARIABLE and col != Config.ID_COLUMN]
        self.is_fitted = True
        
        logger.info(f"✓ Preprocessing complete. Selected {len(self.feature_names)} features")
        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply preprocessing using fitted scalers/encoders.
        
        Args:
            df: DataFrame to transform
            
        Returns:
            pd.DataFrame: Processed DataFrame
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit_transform() first.")
        
        logger.info("Applying fitted preprocessing...")
        
        df = df.copy()
        df = self.remove_duplicates(df)
        df = self.handle_missing_values(df, strategy="median")
        df = self.handle_outliers(df, method="iqr")
        df = self.encode_categorical_features(df, fit=False)
        df = self.scale_numeric_features(df, fit=False)
        
        logger.info("✓ Preprocessing applied to new data")
        return df
    
    def split_data(
        self,
        df: pd.DataFrame,
        test_size: float = 0.3,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split DataFrame into train/test X/y using Config.TARGET_VARIABLE."""
        X = df.drop([Config.TARGET_VARIABLE, Config.ID_COLUMN], axis=1, errors="ignore")
        y = df[Config.TARGET_VARIABLE]
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def fit_transform_split(
        self, df: pd.DataFrame, test_size: float = 0.3
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Preprocess and split data into train and test sets.

        Args:
            df: DataFrame to preprocess and split
            test_size: Proportion for test set

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Train and test DataFrames
        """
        # First fit transform on full data
        processed_df = self.fit_transform(df)
        
        # Split using stratified split on job_role
        if 'job_role' in df.columns:
            X = processed_df.drop([Config.TARGET_VARIABLE, Config.ID_COLUMN], axis=1, errors='ignore')
            y = processed_df[Config.TARGET_VARIABLE]
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=Config.RANDOM_STATE
            )
            
            train_df = pd.concat([X_train, y_train], axis=1)
            test_df = pd.concat([X_test, y_test], axis=1)
        else:
            # Random split if no stratification column
            train_df, test_df = train_test_split(
                processed_df, test_size=test_size, random_state=Config.RANDOM_STATE
            )
        
        logger.info(f"✓ Train set: {len(train_df)} records, Test set: {len(test_df)} records")
        return train_df, test_df
    
    def save_artifacts(self, output_dir: Path = None) -> None:
        """
        Save fitted scalers and encoders.
        
        Args:
            output_dir: Directory to save artifacts
        """
        output_dir = output_dir or Config.MODELS_PATH / "model_artifacts"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.scaler:
            with open(output_dir / "feature_scaler.pkl", "wb") as f:
                pickle.dump(self.scaler, f)
            logger.info(f"✓ Saved scaler to {output_dir / 'feature_scaler.pkl'}")
        
        if self.label_encoders:
            with open(output_dir / "label_encoders.pkl", "wb") as f:
                pickle.dump(self.label_encoders, f)
            logger.info(f"✓ Saved encoders to {output_dir / 'label_encoders.pkl'}")
        
        if self.feature_names:
            import json
            with open(output_dir / "feature_names.json", "w") as f:
                json.dump(self.feature_names, f)
            logger.info(f"✓ Saved feature names to {output_dir / 'feature_names.json'}")
    
    def load_artifacts(self, input_dir: Path = None) -> None:
        """
        Load fitted scalers and encoders.
        
        Args:
            input_dir: Directory containing artifacts
        """
        input_dir = input_dir or Config.MODELS_PATH / "model_artifacts"
        
        scaler_path = input_dir / "feature_scaler.pkl"
        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
            logger.info(f"✓ Loaded scaler from {scaler_path}")
        
        encoders_path = input_dir / "label_encoders.pkl"
        if encoders_path.exists():
            with open(encoders_path, "rb") as f:
                self.label_encoders = pickle.load(f)
            logger.info(f"✓ Loaded encoders from {encoders_path}")
        
        features_path = input_dir / "feature_names.json"
        if features_path.exists():
            import json
            with open(features_path, "r") as f:
                self.feature_names = json.load(f)
            logger.info(f"✓ Loaded feature names from {features_path}")
        
        self.is_fitted = True


if __name__ == "__main__":
    from src.data_loader import DataLoader
    
    # Example usage
    logger.info("Starting data preprocessing pipeline...")
    
    # Load data
    loader = DataLoader()
    df = loader.load_data()
    loader.validate()
    
    # Preprocess
    preprocessor = DataPreprocessor()
    train_df, test_df = preprocessor.fit_transform_split(df, test_size=0.3)
    
    # Save processed data
    train_df.to_parquet(Config.PROCESSED_TRAIN_FILE)
    test_df.to_parquet(Config.PROCESSED_TEST_FILE)
    preprocessor.save_artifacts()
    
    logger.info(f"✓ Preprocessing complete. Train: {len(train_df)}, Test: {len(test_df)}")
