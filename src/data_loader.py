"""
Data Loading Module

Handles data ingestion, validation, and initial loading from multiple sources.
Provides classes for loading raw data and handling different data formats.
"""

import logging
from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
import numpy as np
from src.config import Config

logger = logging.getLogger(__name__)


class DataValidator:
    """Validates data schema and integrity."""
    
    EXPECTED_COLUMNS = {
        'job_id': 'int64',
        'job_role': 'object',
        'industry': 'object',
        'country': 'object',
        'year': 'int64',
        'automation_risk_percent': 'float64',
        'ai_replacement_score': 'float64',
        'skill_gap_index': 'float64',
        'salary_before_usd': 'float64',
        'salary_after_usd': 'float64',
        'salary_change_percent': 'float64',
        'skill_demand_growth_percent': 'float64',
        'remote_feasibility_score': 'float64',
        'ai_adoption_level': 'float64',
        'education_requirement_level': 'int64'
    }
    
    @classmethod
    def validate_schema(cls, df: pd.DataFrame) -> bool:
        """
        Validate dataframe schema against expected columns and types.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            bool: True if valid, raises ValueError otherwise
        """
        # Check columns exist
        missing_cols = set(cls.EXPECTED_COLUMNS.keys()) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        
        # Check data types
        for col, expected_dtype in cls.EXPECTED_COLUMNS.items():
            if df[col].dtype != expected_dtype:
                logger.warning(
                    f"Column '{col}' has type {df[col].dtype}, "
                    f"expected {expected_dtype}. Attempting conversion."
                )
                try:
                    df[col] = df[col].astype(expected_dtype)
                except Exception as e:
                    raise ValueError(f"Cannot convert column '{col}': {e}")
        
        logger.info(f"✓ Schema validation passed for {len(df)} records")
        return True
    
    @classmethod
    def validate_values(cls, df: pd.DataFrame) -> bool:
        """
        Validate value ranges and logical constraints.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            bool: True if valid
        """
        issues = []
        
        # Check numeric ranges
        numeric_cols = {
            'automation_risk_percent': (0, 100),
            'ai_replacement_score': (0, 100),
            'skill_gap_index': (0, 100),
            'remote_feasibility_score': (0, 100),
            'ai_adoption_level': (0, 100)
        }
        
        for col, (min_val, max_val) in numeric_cols.items():
            out_of_range = df[(df[col] < min_val) | (df[col] > max_val)]
            if len(out_of_range) > 0:
                issues.append(
                    f"Column '{col}' has {len(out_of_range)} values "
                    f"outside range [{min_val}, {max_val}]"
                )
        
        # Check year range
        year_range = df['year'].min(), df['year'].max()
        if year_range[0] < 2020 or year_range[1] > 2026:
            issues.append(f"Year range {year_range} outside expected [2020, 2026]")
        
        # Check for negative salaries
        if (df['salary_before_usd'] < 0).any() or (df['salary_after_usd'] < 0).any():
            issues.append("Negative salary values detected")
        
        # Check for duplicates
        duplicates = df[df.duplicated(subset=['job_id'], keep=False)]
        if len(duplicates) > 0:
            issues.append(f"Found {len(duplicates) // 2} duplicate job_id values")
        
        if issues:
            for issue in issues:
                logger.warning(f"⚠ {issue}")
        else:
            logger.info("✓ Value validation passed")
        
        return True


class DataLoader:
    """Loads and explores raw data from CSV files."""
    
    def __init__(self, data_path: Optional[Path] = None):
        """
        Initialize DataLoader.
        
        Args:
            data_path: Path to CSV file. Defaults to Config.RAW_DATA_FILE
        """
        self.data_path = data_path or Config.RAW_DATA_FILE
        self.df = None
        self.validator = DataValidator()
    
    def load_data(self) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Returns:
            pd.DataFrame: Loaded dataframe
            
        Raises:
            FileNotFoundError: If data file doesn't exist
            pd.errors.ParserError: If CSV parsing fails
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        try:
            self.df = pd.read_csv(self.data_path)
            logger.info(f"✓ Loaded {len(self.df)} records from {self.data_path.name}")
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
        
        return self.df
    
    def validate(self) -> bool:
        """
        Validate loaded data.
        
        Returns:
            bool: True if valid
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        self.validator.validate_schema(self.df)
        self.validator.validate_values(self.df)
        return True
    
    def get_data_info(self) -> dict:
        """
        Get comprehensive data information.
        
        Returns:
            dict: Data statistics and metadata
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        info = {
            "shape": self.df.shape,
            "columns": self.df.columns.tolist(),
            "dtypes": self.df.dtypes.to_dict(),
            "missing_values": self.df.isnull().sum().to_dict(),
            "missing_percent": (self.df.isnull().sum() / len(self.df) * 100).round(2).to_dict(),
            "numeric_stats": self.df.describe().to_dict(),
            "categorical_unique": {col: self.df[col].nunique() 
                                   for col in self.df.select_dtypes(include=['object']).columns},
            "year_range": (self.df['year'].min(), self.df['year'].max()),
            "target_stats": {
                "mean": self.df[Config.TARGET_VARIABLE].mean(),
                "std": self.df[Config.TARGET_VARIABLE].std(),
                "min": self.df[Config.TARGET_VARIABLE].min(),
                "max": self.df[Config.TARGET_VARIABLE].max()
            }
        }
        return info
    
    def display_info(self) -> None:
        """Display data information in readable format."""
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        print("\n" + "="*80)
        print("DATASET INFORMATION")
        print("="*80)
        print(f"\nShape: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
        print(f"\nColumn Names and Types:")
        print(self.df.dtypes)
        print(f"\nMissing Values:")
        print(self.df.isnull().sum())
        print(f"\nBasic Statistics:")
        print(self.df.describe())
        print(f"\nCategorical Features Unique Values:")
        for col in Config.CATEGORICAL_FEATURES:
            if col in self.df.columns:
                print(f"  {col}: {self.df[col].nunique()} unique values")
        print("\n" + "="*80)
    
    def split_by_year(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data by year into train and test sets.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Train and test dataframes
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        train_df = self.df[self.df['year'].isin(Config.TRAIN_YEARS)].copy()
        test_df = self.df[self.df['year'].isin(Config.TEST_YEARS)].copy()
        
        logger.info(
            f"✓ Split data: {len(train_df)} train records (years {Config.TRAIN_YEARS}), "
            f"{len(test_df)} test records (years {Config.TEST_YEARS})"
        )
        return train_df, test_df


def load_processed_data(
    train_file: Optional[Path] = None,
    test_file: Optional[Path] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load processed data from parquet files.
    
    Args:
        train_file: Path to train parquet file
        test_file: Path to test parquet file
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Train and test dataframes
    """
    train_file = train_file or Config.PROCESSED_TRAIN_FILE
    test_file = test_file or Config.PROCESSED_TEST_FILE
    
    if not train_file.exists() or not test_file.exists():
        raise FileNotFoundError(
            f"Processed data files not found. "
            f"Run preprocessing pipeline first."
        )
    
    train_df = pd.read_parquet(train_file)
    test_df = pd.read_parquet(test_file)
    
    logger.info(f"✓ Loaded {len(train_df)} train and {len(test_df)} test records")
    return train_df, test_df


if __name__ == "__main__":
    # Example usage
    loader = DataLoader()
    df = loader.load_data()
    loader.validate()
    loader.display_info()
    
    train, test = loader.split_by_year()
    print(f"\nTrain set size: {len(train)}")
    print(f"Test set size: {len(test)}")
