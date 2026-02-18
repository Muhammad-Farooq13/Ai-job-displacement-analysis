"""
Unit tests for data_loader module.

Tests DataValidator and DataLoader classes for proper schema validation,
error handling, and data ingestion.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.data_loader import DataValidator, DataLoader


class TestDataValidator:
    """Test data validation functionality."""
    
    def test_validate_schema_valid_data(self):
        """Test schema validation with valid DataFrame."""
        df = pd.DataFrame({
            'job_id': [1, 2, 3],
            'job_role': ['Engineer', 'Teacher', 'Analyst'],
            'industry': ['Tech', 'Education', 'Finance'],
            'country': ['USA', 'USA', 'USA'],
            'year': [2020, 2021, 2022],
            'automation_risk_percent': [0.5, 0.3, 0.7],
            'ai_replacement_score': [50, 30, 70],
            'skill_gap_index': [0.6, 0.4, 0.5],
            'salary_before_usd': [100000, 60000, 80000],
            'salary_after_usd': [105000, 62000, 75000],
            'salary_change_percent': [5, 3.3, -6.3],
            'skill_demand_growth_percent': [10, 5, 15],
            'remote_feasibility_score': [0.8, 0.3, 0.6],
            'ai_adoption_level': [0.6, 0.2, 0.5],
            'education_requirement_level': [3, 2, 3]
        })
        
        # Convert to correct dtypes
        df = df.astype(DataValidator.EXPECTED_COLUMNS)
        
        assert DataValidator.validate_schema(df) is True
    
    def test_validate_schema_missing_columns(self):
        """Test schema validation with missing columns."""
        df = pd.DataFrame({'job_id': [1, 2, 3]})
        
        with pytest.raises(ValueError, match="Missing required columns"):
            DataValidator.validate_schema(df)
    
    def test_validate_schema_wrong_dtypes(self):
        """Test schema validation with wrong data types."""
        df = pd.DataFrame({
            'job_id': ['a', 'b', 'c'],  # Wrong type
            'job_role': ['Engineer', 'Teacher', 'Analyst'],
            'industry': ['Tech', 'Education', 'Finance'],
            'country': ['USA', 'USA', 'USA'],
            'year': [2020, 2021, 2022],
            'automation_risk_percent': [0.5, 0.3, 0.7],
            'ai_replacement_score': [50, 30, 70],
            'skill_gap_index': [0.6, 0.4, 0.5],
            'salary_before_usd': [100000, 60000, 80000],
            'salary_after_usd': [105000, 62000, 75000],
            'salary_change_percent': [5, 3.3, -6.3],
            'skill_demand_growth_percent': [10, 5, 15],
            'remote_feasibility_score': [0.8, 0.3, 0.6],
            'ai_adoption_level': [0.6, 0.2, 0.5],
            'education_requirement_level': [3, 2, 3]
        })
        
        with pytest.raises(ValueError, match="Schema validation failed"):
            DataValidator.validate_schema(df)


class TestDataLoader:
    """Test data loading functionality."""
    
    def test_load_data_file_not_found(self):
        """Test handling of missing data file."""
        loader = DataLoader()
        
        # Attempt to load non-existent file
        with pytest.raises(FileNotFoundError):
            loader.load_raw_data(path="non_existent_file.csv")
    
    def test_load_data_valid_file(self, tmp_path):
        """Test successful data loading."""
        # Create temporary CSV file
        df_temp = pd.DataFrame({
            'job_id': [1, 2, 3],
            'job_role': ['Engineer', 'Teacher', 'Analyst'],
            'industry': ['Tech', 'Education', 'Finance'],
            'country': ['USA', 'USA', 'USA'],
            'year': [2020, 2021, 2022],
            'automation_risk_percent': [0.5, 0.3, 0.7],
            'ai_replacement_score': [50, 30, 70],
            'skill_gap_index': [0.6, 0.4, 0.5],
            'salary_before_usd': [100000, 60000, 80000],
            'salary_after_usd': [105000, 62000, 75000],
            'salary_change_percent': [5, 3.3, -6.3],
            'skill_demand_growth_percent': [10, 5, 15],
            'remote_feasibility_score': [0.8, 0.3, 0.6],
            'ai_adoption_level': [0.6, 0.2, 0.5],
            'education_requirement_level': [3, 2, 3]
        })
        temp_file = tmp_path / "test_data.csv"
        df_temp.to_csv(temp_file, index=False)
        
        loader = DataLoader()
        df_loaded = loader.load_raw_data(path=str(temp_file))
        
        assert df_loaded.shape == (3, 15)
        assert list(df_loaded.columns) == list(df_temp.columns)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
