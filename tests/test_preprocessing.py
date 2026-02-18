"""
Unit tests for preprocessing module.

Tests DataPreprocessor class for missing value handling, encoding, scaling,
and train-test splitting.
"""

import pytest
import pandas as pd
import numpy as np
from src.preprocessing import DataPreprocessor


class TestDataPreprocessor:
    """Test data preprocessing functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame({
            'job_id': [1, 2, 3, 4, 5],
            'job_role': ['Engineer', 'Teacher', 'Engineer', 'Analyst', 'Teacher'],
            'industry': ['Tech', 'Education', 'Tech', 'Finance', 'Education'],
            'country': ['USA', 'USA', 'Canada', 'USA', 'Canada'],
            'year': [2020, 2021, 2020, 2021, 2022],
            'automation_risk_percent': [0.5, 0.3, 0.6, 0.7, 0.2],
            'ai_replacement_score': [50, 30, 60, 70, 20],
            'skill_gap_index': [0.6, 0.4, 0.65, 0.5, 0.3],
            'salary_before_usd': [100000, 60000, 105000, 80000, 55000],
            'salary_after_usd': [105000, 62000, 100000, 75000, 57000],
            'salary_change_percent': [5, 3.3, -4.8, -6.3, 3.6],
            'skill_demand_growth_percent': [10, 5, 12, 15, 3],
            'remote_feasibility_score': [0.8, 0.3, 0.85, 0.6, 0.2],
            'ai_adoption_level': [0.6, 0.2, 0.65, 0.5, 0.15],
            'education_requirement_level': [3, 2, 3, 3, 1]
        })
    
    def test_preprocessor_initialization(self):
        """Test preprocessor initialization."""
        preprocessor = DataPreprocessor()
        assert preprocessor.scaler is None
        assert preprocessor.label_encoders == {}
    
    def test_handle_missing_values(self, sample_data):
        """Test missing value handling."""
        df = sample_data.copy()
        df.loc[0, 'skill_gap_index'] = np.nan
        df.loc[1, 'automation_risk_percent'] = np.nan
        
        preprocessor = DataPreprocessor()
        df_cleaned = preprocessor.handle_missing_values(df)
        
        assert df_cleaned.isnull().sum().sum() == 0
    
    def test_train_test_split(self, sample_data):
        """Test train-test split functionality."""
        preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.split_data(
            sample_data, 
            test_size=0.4, 
            random_state=42
        )
        
        assert len(X_train) == 3
        assert len(X_test) == 2
        assert len(y_train) == 3
        assert len(y_test) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
