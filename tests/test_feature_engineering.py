"""
Unit tests for feature engineering module.

Tests FeatureEngineer class for feature creation and selection.
"""

import pytest
import pandas as pd
import numpy as np
from src.feature_engineering import FeatureEngineer


class TestFeatureEngineer:
    """Test feature engineering functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame({
            'job_id': [1, 2, 3, 4, 5],
            'job_role': ['Engineer', 'Teacher', 'Engineer', 'Analyst', 'Teacher'],
            'year': [2020, 2021, 2020, 2021, 2022],
            'automation_risk_percent': [0.5, 0.3, 0.6, 0.7, 0.2],
            'ai_replacement_score': [50, 30, 60, 70, 20],
            'skill_gap_index': [0.6, 0.4, 0.65, 0.5, 0.3],
            'salary_before_usd': [100000, 60000, 105000, 80000, 55000],
            'remote_feasibility_score': [0.8, 0.3, 0.85, 0.6, 0.2],
            'ai_adoption_level': [0.6, 0.2, 0.65, 0.5, 0.15],
            'skill_demand_growth_percent': [10, 5, 12, 15, 3],
            'education_requirement_level': [3, 2, 3, 3, 1]
        })
    
    def test_feature_engineer_initialization(self):
        """Test feature engineer initialization."""
        engineer = FeatureEngineer()
        assert engineer is not None
    
    def test_create_temporal_features(self, sample_data):
        """Test temporal feature creation."""
        engineer = FeatureEngineer()
        df = sample_data.copy()
        
        df_engineered = engineer.create_temporal_features(df)
        
        assert 'time_since_2020' in df_engineered.columns
        assert df_engineered['time_since_2020'].min() == 0
    
    def test_create_domain_features(self, sample_data):
        """Test domain-specific feature creation."""
        engineer = FeatureEngineer()
        df = sample_data.copy()
        
        df_engineered = engineer.create_domain_features(df)
        
        assert 'risk_exposure' in df_engineered.columns
        assert 'education_barrier' in df_engineered.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
