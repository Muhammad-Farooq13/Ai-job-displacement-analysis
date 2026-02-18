"""
Unit tests for model training module.

Tests ModelTrainer class for model training, validation, and hyperparameter tuning.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from src.model_training import ModelTrainer


class TestModelTrainer:
    """Test model training functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        X_train = np.random.randn(100, 10)
        y_train = np.random.randint(0, 2, 100)
        return X_train, y_train
    
    def test_trainer_initialization(self):
        """Test model trainer initialization."""
        trainer = ModelTrainer()
        assert trainer is not None
    
    def test_train_model(self, sample_data):
        """Test basic model training."""
        X_train, y_train = sample_data
        trainer = ModelTrainer()
        
        model = trainer.train_baseline_model(X_train, y_train)
        
        assert model is not None
        assert hasattr(model, 'predict')
    
    def test_model_prediction(self, sample_data):
        """Test model prediction capability."""
        X_train, y_train = sample_data
        trainer = ModelTrainer()
        
        model = trainer.train_baseline_model(X_train, y_train)
        predictions = model.predict(X_train[:5])
        
        assert len(predictions) == 5
        assert all(pred in [0, 1] for pred in predictions)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
