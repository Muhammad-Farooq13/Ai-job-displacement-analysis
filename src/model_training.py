"""
Model Training Module

Handles model training, hyperparameter tuning, cross-validation, and model persistence.
Implements multiple algorithms with comprehensive evaluation.
"""

import logging
from typing import Dict, Tuple, Optional, Any
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold, cross_validate
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.config import Config

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Trains and evaluates machine learning models."""
    
    def __init__(self, model_type: str = Config.MODEL_TYPE):
        """
        Initialize model trainer.
        
        Args:
            model_type: Type of model to train
        """
        self.model_type = model_type
        self.model = None
        self.cv_results = None
        self.training_history = {}
        self.feature_importance = None

    def train_baseline_model(self, X_train, y_train):
        """Train a baseline RandomForestClassifier."""
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(X_train, y_train)
        return clf
    
    def build_model(self, params: Optional[Dict] = None) -> Any:
        """
        Build model with specified parameters.
        
        Args:
            params: Model hyperparameters. Uses Config defaults if None.
            
        Returns:
            Fitted model object
        """
        if params is None:
            params = Config.get_params(self.model_type)
        
        logger.info(f"Building {self.model_type} model with params: {params}")
        
        if self.model_type == "linear":
            self.model = LinearRegression(**params)
        elif self.model_type == "ridge":
            self.model = Ridge(**params)
        elif self.model_type == "lasso":
            self.model = Lasso(**params)
        elif self.model_type == "random_forest":
            self.model = RandomForestRegressor(**params)
        elif self.model_type == "xgboost":
            self.model = xgb.XGBRegressor(**params)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return self.model
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, float]:
        """
        Train model on training data.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Dict: Training metrics
        """
        if self.model is None:
            self.build_model()
        
        logger.info(f"Training {self.model_type} model...")
        self.model.fit(X_train, y_train)
        
        # Get training predictions
        y_pred = self.model.predict(X_train)
        
        # Calculate metrics
        metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred)),
            'train_mae': mean_absolute_error(y_train, y_pred),
            'train_r2': r2_score(y_train, y_pred),
            'train_mape': self._calculate_mape(y_train, y_pred)
        }
        
        logger.info(f"✓ Training complete. RMSE: {metrics['train_rmse']:.4f}, R²: {metrics['train_r2']:.4f}")
        self.training_history.update(metrics)
        
        return metrics
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dict: Test metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        y_pred = self.model.predict(X_test)
        
        metrics = {
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'test_mae': mean_absolute_error(y_test, y_pred),
            'test_r2': r2_score(y_test, y_pred),
            'test_mape': self._calculate_mape(y_test, y_pred)
        }
        
        logger.info(f"✓ Evaluation complete. RMSE: {metrics['test_rmse']:.4f}, R²: {metrics['test_r2']:.4f}")
        
        return metrics
    
    def cross_validate(
        self, X: pd.DataFrame, y: pd.Series, n_splits: int = Config.N_SPLITS
    ) -> Dict[str, np.ndarray]:
        """
        Perform k-fold cross-validation.
        
        Args:
            X: Feature matrix
            y: Target variable
            n_splits: Number of folds
            
        Returns:
            Dict: Cross-validation results
        """
        logger.info(f"Performing {n_splits}-fold cross-validation...")
        
        if self.model is None:
            self.build_model()
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=Config.RANDOM_STATE)
        
        scoring = {
            'neg_mse': 'neg_mean_squared_error',
            'neg_mae': 'neg_mean_absolute_error',
            'r2': 'r2'
        }
        
        cv_results = cross_validate(self.model, X, y, cv=kf, scoring=scoring, return_train_score=True)
        
        # Convert negative metrics back to positive and compute RMSE
        test_rmse = np.sqrt(-cv_results['test_neg_mse'])
        train_rmse = np.sqrt(-cv_results['train_neg_mse'])
        
        results_summary = {
            'train_rmse_mean': train_rmse.mean(),
            'train_rmse_std': train_rmse.std(),
            'test_rmse_mean': test_rmse.mean(),
            'test_rmse_std': test_rmse.std(),
            'test_mae_mean': -cv_results['test_neg_mae'].mean(),
            'test_r2_mean': cv_results['test_r2'].mean(),
            'test_r2_std': cv_results['test_r2'].std()
        }
        
        for key, val in results_summary.items():
            logger.info(f"  {key}: {val:.4f}")
        
        self.cv_results = results_summary
        return results_summary
    
    def get_feature_importance(self, feature_names: list) -> pd.DataFrame:
        """
        Extract feature importance from model.
        
        Args:
            feature_names: List of feature names
            
        Returns:
            pd.DataFrame: Feature importance ranking
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        importances = None
        
        if self.model_type in ["random_forest", "xgboost"]:
            importances = self.model.feature_importances_
        elif self.model_type in ["ridge", "lasso"]:
            importances = np.abs(self.model.coef_)
        elif self.model_type == "linear":
            importances = np.abs(self.model.coef_)
        
        if importances is not None:
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            self.feature_importance = importance_df
            return importance_df
        
        return pd.DataFrame()
    
    def save_model(self, output_path: Optional[Path] = None) -> None:
        """
        Save trained model to disk.
        
        Args:
            output_path: Path to save model
        """
        if self.model is None:
            raise ValueError("No model to save. Call train() first.")
        
        output_path = output_path or Config.MODELS_PATH / "trained_models" / f"{self.model_type}_model.pkl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "wb") as f:
            pickle.dump(self.model, f)
        
        logger.info(f"✓ Model saved to {output_path}")
    
    def load_model(self, model_path: Path) -> None:
        """
        Load trained model from disk.
        
        Args:
            model_path: Path to saved model
        """
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        
        logger.info(f"✓ Model loaded from {model_path}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Feature matrix
            
        Returns:
            np.ndarray: Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() or load_model() first.")
        
        return self.model.predict(X)
    
    def predict_with_uncertainty(
        self, X: pd.DataFrame, uncertainty_method: str = "std"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty estimates.
        
        Args:
            X: Feature matrix
            uncertainty_method: 'std' for random forest OOB errors, 'residual' for others
            
        Returns:
            Tuple: (predictions, uncertainty)
        """
        predictions = self.predict(X)
        
        if self.model_type == "random_forest" and uncertainty_method == "std":
            # Use ensemble uncertainty
            individual_predictions = np.array([
                tree.predict(X) for tree in self.model.estimators_
            ])
            uncertainty = individual_predictions.std(axis=0)
        else:
            # Use standard residual error estimate
            uncertainty = np.std(self.training_history.get('residuals', 0))
        
        return predictions, uncertainty
    
    @staticmethod
    def _calculate_mape(y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error."""
        # Avoid division by zero
        mask = y_true != 0
        if mask.sum() == 0:
            return 0.0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


class ModelComparison:
    """Compare multiple models."""
    
    @staticmethod
    def train_and_compare(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        models: list = None
    ) -> pd.DataFrame:
        """
        Train multiple models and compare performance.
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training target
            y_test: Test target
            models: List of model types to compare
            
        Returns:
            pd.DataFrame: Comparison results
        """
        if models is None:
            models = ["linear", "ridge", "random_forest", "xgboost"]
        
        results = []
        
        for model_type in models:
            logger.info(f"\nTraining {model_type}...")
            
            try:
                trainer = ModelTrainer(model_type)
                trainer.build_model()
                train_metrics = trainer.train(X_train, y_train)
                test_metrics = trainer.evaluate(X_test, y_test)
                
                result = {
                    'model': model_type,
                    **train_metrics,
                    **test_metrics
                }
                results.append(result)
            except Exception as e:
                logger.error(f"Error training {model_type}: {e}")
        
        comparison_df = pd.DataFrame(results)
        logger.info("\n" + "="*80)
        logger.info("MODEL COMPARISON")
        logger.info("="*80)
        logger.info(comparison_df.to_string())
        
        return comparison_df


if __name__ == "__main__":
    logger.info("Model training module loaded successfully")
