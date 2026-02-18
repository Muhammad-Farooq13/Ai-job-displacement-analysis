"""
Model Evaluation Module

Comprehensive model evaluation, visualization, error analysis, and metrics reporting.
"""

import logging
from typing import Dict, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path

from src.config import Config

logger = logging.getLogger(__name__)
sns.set_style("whitegrid")


class ModelEvaluator:
    """Evaluates model performance with comprehensive metrics and visualizations."""
    
    def __init__(self, output_dir: Path = None):
        """
        Initialize evaluator.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir or Config.REPORTS_PATH / "figures"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_history = {}
    
    def calculate_metrics(
        self, y_true: pd.Series, y_pred: np.ndarray, dataset_name: str = "test"
    ) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            dataset_name: Name of dataset (for logging)
            
        Returns:
            Dict: Dictionary of metrics
        """
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # MAPE
        mask = y_true != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = 0.0
        
        # Residuals
        residuals = y_true - y_pred
        
        metrics = {
            'rmse': rmse,
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'mape': mape,
            'residuals': residuals
        }
        
        self.metrics_history[dataset_name] = metrics
        
        logger.info(f"\n{dataset_name.upper()} METRICS:")
        logger.info(f"  RMSE: {rmse:.4f}")
        logger.info(f"  MAE:  {mae:.4f}")
        logger.info(f"  R²:   {r2:.4f}")
        logger.info(f"  MAPE: {mape:.4f}%")
        
        return metrics
    
    def plot_predictions(
        self, y_true: pd.Series, y_pred: np.ndarray, dataset_name: str = "test"
    ) -> None:
        """
        Plot actual vs predicted values.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            dataset_name: Name of dataset
        """
        plt.figure(figsize=(10, 6))
        
        plt.scatter(y_true, y_pred, alpha=0.6, s=30)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        plt.xlabel('Actual Value', fontsize=12)
        plt.ylabel('Predicted Value', fontsize=12)
        plt.title(f'Actual vs Predicted - {dataset_name.upper()}', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        output_file = self.output_dir / f"actual_vs_predicted_{dataset_name}.png"
        plt.savefig(output_file, dpi=Config.FIGURE_DPI, bbox_inches='tight')
        logger.info(f"✓ Saved plot: {output_file.name}")
        plt.close()
    
    def plot_residuals(
        self, y_true: pd.Series, y_pred: np.ndarray, dataset_name: str = "test"
    ) -> None:
        """
        Plot residuals analysis.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            dataset_name: Name of dataset
        """
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Residuals vs Predicted
        axes[0, 0].scatter(y_pred, residuals, alpha=0.6, s=30)
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_xlabel('Predicted Value')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Predicted')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Residuals histogram
        axes[0, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[0, 1].set_xlabel('Residuals')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Residuals')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Absolute error distribution
        abs_errors = np.abs(residuals)
        axes[1, 1].hist(abs_errors, bins=30, edgecolor='black', alpha=0.7)
        axes[1, 1].set_xlabel('Absolute Error')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Absolute Error Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Residuals Analysis - {dataset_name.upper()}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_file = self.output_dir / f"residuals_{dataset_name}.png"
        plt.savefig(output_file, dpi=Config.FIGURE_DPI, bbox_inches='tight')
        logger.info(f"✓ Saved plot: {output_file.name}")
        plt.close()
    
    def plot_feature_importance(
        self, feature_importance_df: pd.DataFrame, top_n: int = 15
    ) -> None:
        """
        Plot feature importance ranking.
        
        Args:
            feature_importance_df: DataFrame with feature and importance columns
            top_n: Number of top features to display
        """
        if feature_importance_df is None or feature_importance_df.empty:
            logger.warning("Feature importance data not available")
            return
        
        top_features = feature_importance_df.head(top_n).copy()
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(top_features)), top_features['importance'].values)
        plt.yticks(range(len(top_features)), top_features['feature'].values)
        plt.xlabel('Importance Score', fontsize=12)
        plt.title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        
        output_file = self.output_dir / "feature_importance.png"
        plt.savefig(output_file, dpi=Config.FIGURE_DPI, bbox_inches='tight')
        logger.info(f"✓ Saved plot: {output_file.name}")
        plt.close()
    
    def plot_error_distribution(
        self, y_true: pd.Series, y_pred: np.ndarray, bins: int = 50
    ) -> None:
        """
        Plot error distribution across prediction ranges.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            bins: Number of bins for histogram
        """
        errors = np.abs(y_true - y_pred)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Error histogram
        axes[0].hist(errors, bins=bins, edgecolor='black', alpha=0.7)
        axes[0].axvline(errors.mean(), color='r', linestyle='--', label=f'Mean: {errors.mean():.2f}')
        axes[0].axvline(np.median(errors), color='g', linestyle='--', label=f'Median: {np.median(errors):.2f}')
        axes[0].set_xlabel('Absolute Error')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Error Distribution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Error by prediction range
        sorted_indices = np.argsort(y_pred)
        window_size = len(y_true) // 10
        
        prediction_ranges = []
        error_means = []
        
        for i in range(0, len(y_true), window_size):
            end_idx = min(i + window_size, len(y_true))
            window_indices = sorted_indices[i:end_idx]
            prediction_ranges.append(y_pred[window_indices].mean())
            error_means.append(errors[window_indices].mean())
        
        axes[1].plot(prediction_ranges, error_means, 'o-', linewidth=2, markersize=8)
        axes[1].set_xlabel('Prediction Value')
        axes[1].set_ylabel('Mean Absolute Error')
        axes[1].set_title('Error by Prediction Range')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_file = self.output_dir / "error_distribution.png"
        plt.savefig(output_file, dpi=Config.FIGURE_DPI, bbox_inches='tight')
        logger.info(f"✓ Saved plot: {output_file.name}")
        plt.close()
    
    def generate_evaluation_report(
        self, metrics_dict: Dict[str, Dict[str, float]], model_name: str = "Model"
    ) -> str:
        """
        Generate comprehensive evaluation report.
        
        Args:
            metrics_dict: Dictionary with train/test metrics
            model_name: Name of the model
            
        Returns:
            str: Formatted report
        """
        report = f"\n{'='*80}\nEVALUATION REPORT: {model_name}\n{'='*80}\n"
        
        for dataset_name, metrics in metrics_dict.items():
            report += f"\n{dataset_name.upper()} PERFORMANCE:\n"
            report += "-" * 40 + "\n"
            
            if 'rmse' in metrics:
                report += f"RMSE (Root Mean Squared Error): {metrics['rmse']:.4f}\n"
            if 'mae' in metrics:
                report += f"MAE  (Mean Absolute Error):     {metrics['mae']:.4f}\n"
            if 'r2' in metrics:
                report += f"R²   (Coefficient of Determination): {metrics['r2']:.4f}\n"
            if 'mape' in metrics:
                report += f"MAPE (Mean Absolute Percentage Error): {metrics['mape']:.4f}%\n"
        
        report += "\n" + "="*80 + "\n"
        
        logger.info(report)
        return report
    
    def save_evaluation_report(self, report: str, filename: str = "evaluation_report.txt") -> None:
        """
        Save evaluation report to file.
        
        Args:
            report: Report text
            filename: Output filename
        """
        output_file = Config.REPORTS_PATH / filename
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            f.write(report)
        
        logger.info(f"✓ Saved evaluation report: {output_file}")


if __name__ == "__main__":
    logger.info("Evaluation module loaded successfully")
