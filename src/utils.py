"""
Utilities Module

Helper functions for logging, plotting, metrics, and general utilities.
"""

import logging
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logger = logging.getLogger(__name__)


class DataSummaryPrinter:
    """Pretty-print data and analysis summaries."""
    
    @staticmethod
    def print_dataframe_summary(df: pd.DataFrame) -> None:
        """Print comprehensive DataFrame summary."""
        print("\n" + "="*80)
        print("DATAFRAME SUMMARY")
        print("="*80)
        print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"\nData Types:\n{df.dtypes}")
        print(f"\nMissing Values:\n{df.isnull().sum()}")
        print(f"\nBasic Statistics:\n{df.describe()}")
        print("="*80 + "\n")
    
    @staticmethod
    def print_model_comparison(comparison_df: pd.DataFrame) -> None:
        """Print formatted model comparison."""
        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)
        print(comparison_df.to_string(index=False))
        print("="*80 + "\n")


class PlotingUtils:
    """Utilities for creating visualizations."""
    
    @staticmethod
    def plot_distribution(series: pd.Series, title: str = "", output_path: Path = None) -> None:
        """Plot distribution of a series."""
        plt.figure(figsize=(10, 6))
        plt.hist(series, bins=30, edgecolor='black', alpha=0.7)
        plt.xlabel('Value', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"✓ Saved plot: {output_path}")
        
        plt.show()
    
    @staticmethod
    def plot_correlation_matrix(df: pd.DataFrame, output_path: Path = None) -> None:
        """Plot correlation matrix heatmap."""
        numeric_df = df.select_dtypes(include=[np.number])
        
        plt.figure(figsize=(12, 10))
        correlation_matrix = numeric_df.corr()
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, cbar_kws={"shrink": 0.8})
        plt.title('Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"✓ Saved plot: {output_path}")
        
        plt.show()
    
    @staticmethod
    def plot_categorical_distribution(df: pd.DataFrame, column: str, 
                                     output_path: Path = None, top_n: int = 10) -> None:
        """Plot top N categories."""
        plt.figure(figsize=(12, 6))
        
        top_categories = df[column].value_counts().head(top_n)
        plt.bar(range(len(top_categories)), top_categories.values)
        plt.xticks(range(len(top_categories)), top_categories.index, rotation=45, ha='right')
        plt.ylabel('Count', fontsize=12)
        plt.title(f'Top {top_n} {column} Categories', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"✓ Saved plot: {output_path}")
        
        plt.show()


class MetricsCalculator:
    """Utility functions for metrics calculation."""
    
    @staticmethod
    def calculate_percentage_change(before: float, after: float) -> float:
        """Calculate percentage change."""
        if before == 0:
            return 0
        return ((after - before) / abs(before)) * 100
    
    @staticmethod
    def calculate_confidence_interval(data: np.ndarray, confidence: float = 0.95) -> tuple:
        """
        Calculate confidence interval for a dataset.
        
        Args:
            data: Array of values
            confidence: Confidence level (0-1)
            
        Returns:
            Tuple: (mean, lower_bound, upper_bound)
        """
        from scipy import stats
        
        mean = np.mean(data)
        std_error = stats.sem(data)
        margin_error = std_error * stats.t.ppf((1 + confidence) / 2, len(data) - 1)
        
        return mean, mean - margin_error, mean + margin_error
    
    @staticmethod
    def calculate_effect_size(group1: np.ndarray, group2: np.ndarray) -> float:
        """
        Calculate Cohen's d effect size.
        
        Args:
            group1: First group of values
            group2: Second group of values
            
        Returns:
            float: Cohen's d statistic
        """
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        return (np.mean(group1) - np.mean(group2)) / pooled_std


class FileUtils:
    """File and path utilities."""
    
    @staticmethod
    def ensure_directory(path: Path) -> Path:
        """Ensure directory exists."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @staticmethod
    def get_file_size(file_path: Path) -> str:
        """Get human-readable file size."""
        size_bytes = file_path.stat().st_size
        
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024
        
        return f"{size_bytes:.2f} TB"


if __name__ == "__main__":
    logger.info("Utilities module loaded successfully")
