"""
AI Job Displacement Analysis: Complete Data Science Project

A professional, production-ready machine learning project investigating
AI-driven job displacement trends across industries and regions.

Author: Muhammad Farooq
Email: mfarooqshafee333@gmail.com
GitHub: Muhammad-Farooq-13
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Muhammad Farooq"

from src.config import Config, setup_logging
from src.data_loader import DataLoader, load_processed_data
from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.model_training import ModelTrainer, ModelComparison
from src.evaluation import ModelEvaluator
from src.utils import DataSummaryPrinter, PlotingUtils, MetricsCalculator

__all__ = [
    "Config",
    "setup_logging",
    "DataLoader",
    "load_processed_data",
    "DataPreprocessor",
    "FeatureEngineer",
    "ModelTrainer",
    "ModelComparison",
    "ModelEvaluator",
    "DataSummaryPrinter",
    "PlotingUtils",
    "MetricsCalculator",
]
