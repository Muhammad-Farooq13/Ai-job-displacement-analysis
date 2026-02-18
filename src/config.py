"""
Configuration Management Module

This module handles all configuration settings for the data science pipeline,
including paths, model parameters, data processing settings, and logging configuration.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import yaml

# Project Root
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW_PATH = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"
MODELS_PATH = PROJECT_ROOT / "models"
REPORTS_PATH = PROJECT_ROOT / "reports"
NOTEBOOKS_PATH = PROJECT_ROOT / "notebooks"

# Create directories if they don't exist
for path in [DATA_RAW_PATH, DATA_PROCESSED_PATH, MODELS_PATH, REPORTS_PATH]:
    path.mkdir(parents=True, exist_ok=True)


class Config:
    """Base configuration class with all settings."""
    
    # ========== Data Settings ==========
    RAW_DATA_FILE = DATA_RAW_PATH / "ai_job_replacement_2020_2026.csv"
    PROCESSED_TRAIN_FILE = DATA_PROCESSED_PATH / "train.parquet"
    PROCESSED_TEST_FILE = DATA_PROCESSED_PATH / "test.parquet"
    
    # Data processing
    TEST_SIZE = 0.30
    RANDOM_STATE = 42
    TRAIN_YEARS = [2020, 2021, 2022, 2023, 2024]
    TEST_YEARS = [2025, 2026]
    
    # ========== Feature Settings ==========
    TARGET_VARIABLE = "ai_replacement_score"
    ID_COLUMN = "job_id"
    
    CATEGORICAL_FEATURES = [
        "job_role",
        "industry",
        "country",
        "education_requirement_level"
    ]
    
    NUMERIC_FEATURES = [
        "automation_risk_percent",
        "skill_gap_index",
        "salary_before_usd",
        "salary_after_usd",
        "salary_change_percent",
        "skill_demand_growth_percent",
        "remote_feasibility_score",
        "ai_adoption_level",
        "year"
    ]
    
    # Feature engineering parameters
    SCALING_METHOD = "standard"  # 'standard' or 'minmax'
    ENCODING_METHOD = "onehot"   # 'onehot' or 'target'
    
    # ========== Model Settings ==========
    MODEL_TYPE = "xgboost"  # Options: 'linear', 'ridge', 'lasso', 'random_forest', 'xgboost'
    
    # Model hyperparameters
    MODEL_PARAMS = {
        "xgboost": {
            "max_depth": 6,
            "learning_rate": 0.05,
            "n_estimators": 200,
            "subsample": 0.8,
            "colsample_bytree": 0.9,
            "random_state": 42,
            "verbosity": 1,
            "tree_method": "hist"
        },
        "random_forest": {
            "n_estimators": 200,
            "max_depth": 15,
            "min_samples_split": 5,
            "random_state": 42,
            "n_jobs": -1
        },
        "ridge": {
            "alpha": 0.1,
            "random_state": 42,
            "max_iter": 1000
        }
    }
    
    # Cross-validation
    N_SPLITS = 5
    CV_STRATEGY = "stratified"  # 'kfold' or 'stratified'
    
    # ========== Training Settings ==========
    EARLY_STOPPING_ROUNDS = 20
    EARLY_STOPPING_METRIC = "rmse"
    BATCH_SIZE = 32
    EPOCHS = 100
    
    # ========== Evaluation Settings ==========
    EVALUATION_METRICS = ["rmse", "mae", "r2", "mape"]
    CONFIDENCE_LEVEL = 0.95
    PERCENTILES = [5, 25, 50, 75, 95]
    
    # ========== Logging Settings ==========
    LOG_LEVEL = logging.INFO
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE = PROJECT_ROOT / "logs" / "project.log"
    
    # ========== Output Settings ==========
    SAVE_FIGURES = True
    FIGURE_DPI = 300
    FIGURE_FORMAT = "png"
    
    @classmethod
    def get_params(cls, model_type: str) -> Dict[str, Any]:
        """Get model parameters for specified model type."""
        return cls.MODEL_PARAMS.get(model_type, {})
    
    @classmethod
    def get_all_features(cls) -> list:
        """Get all feature names."""
        return cls.CATEGORICAL_FEATURES + cls.NUMERIC_FEATURES
    
    @classmethod
    def load_from_yaml(cls, yaml_file: Path) -> "Config":
        """Load configuration from YAML file."""
        if not yaml_file.exists():
            logging.warning(f"Config file {yaml_file} not found. Using defaults.")
            return cls
        
        try:
            with open(yaml_file, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            # Update class attributes
            for key, value in config_dict.items():
                if hasattr(cls, key):
                    setattr(cls, key, value)
            logging.info(f"Configuration loaded from {yaml_file}")
        except Exception as e:
            logging.error(f"Error loading config from {yaml_file}: {e}")
        
        return cls


def setup_logging(log_file: Optional[Path] = None, log_level: int = logging.INFO) -> None:
    """Configure logging for the entire project."""
    if log_file is None:
        log_file = Config.LOG_FILE
    
    # Create log directory
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(Config.LOG_FORMAT))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(Config.LOG_FORMAT))
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logging.info("Logging configured successfully")


# Initialize logging on import
setup_logging()


if __name__ == "__main__":
    # Test configuration
    print("Project Root:", Config.PROJECT_ROOT)
    print("Model Type:", Config.MODEL_TYPE)
    print("Target Variable:", Config.TARGET_VARIABLE)
    print("Number of Numeric Features:", len(Config.NUMERIC_FEATURES))
    print("Number of Categorical Features:", len(Config.CATEGORICAL_FEATURES))
