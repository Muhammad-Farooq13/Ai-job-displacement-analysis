#!/usr/bin/env python
"""
End-to-End Model Training Pipeline

Executes the complete training workflow:
1. Load raw data
2. Preprocess and validate
3. Engineer features
4. Train models
5. Evaluate and save artifacts
"""

import sys
import logging
from pathlib import Path
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Add src to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config, setup_logging
from src.data_loader import DataLoader, DataValidator
from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.evaluation import ModelEvaluator

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


def main():
    """Execute complete training pipeline."""
    logger.info("=" * 80)
    logger.info("STARTING MODEL TRAINING PIPELINE")
    logger.info("=" * 80)
    
    try:
        # Step 1: Load Data
        logger.info("\n[Step 1/6] Loading raw data...")
        loader = DataLoader()
        
        # Try to load from data/raw, if not found, copy from archive
        raw_file = Config.RAW_DATA_FILE
        if not raw_file.exists():
            archive_file = PROJECT_ROOT.parent / "ai_job_replacement_2020_2026.csv"
            if archive_file.exists():
                logger.info(f"Copying data from {archive_file} to {raw_file}")
                import shutil
                raw_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(archive_file, raw_file)
                logger.info("✓ Data copied successfully")
            else:
                raise FileNotFoundError(f"Data file not found at {archive_file}")
        
        df = loader.load_raw_data()
        logger.info(f"✓ Loaded {df.shape[0]} records with {df.shape[1]} columns")
        
        # Step 2: Validate Schema
        logger.info("\n[Step 2/6] Validating data schema...")
        DataValidator.validate_schema(df)
        logger.info("✓ Schema validation passed")
        
        # Step 3: Preprocess Data
        logger.info("\n[Step 3/6] Preprocessing data...")
        preprocessor = DataPreprocessor()
        
        # Handle missing values
        df = preprocessor.handle_missing_values(df)
        logger.info(f"✓ Missing values handled")
        
        # Separate features and target
        X = df.drop([Config.TARGET_VARIABLE, Config.ID_COLUMN], axis=1, errors='ignore')
        y = df[Config.TARGET_VARIABLE]
        
        logger.info(f"✓ Features shape: {X.shape}")
        logger.info(f"✓ Target shape: {y.shape}")
        
        # Step 4: Feature Engineering
        logger.info("\n[Step 4/6] Engineering features...")
        engineer = FeatureEngineer()
        
        # Create temporal features
        if 'year' in X.columns:
            X = engineer.create_temporal_features(X)
        
        # Create domain features
        X = engineer.create_domain_features(X)
        logger.info(f"✓ Features engineered. New shape: {X.shape}")
        
        # Encode categorical variables
        categorical_cols = [col for col in X.columns if X[col].dtype == 'object']
        for col in categorical_cols:
            X[col] = pd.factorize(X[col])[0]
        logger.info(f"✓ Categorical encoding completed")
        
        # Step 5: Train-Test Split & Scaling
        logger.info("\n[Step 5/6] Splitting and scaling data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        logger.info(f"✓ Train set: {X_train.shape[0]} samples")
        logger.info(f"✓ Test set: {X_test.shape[0]} samples")
        
        # Step 6: Train Models
        logger.info("\n[Step 6/6] Training models...")
        
        models = {
            'RandomForest': RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=Config.RANDOM_STATE,
                n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=7,
                random_state=Config.RANDOM_STATE
            )
        }
        
        best_model = None
        best_score = -np.inf
        results = {}
        
        for model_name, model in models.items():
            logger.info(f"\n  Training {model_name}...")
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)
            
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            
            results[model_name] = {
                'train_r2': float(train_r2),
                'test_r2': float(test_r2),
                'train_rmse': float(train_rmse),
                'test_rmse': float(test_rmse)
            }
            
            logger.info(f"  ✓ {model_name}: Train R²={train_r2:.4f}, Test R²={test_r2:.4f}")
            logger.info(f"                Train RMSE={train_rmse:.4f}, Test RMSE={test_rmse:.4f}")
            
            if test_r2 > best_score:
                best_score = test_r2
                best_model = model
                best_model_name = model_name
        
        # Save Best Model
        logger.info("\n" + "=" * 80)
        logger.info(f"SAVING ARTIFACTS - Best Model: {best_model_name} (R² = {best_score:.4f})")
        logger.info("=" * 80)
        
        models_dir = Config.MODELS_PATH / "trained_models"
        artifacts_dir = Config.MODELS_PATH / "model_artifacts"
        models_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = models_dir / "ai_replacement_predictor.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(best_model, f)
        logger.info(f"✓ Model saved: {model_path}")
        
        # Save scaler
        scaler_path = artifacts_dir / "feature_scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        logger.info(f"✓ Scaler saved: {scaler_path}")
        
        # Save feature names
        feature_names_path = artifacts_dir / "feature_names.json"
        with open(feature_names_path, 'w') as f:
            json.dump(X.columns.tolist(), f, indent=2)
        logger.info(f"✓ Feature names saved: {feature_names_path}")
        
        # Save model metadata
        metadata = {
            'model_type': best_model_name,
            'best_r2_score': float(best_score),
            'train_test_split': Config.TEST_SIZE,
            'random_state': Config.RANDOM_STATE,
            'n_features': X.shape[1],
            'n_samples': X.shape[0],
            'target_variable': Config.TARGET_VARIABLE,
            'model_results': results,
            'training_date': pd.Timestamp.now().isoformat()
        }
        
        metadata_path = Config.MODELS_PATH / "model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"✓ Model metadata saved: {metadata_path}")
        
        # Summary Report
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING COMPLETE - SUMMARY")
        logger.info("=" * 80)
        logger.info(f"\nBest Model: {best_model_name}")
        logger.info(f"Test R² Score: {best_score:.4f}")
        logger.info(f"Test RMSE: {results[best_model_name]['test_rmse']:.4f}")
        logger.info(f"\nAll Model Results:")
        for name, metrics in results.items():
            logger.info(f"  {name}:")
            logger.info(f"    Train R²: {metrics['train_r2']:.4f}, Test R²: {metrics['test_r2']:.4f}")
            logger.info(f"    Train RMSE: {metrics['train_rmse']:.4f}, Test RMSE: {metrics['test_rmse']:.4f}")
        
        logger.info(f"\nArtifacts saved to: {Config.MODELS_PATH}")
        logger.info("\n✅ Model training pipeline completed successfully!")
        logger.info("=" * 80)
        
        return 0
    
    except Exception as e:
        logger.error(f"\n❌ ERROR: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
