"""
Simplified Model Training Script
Trains models with minimal dependencies
"""

import sys
import os
from pathlib import Path
import shutil
import logging

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main training function."""
    try:
        # Import heavy dependencies
        import numpy as np
        import pandas as pd
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import mean_squared_error, r2_score
        import pickle
        import json
        
        logger.info("=" * 80)
        logger.info("MODEL TRAINING PIPELINE")
        logger.info("=" * 80)
        
        # Define paths
        project_root = Path(__file__).parent
        data_dir = project_root / "data" / "raw"
        models_dir = project_root / "models" / "trained_models"
        artifacts_dir = project_root / "models" / "model_artifacts"
        
        # Create directories
        data_dir.mkdir(parents=True, exist_ok=True)
        models_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Copy data if not present
        logger.info("\n[1/5] Checking data...")
        data_file = data_dir / "ai_job_replacement_2020_2026.csv"
        
        if not data_file.exists():
            source_file = project_root.parent / "ai_job_replacement_2020_2026.csv"
            if source_file.exists():
                logger.info(f"Copying data from {source_file}")
                shutil.copy(source_file, data_file)
                logger.info("✓ Data copied")
            else:
                raise FileNotFoundError(f"Data not found at {source_file}")
        
        # Step 2: Load data
        logger.info("\n[2/5] Loading data...")
        df = pd.read_csv(data_file)
        logger.info(f"✓ Loaded {len(df)} rows x {len(df.columns)} columns")
        logger.info(f"Columns: {list(df.columns)[:5]}... (showing first 5)")
        
        # Step 3: Prepare data
        logger.info("\n[3/5] Preparing data...")
        
        # Handle missing values
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        logger.info("✓ Handled missing values")
        
        # Encode categorical features
        target_col = 'ai_replacement_score'
        id_col = 'job_id'
        
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        for col in categorical_cols:
            df[col] = pd.factorize(df[col])[0]
        
        logger.info(f"✓ Encoded {len(categorical_cols)} categorical features")
        
        # Separate features and target
        X = df.drop([target_col, id_col], axis=1, errors='ignore')
        y = df[target_col]
        
        logger.info(f"✓ Features: {X.shape[0]} x {X.shape[1]}")
        logger.info(f"✓ Target: {y.shape[0]}")
        
        # Step 4: Train-test split and scaling
        logger.info("\n[4/5] Splitting and scaling...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        logger.info(f"✓ Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
        
        # Step 5: Train models
        logger.info("\n[5/5] Training models...")
        
        results = {}
        best_model = None
        best_score = -np.inf
        best_name = None
        
        # Random Forest
        logger.info("  • Training RandomForest...")
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train_scaled, y_train)
        rf_r2 = rf_model.score(X_test_scaled, y_test)
        rf_rmse = np.sqrt(mean_squared_error(y_test, rf_model.predict(X_test_scaled)))
        
        results['RandomForest'] = {'r2': rf_r2, 'rmse': rf_rmse}
        logger.info(f"    R² = {rf_r2:.4f}, RMSE = {rf_rmse:.4f}")
        
        if rf_r2 > best_score:
            best_score = rf_r2
            best_model = rf_model
            best_name = 'RandomForest'
        
        # Gradient Boosting
        logger.info("  • Training GradientBoosting...")
        gb_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        gb_model.fit(X_train_scaled, y_train)
        gb_r2 = gb_model.score(X_test_scaled, y_test)
        gb_rmse = np.sqrt(mean_squared_error(y_test, gb_model.predict(X_test_scaled)))
        
        results['GradientBoosting'] = {'r2': gb_r2, 'rmse': gb_rmse}
        logger.info(f"    R² = {gb_r2:.4f}, RMSE = {gb_rmse:.4f}")
        
        if gb_r2 > best_score:
            best_score = gb_r2
            best_model = gb_model
            best_name = 'GradientBoosting'
        
        # Save artifacts
        logger.info("\n" + "=" * 80)
        logger.info("SAVING ARTIFACTS")
        logger.info("=" * 80)
        
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
        features_path = artifacts_dir / "feature_names.json"
        with open(features_path, 'w') as f:
            json.dump(X.columns.tolist(), f, indent=2)
        logger.info(f"✓ Features saved: {features_path}")
        
        # Save metadata
        metadata = {
            'model_type': best_name,
            'best_r2': float(best_score),
            'best_rmse': float(results[best_name]['rmse']),
            'n_features': X.shape[1],
            'n_samples': X.shape[0],
            'training_date': pd.Timestamp.now().isoformat(),
            'all_results': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()}
        }
        
        metadata_path = project_root / "models" / "model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"✓ Metadata saved: {metadata_path}")
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING COMPLETE ✅")
        logger.info("=" * 80)
        logger.info(f"\nBest Model: {best_name}")
        logger.info(f"Test R²: {best_score:.4f}")
        logger.info(f"Test RMSE: {results[best_name]['rmse']:.4f}")
        logger.info(f"\nAll Results:")
        for name, metrics in results.items():
            logger.info(f"  {name}: R² = {metrics['r2']:.4f}, RMSE = {metrics['rmse']:.4f}")
        
        logger.info(f"\nModel artifacts saved to: {models_dir.parent}")
        logger.info("=" * 80)
        
        return 0
    
    except Exception as e:
        logger.error(f"ERROR: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
