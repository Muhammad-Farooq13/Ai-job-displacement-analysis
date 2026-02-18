"""
Feature Engineering Module

Handles creation of new features, feature selection, and dimensionality reduction.
Implements domain-knowledge-driven feature engineering for the job displacement problem.
"""

import logging
from typing import Tuple, List
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.preprocessing import PolynomialFeatures
import pickle
from pathlib import Path

from src.config import Config

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Creates and selects meaningful features for model training."""
    
    def __init__(self):
        """Initialize feature engineer."""
        self.polynomial_generator = None
        self.feature_selector = None
        self.selected_features = None
        self.feature_importances = None
        self.is_fitted = False
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between key variables.
        
        Args:
            df: DataFrame to enhance
            
        Returns:
            pd.DataFrame: DataFrame with interaction features
        """
        df = df.copy()
        
        # AI adoption × Skill gap interaction
        if 'ai_adoption_level' in df.columns and 'skill_gap_index' in df.columns:
            df['ai_adoption_x_skill_gap'] = (
                df['ai_adoption_level'] * df['skill_gap_index']
            )
            logger.info("  Created: ai_adoption_x_skill_gap")
        
        # Remote feasibility × Salary change interaction
        if 'remote_feasibility_score' in df.columns and 'salary_change_percent' in df.columns:
            df['remote_x_salary_impact'] = (
                df['remote_feasibility_score'] * df['salary_change_percent']
            )
            logger.info("  Created: remote_x_salary_impact")
        
        # Automation risk × Skill demand
        if 'automation_risk_percent' in df.columns and 'skill_demand_growth_percent' in df.columns:
            df['automation_x_demand'] = (
                df['automation_risk_percent'] * df['skill_demand_growth_percent']
            )
            logger.info("  Created: automation_x_demand")
        
        return df
    
    def create_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create domain-specific features based on business logic.
        
        Args:
            df: DataFrame to enhance
            
        Returns:
            pd.DataFrame: DataFrame with domain features
        """
        df = df.copy()
        
        # Job resilience score (inverse of automation risk)
        if 'automation_risk_percent' in df.columns:
            df['job_resilience_score'] = 100 - df['automation_risk_percent']
            logger.info("  Created: job_resilience_score")
        
        # Salary volatility (absolute change)
        if 'salary_change_percent' in df.columns:
            df['salary_volatility'] = df['salary_change_percent'].abs()
            logger.info("  Created: salary_volatility")
        
        # Skill demand ratio (growth relative to gap)
        if 'skill_demand_growth_percent' in df.columns and 'skill_gap_index' in df.columns:
            # Avoid division by zero
            df['skill_demand_ratio'] = df['skill_demand_growth_percent'] / (
                df['skill_gap_index'].replace(0, 1)
            )
            logger.info("  Created: skill_demand_ratio")
        
        # Salary efficiency (salary relative to AI adoption)
        if 'salary_before_usd' in df.columns and 'ai_adoption_level' in df.columns:
            df['salary_per_ai_unit'] = df['salary_before_usd'] / (
                df['ai_adoption_level'].replace(0, 1)
            )
            logger.info("  Created: salary_per_ai_unit")
        
        # Education-adjusted skill gap
        if 'skill_gap_index' in df.columns and 'education_requirement_level' in df.columns:
            df['education_adjusted_gap'] = (
                df['skill_gap_index'] / df['education_requirement_level'].replace(0, 1)
            )
            logger.info("  Created: education_adjusted_gap")
        
        return df
    
    def create_aggregate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create aggregate features at industry and country level.
        
        Args:
            df: DataFrame to enhance (must contain job_role, industry, country)
            
        Returns:
            pd.DataFrame: DataFrame with aggregate features
        """
        df = df.copy()
        
        # Industry-level aggregations
        if 'industry' in df.columns:
            industry_stats = df.groupby('industry').agg({
                'ai_adoption_level': 'mean',
                'salary_change_percent': 'mean',
                'automation_risk_percent': 'mean'
            }).rename(columns={
                'ai_adoption_level': 'industry_avg_ai_adoption',
                'salary_change_percent': 'industry_avg_salary_change',
                'automation_risk_percent': 'industry_avg_automation_risk'
            })
            
            df = df.merge(industry_stats, left_on='industry', right_index=True, how='left')
            logger.info("  Created: industry_avg_* features")
        
        # Country-level aggregations
        if 'country' in df.columns:
            country_stats = df.groupby('country').agg({
                'salary_before_usd': 'mean',
                'ai_adoption_level': 'mean'
            }).rename(columns={
                'salary_before_usd': 'country_avg_salary',
                'ai_adoption_level': 'country_avg_ai_adoption'
            })
            
            df = df.merge(country_stats, left_on='country', right_index=True, how='left')
            logger.info("  Created: country_avg_* features")
        
        return df
    
    def create_polynomial_features(self, df: pd.DataFrame, degree: int = 2, fit: bool = True) -> pd.DataFrame:
        """
        Create polynomial features.
        
        Args:
            df: DataFrame to enhance
            degree: Polynomial degree
            fit: If True, fit the generator; if False, use fitted generator
            
        Returns:
            pd.DataFrame: DataFrame with polynomial features
        """
        if fit:
            key_features = ['automation_risk_percent', 'skill_gap_index', 'ai_adoption_level']
            available_features = [f for f in key_features if f in df.columns]
            
            self.polynomial_generator = PolynomialFeatures(degree=degree, include_bias=False)
            poly_features = self.polynomial_generator.fit_transform(df[available_features])
            
            # Get feature names
            feature_names = self.polynomial_generator.get_feature_names_out(available_features)
            logger.info(f"  Created {len(feature_names)} polynomial features (degree={degree})")
        else:
            if self.polynomial_generator is None:
                logger.warning("Polynomial generator not fitted. Skipping.")
                return df
            
            key_features = [f for f in ['automation_risk_percent', 'skill_gap_index', 'ai_adoption_level'] 
                          if f in df.columns]
            poly_features = self.polynomial_generator.transform(df[key_features])
            feature_names = self.polynomial_generator.get_feature_names_out(key_features)
        
        # Add polynomial features to dataframe
        for i, name in enumerate(feature_names):
            df[f'poly_{name}'] = poly_features[:, i]
        
        return df
    
    def perform_feature_selection(
        self, X: pd.DataFrame, y: pd.Series, k: int = 15, method: str = "f_regression"
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select top k features using statistical methods.
        
        Args:
            X: Feature matrix
            y: Target variable
            k: Number of features to select
            method: 'f_regression', 'mutual_info'
            
        Returns:
            Tuple[pd.DataFrame, List[str]]: Selected features dataframe and feature names
        """
        if method == "f_regression":
            selector = SelectKBest(f_regression, k=min(k, X.shape[1]))
        else:
            selector = SelectKBest(mutual_info_regression, k=min(k, X.shape[1]))
        
        X_selected = selector.fit_transform(X, y)
        self.feature_selector = selector
        
        # Get selected feature names
        selected_mask = selector.get_support()
        self.selected_features = X.columns[selected_mask].tolist()
        
        # Get feature scores
        scores = selector.scores_
        feature_scores = pd.DataFrame({
            'feature': X.columns,
            'score': scores
        }).sort_values('score', ascending=False)
        
        logger.info(f"✓ Feature selection complete ({method}, k={k})")
        logger.info(f"  Selected features:\n{feature_scores.head(10).to_string()}")
        
        return pd.DataFrame(X_selected, columns=self.selected_features), self.selected_features
    
    def fit_transform(
        self, X: pd.DataFrame, y: pd.Series = None, 
        create_interactions: bool = True,
        create_domain: bool = True,
        create_aggregates: bool = True,
        select_features: bool = True,
        n_features: int = 15
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Complete feature engineering pipeline.
        
        Args:
            X: Feature matrix (before feature engineering)
            y: Target variable (required if select_features=True)
            create_interactions: Whether to create interaction features
            create_domain: Whether to create domain features
            create_aggregates: Whether to create aggregate features
            select_features: Whether to perform feature selection
            n_features: Number of features to select
            
        Returns:
            Tuple[pd.DataFrame, List[str]]: Engineered features and feature names
        """
        logger.info("Starting feature engineering pipeline...")
        
        X = X.copy()
        
        # Create features
        if create_interactions:
            X = self.create_interaction_features(X)
        
        if create_domain:
            X = self.create_domain_features(X)
        
        if create_aggregates:
            X = self.create_aggregate_features(X)
        
        logger.info(f"✓ Feature engineering complete. Total features: {X.shape[1]}")
        
        # Feature selection
        if select_features:
            if y is None:
                raise ValueError("Target variable (y) required for feature selection")
            X_selected, selected_features = self.perform_feature_selection(X, y, k=n_features)
            self.is_fitted = True
            return X_selected, selected_features
        
        self.selected_features = X.columns.tolist()
        self.is_fitted = True
        return X, self.selected_features
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fitted feature engineering to new data.
        
        Args:
            X: Feature matrix
            
        Returns:
            pd.DataFrame: Transformed features
        """
        if not self.is_fitted:
            raise ValueError("Feature engineer not fitted. Call fit_transform() first.")
        
        X = X.copy()
        
        # Apply feature creation
        X = self.create_interaction_features(X)
        X = self.create_domain_features(X)
        X = self.create_aggregate_features(X)
        
        # Select fitted features
        if self.selected_features:
            X = X[self.selected_features]
        
        logger.info(f"✓ Feature transformation applied. Shape: {X.shape}")
        return X
    
    def save_artifacts(self, output_dir: Path = None) -> None:
        """Save feature engineering artifacts."""
        output_dir = output_dir or Config.MODELS_PATH / "model_artifacts"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.feature_selector:
            with open(output_dir / "feature_selector.pkl", "wb") as f:
                pickle.dump(self.feature_selector, f)
            logger.info(f"✓ Saved feature selector")
        
        if self.selected_features:
            import json
            with open(output_dir / "selected_features.json", "w") as f:
                json.dump(self.selected_features, f)
            logger.info(f"✓ Saved selected features")


if __name__ == "__main__":
    logger.info("Feature engineering module loaded successfully")
