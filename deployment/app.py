"""
FastAPI Deployment Application

Provides REST API endpoints for model predictions and health checks.
Handles model loading, input validation, and error handling.
"""

import logging
from typing import List, Dict, Optional
from contextlib import asynccontextmanager
import pickle
from pathlib import Path
import numpy as np
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model path
MODEL_PATH = Path(__file__).parent.parent / "models" / "trained_models" / "ai_replacement_predictor.pkl"
SCALER_PATH = Path(__file__).parent.parent / "models" / "model_artifacts" / "feature_scaler.pkl"


class JobInput(BaseModel):
    """Input schema for single prediction."""
    job_role: str = Field(..., description="Job role/title")
    industry: str = Field(..., description="Industry sector")
    country: str = Field(..., description="Country/region")
    automation_risk_percent: float = Field(..., ge=0, le=100, description="Automation risk (0-100)")
    skill_gap_index: float = Field(..., ge=0, le=1, description="Skill gap score (0-1)")
    salary_before_usd: float = Field(..., gt=0, description="Current salary in USD")
    remote_feasibility_score: float = Field(..., ge=0, le=1, description="Remote work feasibility")
    ai_adoption_level: float = Field(..., ge=0, le=1, description="AI adoption in industry")
    education_requirement_level: int = Field(..., ge=1, le=5, description="Education level required")
    skill_demand_growth_percent: float = Field(..., description="Skill demand growth rate")
    
    class Config:
        schema_extra = {
            "example": {
                "job_role": "Data Analyst",
                "industry": "Technology",
                "country": "USA",
                "automation_risk_percent": 65,
                "skill_gap_index": 0.65,
                "salary_before_usd": 90000,
                "remote_feasibility_score": 0.85,
                "ai_adoption_level": 0.75,
                "education_requirement_level": 3,
                "skill_demand_growth_percent": 12
            }
        }


class PredictionOutput(BaseModel):
    """Output schema for predictions."""
    ai_replacement_score: float = Field(..., description="AI replacement score (0-100)")
    risk_category: str = Field(..., description="Risk level: LOW, MEDIUM, HIGH")
    confidence: float = Field(..., description="Model confidence (0-1)")
    recommendations: List[str] = Field(..., description="Actionable recommendations")
    salary_trend: Dict[str, float] = Field(..., description="Salary impact analysis")


class BatchPredictionInput(BaseModel):
    """Input schema for batch predictions."""
    jobs: List[JobInput]
    
    @validator('jobs')
    def validate_batch_size(cls, v):
        if len(v) > 1000:
            raise ValueError('Maximum batch size is 1000')
        return v


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    version: str


# Global model storage
loaded_model = None
loaded_scaler = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global loaded_model, loaded_scaler
    try:
        if MODEL_PATH.exists():
            with open(MODEL_PATH, 'rb') as f:
                loaded_model = pickle.load(f)
            logger.info("Model loaded successfully")
        else:
            logger.warning(f"Model not found at {MODEL_PATH}")
        
        if SCALER_PATH.exists():
            with open(SCALER_PATH, 'rb') as f:
                loaded_scaler = pickle.load(f)
            logger.info("Scaler loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
    
    yield
    logger.info("Application shutdown")


# Create FastAPI app
app = FastAPI(
    title="AI Job Displacement Predictor",
    description="Predict job displacement risk from AI automation using machine learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=loaded_model is not None,
        version="1.0.0"
    )


@app.post("/predict", response_model=PredictionOutput, tags=["Predictions"])
async def predict(job: JobInput):
    """
    Predict AI displacement risk for a single job.
    
    Returns:
        - **ai_replacement_score**: Probability of AI replacement (0-100)
        - **risk_category**: LOW/MEDIUM/HIGH based on score
        - **confidence**: Model confidence in prediction (0-1)
        - **recommendations**: Personalized career recommendations
        - **salary_trend**: Expected salary impacts
    
    Example:
        ```json
        {
            "job_role": "Data Analyst",
            "industry": "Technology",
            "country": "USA",
            "automation_risk_percent": 65,
            ...
        }
        ```
    """
    try:
        if loaded_model is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Service temporarily unavailable."
            )
        
        # Prepare features (simplified example)
        features = np.array([[
            job.automation_risk_percent,
            job.skill_gap_index,
            job.salary_before_usd / 100000,  # Normalize
            job.remote_feasibility_score,
            job.ai_adoption_level,
            job.education_requirement_level,
            job.skill_demand_growth_percent
        ]])
        
        # Get prediction
        score = loaded_model.predict(features)[0] * 100  # Scale to 0-100
        
        # Determine risk category
        if score < 30:
            risk_category = "LOW"
        elif score < 70:
            risk_category = "MEDIUM"
        else:
            risk_category = "HIGH"
        
        # Generate recommendations
        recommendations = _generate_recommendations(job, risk_category)
        
        # Calculate salary trend
        salary_trend = {
            "current": job.salary_before_usd,
            "expected_impact_percent": _estimate_salary_impact(risk_category),
            "trend": "stable" if risk_category == "LOW" else "declining"
        }
        
        return PredictionOutput(
            ai_replacement_score=float(score),
            risk_category=risk_category,
            confidence=min(0.95, 0.7 + (abs(score - 50) / 100) * 0.25),
            recommendations=recommendations,
            salary_trend=salary_trend
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict_batch", response_model=List[PredictionOutput], tags=["Predictions"])
async def predict_batch(batch_input: BatchPredictionInput):
    """
    Batch prediction for multiple jobs.
    
    Maximum 1000 jobs per request.
    
    Returns:
        List of predictions for each job with same schema as /predict
    """
    try:
        if loaded_model is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Service temporarily unavailable."
            )
        
        predictions = []
        for job in batch_input.jobs:
            pred = await predict(job)
            predictions.append(pred)
        
        return predictions
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/model-info", tags=["Model"])
async def model_info():
    """Get model metadata and information."""
    return {
        "model_path": str(MODEL_PATH),
        "model_loaded": loaded_model is not None,
        "scaler_loaded": loaded_scaler is not None,
        "model_type": "RandomForest/GradientBoosting",
        "target_variable": "ai_replacement_score",
        "training_data": "AI Job Displacement 2020-2026",
        "features_count": 7,
        "supported_regions": ["USA", "Canada", "UK", "Germany", "Japan", "Singapore", "Australia", "India", "Brazil"]
    }


def _generate_recommendations(job: JobInput, risk_category: str) -> List[str]:
    """Generate personalized career recommendations."""
    recommendations = []
    
    if risk_category == "HIGH":
        recommendations.extend([
            "Invest in continuous upskilling in AI-complementary skills",
            "Consider certification programs in emerging technologies",
            "Explore adjacent roles with lower automation risk",
            "Develop uniquely human skills: leadership, creativity, emotional intelligence"
        ])
    elif risk_category == "MEDIUM":
        recommendations.extend([
            "Build expertise in AI tools and automation frameworks",
            "Enhance soft skills that complement AI (communication, problem-solving)",
            "Monitor industry trends in your sector"
        ])
    else:
        recommendations.extend([
            "Your role has strong resilience to AI disruption",
            "Focus on deepening expertise in specialized domain knowledge",
            "Stay updated with industry innovations"
        ])
    
    # Skill-based recommendations
    if job.skill_gap_index > 0.7:
        recommendations.append("High skill gap detected - prioritize training")
    
    if job.remote_feasibility_score < 0.4:
        recommendations.append("Consider roles with remote flexibility")
    
    return recommendations[:5]  # Return top 5 recommendations


def _estimate_salary_impact(risk_category: str) -> float:
    """Estimate salary impact percentage."""
    impact_map = {
        "LOW": 3.0,
        "MEDIUM": -2.0,
        "HIGH": -8.0
    }
    return impact_map.get(risk_category, 0.0)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "AI Job Displacement Predictor API",
        "version": "1.0.0",
        "documentation": "/docs",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "predict_batch": "/predict_batch",
            "model_info": "/model-info"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
