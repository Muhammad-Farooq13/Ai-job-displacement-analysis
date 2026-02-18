# Deployment Documentation

## Overview
This directory contains deployment configurations for the AI Job Displacement Analysis model.

## Quick Start

### Local Development
```bash
cd deployment
python app.py
```
Access API at `http://localhost:8000`
API docs at `http://localhost:8000/docs`

### Docker Deployment

#### Build Image
```bash
docker build -t ai-job-displacement:latest .
```

#### Run Container
```bash
docker run -p 8000:8000 ai-job-displacement:latest
```

### Docker Compose
```bash
docker-compose up -d
```

## Environment Variables
- `PYTHONUNBUFFERED=1` - Python output buffering disabled
- `LOG_LEVEL=INFO` - Logging level (DEBUG, INFO, WARNING, ERROR)

## API Endpoints

### Health Check
```
GET /health
```

### Single Prediction
```
POST /predict
Content-Type: application/json

{
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
```

### Batch Prediction
```
POST /predict_batch
Content-Type: application/json

{
    "jobs": [
        {...},
        {...}
    ]
}
```

### Model Info
```
GET /model-info
```

## Cloud Deployment Options

### Google Cloud Run
```bash
gcloud run deploy ai-job-displacement \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### AWS Lambda (with API Gateway)
Package as a container and deploy to ECR, then use Lambda.

### Heroku
```bash
heroku create ai-job-displacement
git push heroku main
```

## Monitoring
- Health endpoint: `/health`
- Logs: Docker logs or application logs
- Metrics: Can be extended with Prometheus

## Security Considerations
- Model runs as non-root user (appuser)
- Input validation on all endpoints
- Rate limiting recommended for production
- Use API key authentication for sensitive deployments

## Performance Tuning
- Worker count in uvicorn can be adjusted based on CPU cores
- Increase memory allocation for large batch predictions
- Consider caching for frequently predicted jobs

## Troubleshooting

### Model not loading
Check that `models/trained_models/ai_replacement_predictor.pkl` exists

### Port already in use
Change port binding: `docker run -p 8001:8000 ...`

### Memory issues
Increase container memory: `docker run -m 2g ...`
