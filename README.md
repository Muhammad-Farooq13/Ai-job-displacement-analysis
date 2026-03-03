# AI Job Displacement Analysis

End-to-end data science project analyzing AI-driven automation risk and salary impact across industries.

## Project Snapshot
- **Problem**: Estimate job displacement risk and salary change due to AI adoption.
- **Scope**: Multi-year workforce records (2020–2026), industry and geography segmentation.
- **Goal**: Build interpretable predictive models and produce actionable labor-market insights.

## Business Questions
- Which job categories are most vulnerable to AI replacement?
- Which factors most strongly drive automation risk?
- How is salary impacted across sectors and regions?

## Data
- Primary dataset: `ai_job_replacement_2020_2026.csv`
- Typical features include role, industry, country, year, automation indicators, and salary variables.
- Target examples: `ai_replacement_score`, `automation_risk_percent`, and salary change indicators.

## Workflow
1. Data ingestion and schema validation
2. Cleaning, imputation, and encoding
3. Feature engineering
4. Model training and evaluation
5. Interpretation and reporting

Core code is in `src/`, experiments in `notebooks/`, and tests in `tests/`.

## Modeling and Evaluation
- Regression/ensemble models are used for risk-score prediction.
- Evaluation is performed with standard metrics (for example RMSE/MAE/R²).
- Feature importance analysis is included for interpretability.

## Project Structure
```
Ai-job-displacement-analysis/
├── src/                  # Data, features, modeling, evaluation modules
├── notebooks/            # EDA and experimentation
├── tests/                # Unit tests
├── models/               # Saved model artifacts
├── deployment/           # Serving/deployment assets
├── requirements.txt
├── setup.py
└── train_model.py
```

## Quick Start
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python train_model.py
```

## Why This Project Is Portfolio-Ready
- Addresses a real labor-market impact question.
- Demonstrates full ML lifecycle from raw data to deployable artifacts.
- Uses modular code structure, testing, and reproducible dependencies.

## Author
- Muhammad Farooq
- GitHub: https://github.com/Muhammad-Farooq13
