# 🤖 AI Job Displacement Analysis

[![CI](https://github.com/Muhammad-Farooq13/Ai-job-displacement-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/Muhammad-Farooq13/Ai-job-displacement-analysis/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%E2%89%A51.8-orange)
![Tests](https://img.shields.io/badge/tests-14%20passed-brightgreen)

Predict AI-driven job displacement scores across 20 roles, 10 industries, and 10 countries (2020–2030).

Live demo → [Streamlit Cloud](https://share.streamlit.io) (deploy `streamlit_app.py`)

---

## Problem Statement

As AI adoption accelerates, the labour market faces profound disruption. This project builds an
end-to-end ML pipeline that estimates **AI Replacement Score (0–100)** for a given job, given
automation risk, AI adoption level, skill gap, salary data, and demographic context.

Higher scores indicate stronger displacement pressure.

---

## Key Results

| Metric | Value |
|--------|-------|
| **R² Score** | 0.9497 |
| **RMSE** | 3.106 |
| **MAE** | 2.464 |
| **Training samples** | 12,000 |
| **Test samples** | 3,000 |

---

## Dataset Schema

Synthetic dataset of 15,000 rows matching `ai_job_replacement_2020_2026.csv`:

| Column | Type | Description |
|--------|------|-------------|
| `job_id` | int | Unique identifier |
| `job_role` | str | One of 20 roles (Data Analyst, Truck Driver, …) |
| `industry` | str | One of 10 industries |
| `country` | str | One of 10 countries |
| `year` | int | 2020 – 2026 |
| `automation_risk_percent` | float | % likelihood of task automation |
| `skill_gap_index` | float | 0–1, gap between current and required skills |
| `ai_adoption_level` | float | 0–1, industry's AI penetration |
| `remote_feasibility_score` | float | 0–1, remote work viability |
| `education_requirement_level` | int | 1–5 |
| `salary_before_usd` | float | Annual salary pre-AI transition |
| `salary_after_usd` | float | Annual salary post-AI transition |
| `salary_change_percent` | float | % salary delta |
| `skill_demand_growth_percent` | float | % growth in skill demand |
| **`ai_replacement_score`** | **float** | **Target (0–100)** |

---

## Quick Start

```bash
git clone https://github.com/Muhammad-Farooq13/Ai-job-displacement-analysis.git
cd Ai-job-displacement-analysis
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt

# Build demo bundle (generates synthetic data + trains model)
python train_demo.py

# Launch Streamlit dashboard
streamlit run streamlit_app.py
```

---

## Streamlit Dashboard (5 Tabs)

| Tab | Content |
|-----|---------|
| 🌍 **Overview** | KPIs, risk by industry bar chart, year-on-year trend |
| 📊 **Model Results** | R² gauge, actual vs predicted scatter, feature importance |
| 📈 **Analytics** | Risk by country, salary impact by risk tier, role × industry heatmap |
| ⚙️ **Pipeline & API** | Pipeline diagram, FastAPI endpoints, Docker commands |
| 🔮 **Predict** | Interactive sliders → real-time AI Replacement Score + risk badge + recommendations |

---

## Project Structure

```
Ai-job-displacement-analysis/
├── src/
│   ├── config.py              ← Config class, feature metadata
│   ├── data_loader.py         ← DataValidator, DataLoader
│   ├── preprocessing.py       ← DataPreprocessor (impute, encode, split)
│   ├── feature_engineering.py ← Interaction, domain, temporal features
│   ├── model_training.py      ← ModelTrainer (GBR + baseline RF)
│   ├── evaluation.py          ← ModelEvaluator (metrics + plots)
│   └── utils.py
├── deployment/
│   └── app.py                 ← FastAPI: /health /predict /predict_batch /model-info
├── models/
│   ├── demo_bundle.pkl        ← Streamlit artefacts (auto-rebuilt if missing)
│   └── model_artifacts/       ← feature_names.json, model_metadata.json
├── tests/                     ← 14 unit tests (pytest)
├── .github/workflows/
│   └── ci.yml                 ← Python 3.11 & 3.12 matrix, ruff, codecov
├── train_demo.py              ← Synthesize data + build demo bundle
├── train_model.py             ← Full training pipeline
├── streamlit_app.py           ← Interactive dashboard
├── requirements.txt
├── requirements-ci.txt
├── runtime.txt                ← python-3.11.9  (Streamlit Cloud)
└── .python-version            ← 3.11
```

---

## FastAPI Deployment

```bash
# Install
pip install fastapi uvicorn

# Run
uvicorn deployment.app:app --reload --port 8000

# Docker
docker build -t ai-job-displacement .
docker run -p 8000:8000 ai-job-displacement
```

Endpoints: `GET /health` · `POST /predict` · `POST /predict_batch` · `GET /model-info`

---

## CI/CD

GitHub Actions matrix (`Python 3.11 / 3.12`):

- `ruff check src/` — linting
- `pytest tests/ -v --cov=src --cov-report=xml` — 14 tests
- Codecov upload
- Docker build (no push, validation only)

---

## Author

- **Muhammad Farooq**
- GitHub: [Muhammad-Farooq13](https://github.com/Muhammad-Farooq13)
