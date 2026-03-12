"""
train_demo.py — Build demo bundle for the AI Job Displacement Streamlit dashboard.

Generates a synthetic dataset aligned with the ai_job_replacement_2020_2026 schema
(15 000 rows, job roles × industries × countries), trains a GradientBoostingRegressor,
and saves models/demo_bundle.pkl.
"""

from __future__ import annotations

import pathlib
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

SEED = 42
N_ROWS = 15_000
BUNDLE_PATH = pathlib.Path("models/demo_bundle.pkl")

JOB_ROLES = [
    "Data Analyst", "Software Engineer", "Accountant", "Customer Service Rep",
    "Marketing Manager", "HR Specialist", "Nurse", "Teacher", "Lawyer",
    "Supply Chain Manager", "Sales Representative", "Financial Analyst",
    "Graphic Designer", "Truck Driver", "Radiologist", "Pharmacist",
    "Journalist", "Architect", "Civil Engineer", "Project Manager",
]

INDUSTRIES = [
    "Technology", "Finance", "Healthcare", "Education", "Manufacturing",
    "Retail", "Transportation", "Energy", "Media", "Legal Services",
]

COUNTRIES = [
    "United States", "United Kingdom", "Germany", "Japan", "Canada",
    "Australia", "India", "France", "Singapore", "Brazil",
]

# Pre-defined automation risk by job role (0-100)
ROLE_BASE_RISK = {
    "Data Analyst": 45, "Software Engineer": 30, "Accountant": 75,
    "Customer Service Rep": 80, "Marketing Manager": 40, "HR Specialist": 55,
    "Nurse": 20, "Teacher": 25, "Lawyer": 35, "Supply Chain Manager": 50,
    "Sales Representative": 60, "Financial Analyst": 65, "Graphic Designer": 50,
    "Truck Driver": 85, "Radiologist": 55, "Pharmacist": 45,
    "Journalist": 40, "Architect": 35, "Civil Engineer": 30, "Project Manager": 38,
}

# Base salaries (USD) by role
ROLE_BASE_SALARY = {
    "Data Analyst": 80_000, "Software Engineer": 120_000, "Accountant": 65_000,
    "Customer Service Rep": 40_000, "Marketing Manager": 90_000, "HR Specialist": 60_000,
    "Nurse": 75_000, "Teacher": 55_000, "Lawyer": 130_000, "Supply Chain Manager": 85_000,
    "Sales Representative": 60_000, "Financial Analyst": 95_000, "Graphic Designer": 60_000,
    "Truck Driver": 50_000, "Radiologist": 250_000, "Pharmacist": 120_000,
    "Journalist": 55_000, "Architect": 80_000, "Civil Engineer": 85_000, "Project Manager": 95_000,
}


def generate_synthetic_data(n_rows: int = N_ROWS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    job_roles = rng.choice(JOB_ROLES, n_rows)
    industries = rng.choice(INDUSTRIES, n_rows)
    countries = rng.choice(COUNTRIES, n_rows)
    years = rng.integers(2020, 2027, n_rows)

    automation_risk = np.array([
        np.clip(ROLE_BASE_RISK[r] + rng.normal(0, 8), 5, 99)
        for r in job_roles
    ])
    skill_gap_index = np.clip(automation_risk / 100 * 0.8 + rng.normal(0, 0.1, n_rows), 0.01, 0.99)
    ai_adoption_level = np.clip(rng.uniform(0.1, 0.9, n_rows), 0.05, 0.99)
    remote_feasibility = np.clip(rng.uniform(0.1, 0.95, n_rows), 0.05, 0.99)
    skill_demand_growth = rng.uniform(-5, 30, n_rows)
    education_level = rng.integers(1, 6, n_rows)

    salary_before = np.array([
        max(20_000, ROLE_BASE_SALARY[r] * rng.uniform(0.8, 1.3))
        for r in job_roles
    ])
    # Salary change correlated with risk: high risk → negative change
    salary_change_pct = -automation_risk * 0.05 + rng.normal(2, 5, n_rows)
    salary_after = salary_before * (1 + salary_change_pct / 100)

    # Target: ai_replacement_score (0-100), driven by automation risk + AI adoption + skill gap
    ai_replacement_score = np.clip(
        0.50 * automation_risk
        + 0.25 * ai_adoption_level * 100
        + 0.15 * skill_gap_index * 100
        - 0.10 * remote_feasibility * 100
        + rng.normal(0, 3, n_rows),
        0, 100,
    )

    df = pd.DataFrame({
        "job_id": np.arange(1, n_rows + 1),
        "job_role": job_roles,
        "industry": industries,
        "country": countries,
        "year": years,
        "automation_risk_percent": automation_risk.round(2),
        "ai_replacement_score": ai_replacement_score.round(2),
        "skill_gap_index": skill_gap_index.round(4),
        "salary_before_usd": salary_before.round(0),
        "salary_after_usd": salary_after.round(0),
        "salary_change_percent": salary_change_pct.round(2),
        "skill_demand_growth_percent": skill_demand_growth.round(2),
        "remote_feasibility_score": remote_feasibility.round(4),
        "ai_adoption_level": ai_adoption_level.round(4),
        "education_requirement_level": education_level,
    })
    return df


def build_bundle(bundle_path: str = str(BUNDLE_PATH)) -> dict:
    print("Generating synthetic AI job displacement data …")
    df = generate_synthetic_data()
    print(f"  {len(df)} rows | {df['job_role'].nunique()} roles | "
          f"{df['industry'].nunique()} industries | {df['country'].nunique()} countries")

    # Encode categoricals
    le_role = LabelEncoder()
    le_industry = LabelEncoder()
    le_country = LabelEncoder()
    df_enc = df.copy()
    df_enc["job_role_enc"] = le_role.fit_transform(df["job_role"])
    df_enc["industry_enc"] = le_industry.fit_transform(df["industry"])
    df_enc["country_enc"] = le_country.fit_transform(df["country"])

    feature_cols = [
        "year", "automation_risk_percent", "skill_gap_index",
        "salary_before_usd", "salary_change_percent",
        "skill_demand_growth_percent", "remote_feasibility_score",
        "ai_adoption_level", "education_requirement_level",
        "job_role_enc", "industry_enc", "country_enc",
    ]
    TARGET = "ai_replacement_score"

    X = df_enc[feature_cols]
    y = df_enc[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    print("Training GradientBoostingRegressor …")
    model = GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.08, max_depth=5,
        subsample=0.85, random_state=SEED,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    print(f"  RMSE={rmse:.3f}  MAE={mae:.3f}  R2={r2:.4f}")

    # ── Analytics ──────────────────────────────────────────────────────────────
    feat_imp = dict(zip(feature_cols, model.feature_importances_.tolist()))

    # Risk distribution by job role
    role_risk = (
        df.groupby("job_role")["ai_replacement_score"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "avg_score", "std": "std_score", "count": "obs"})
        .sort_values("avg_score", ascending=False)
    )

    # Risk by industry
    industry_risk = (
        df.groupby("industry")["ai_replacement_score"]
        .mean()
        .reset_index()
        .rename(columns={"ai_replacement_score": "avg_score"})
        .sort_values("avg_score", ascending=False)
    )

    # Risk by country
    country_risk = (
        df.groupby("country")["ai_replacement_score"]
        .mean()
        .reset_index()
        .rename(columns={"ai_replacement_score": "avg_score"})
        .sort_values("avg_score", ascending=False)
    )

    # Year trend
    year_trend = (
        df.groupby("year")["ai_replacement_score"]
        .mean()
        .reset_index()
        .rename(columns={"ai_replacement_score": "avg_score"})
    )

    # Salary impact by risk category
    df["risk_category"] = pd.cut(
        df["ai_replacement_score"],
        bins=[0, 30, 70, 100],
        labels=["LOW", "MEDIUM", "HIGH"],
    )
    salary_impact = (
        df.groupby("risk_category")[["salary_before_usd", "salary_after_usd", "salary_change_percent"]]
        .mean()
        .reset_index()
    )

    # Actual vs predicted
    avp = pd.DataFrame({"actual": y_test.values[:200], "predicted": y_pred[:200]})

    # Industry × automation heatmap data
    heatmap_data = (
        df.groupby(["industry", "job_role"])["ai_replacement_score"]
        .mean()
        .reset_index()
    )

    bundle = {
        "model": model,
        "feature_cols": feature_cols,
        "le_role": le_role,
        "le_industry": le_industry,
        "le_country": le_country,
        "job_roles": JOB_ROLES,
        "industries": INDUSTRIES,
        "countries": COUNTRIES,
        "dataset_stats": {
            "total_rows": len(df),
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "n_roles": df["job_role"].nunique(),
            "n_industries": df["industry"].nunique(),
            "n_countries": df["country"].nunique(),
            "year_min": int(df["year"].min()),
            "year_max": int(df["year"].max()),
        },
        "results": {"rmse": round(rmse, 3), "mae": round(mae, 3), "r2": round(r2, 4)},
        "feature_importance": feat_imp,
        "role_risk": role_risk.to_dict(orient="records"),
        "industry_risk": industry_risk.to_dict(orient="records"),
        "country_risk": country_risk.to_dict(orient="records"),
        "year_trend": year_trend.to_dict(orient="records"),
        "salary_impact": salary_impact.to_dict(orient="records"),
        "actual_vs_pred": avp.to_dict(orient="records"),
        "heatmap_data": heatmap_data.to_dict(orient="records"),
    }

    out_path = pathlib.Path(bundle_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"  Bundle saved → {out_path}")
    print("Done.")
    return bundle


def main() -> None:
    build_bundle()


if __name__ == "__main__":
    main()
