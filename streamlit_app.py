"""
streamlit_app.py — AI Job Displacement Analysis Dashboard
5-tab Streamlit interface: Overview | Model Results | Analytics | Pipeline & API | 🔮 Predict
"""

from __future__ import annotations

import pathlib
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Job Displacement Analysis",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BUNDLE_PATH = pathlib.Path("models/demo_bundle.pkl")


# ── Bundle loading / self-healing rebuild ──────────────────────────────────────
def _rebuild_bundle() -> None:
    """Rebuild demo_bundle.pkl by re-running train_demo.py."""
    train_script = pathlib.Path("train_demo.py")
    if not train_script.exists():
        raise FileNotFoundError("train_demo.py not found — cannot rebuild bundle.")
    subprocess.run([sys.executable, str(train_script)], check=True)


@st.cache_resource(show_spinner="Loading model …")
def load_bundle() -> dict:
    if not BUNDLE_PATH.exists():
        st.warning("Bundle not found — building now …")
        _rebuild_bundle()
    try:
        with open(BUNDLE_PATH, "rb") as f:
            return pickle.load(f)
    except Exception as exc:
        st.warning(f"Bundle load failed ({exc}) — rebuilding …")
        _rebuild_bundle()
        with open(BUNDLE_PATH, "rb") as f:
            return pickle.load(f)


bundle = load_bundle()

# Convenience unpacking
results = bundle["results"]
stats = bundle["dataset_stats"]
feat_imp = bundle["feature_importance"]
industry_risk = pd.DataFrame(bundle["industry_risk"])
role_risk = pd.DataFrame(bundle["role_risk"])
country_risk = pd.DataFrame(bundle["country_risk"])
year_trend = pd.DataFrame(bundle["year_trend"])
salary_impact = pd.DataFrame(bundle["salary_impact"])
avp = pd.DataFrame(bundle["actual_vs_pred"])
heatmap_df = pd.DataFrame(bundle["heatmap_data"])

RISK_COLOR = {"LOW": "#2ECC71", "MEDIUM": "#F39C12", "HIGH": "#E74C3C"}


# ── Helper: risk badge ─────────────────────────────────────────────────────────
def risk_badge(score: float) -> tuple[str, str]:
    if score < 30:
        return "LOW", "#2ECC71"
    if score < 70:
        return "MEDIUM", "#F39C12"
    return "HIGH", "#E74C3C"


# ══════════════════════════════════════════════════════════════════════════════
# Header
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    """
    <h1 style='text-align:center; color:#E74C3C;'>🤖 AI Job Displacement Analysis</h1>
    <p style='text-align:center; color:#AAAAAA; font-size:16px;'>
        Analyse automation risk across roles, industries and countries
        &nbsp;|&nbsp; GradientBoosting Regressor
    </p>
    <hr style='border:1px solid #333;'>
    """,
    unsafe_allow_html=True,
)

tabs = st.tabs(
    ["🌍 Overview", "📊 Model Results", "📈 Analytics", "⚙️ Pipeline & API", "🔮 Predict"]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Overview
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.subheader("Dataset Overview")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Records", f"{stats['total_rows']:,}")
    c2.metric("Job Roles", stats["n_roles"])
    c3.metric("Industries", stats["n_industries"])
    c4.metric("Countries", stats["n_countries"])
    c5.metric("Year Range", f"{stats['year_min']} – {stats['year_max']}")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Average AI Replacement Score by Industry")
        fig = px.bar(
            industry_risk,
            x="avg_score",
            y="industry",
            orientation="h",
            color="avg_score",
            color_continuous_scale="RdYlGn_r",
            labels={"avg_score": "Avg AI Replacement Score", "industry": "Industry"},
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#DDDDDD",
            coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Year-on-Year Risk Trend")
        fig2 = px.line(
            year_trend,
            x="year",
            y="avg_score",
            markers=True,
            labels={"avg_score": "Avg AI Replacement Score", "year": "Year"},
            color_discrete_sequence=["#E74C3C"],
        )
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#DDDDDD",
            xaxis=dict(tickmode="linear"),
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### Top 10 Highest-Risk Roles")
    top10 = role_risk[:10]
    fig3 = px.bar(
        top10,
        x="job_role",
        y="avg_score",
        color="avg_score",
        color_continuous_scale="Reds",
        error_y="std_score",
        labels={"avg_score": "Avg AI Replacement Score", "job_role": "Job Role"},
    )
    fig3.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#DDDDDD",
        coloraxis_showscale=False,
        margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Model Results
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.subheader("Model Performance")

    m1, m2, m3 = st.columns(3)
    m1.metric("R² Score", results["r2"])
    m2.metric("RMSE", results["rmse"])
    m3.metric("MAE", results["mae"])

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Actual vs Predicted")
        fig = px.scatter(
            avp, x="actual", y="predicted",
            labels={"actual": "Actual Score", "predicted": "Predicted Score"},
            opacity=0.6,
            color_discrete_sequence=["#E74C3C"],
        )
        mn, mx = avp.min().min(), avp.max().max()
        fig.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines",
                                 line=dict(dash="dash", color="#888"), name="Perfect fit",
                                 showlegend=False))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#DDDDDD", margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Feature Importance")
        fi_df = pd.DataFrame(
            {"feature": list(feat_imp.keys()), "importance": list(feat_imp.values())}
        ).sort_values("importance", ascending=True)
        fig2 = px.bar(
            fi_df, x="importance", y="feature", orientation="h",
            color="importance", color_continuous_scale="Oranges",
            labels={"importance": "Importance", "feature": "Feature"},
        )
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#DDDDDD", coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### R² Gauge")
    fig3 = go.Figure(go.Indicator(
        mode="gauge+number",
        value=results["r2"],
        title={"text": "R² Score", "font": {"color": "#DDDDDD"}},
        gauge={
            "axis": {"range": [0, 1], "tickfont": {"color": "#DDDDDD"}},
            "bar": {"color": "#E74C3C"},
            "steps": [
                {"range": [0, 0.5], "color": "#444"},
                {"range": [0.5, 0.8], "color": "#555"},
                {"range": [0.8, 1.0], "color": "#666"},
            ],
            "threshold": {
                "line": {"color": "#2ECC71", "width": 3},
                "thickness": 0.75,
                "value": 0.9,
            },
        },
        number={"font": {"color": "#E74C3C"}},
    ))
    fig3.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", font_color="#DDDDDD",
        height=280, margin=dict(l=30, r=30, t=30, b=10),
    )
    st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Analytics
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.subheader("In-Depth Analytics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Risk Score by Country")
        fig = px.bar(
            country_risk,
            x="avg_score", y="country", orientation="h",
            color="avg_score", color_continuous_scale="RdYlGn_r",
            labels={"avg_score": "Avg AI Replacement Score", "country": "Country"},
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#DDDDDD", coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Salary Impact by Risk Category")
        fig2 = go.Figure()
        for col_name, label, colour in [
            ("salary_before_usd", "Before AI", "#3498DB"),
            ("salary_after_usd", "After AI", "#E74C3C"),
        ]:
            fig2.add_trace(go.Bar(
                x=salary_impact["risk_category"].astype(str),
                y=salary_impact[col_name],
                name=label, marker_color=colour,
            ))
        fig2.update_layout(
            barmode="group",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#DDDDDD", legend=dict(font=dict(color="#DDDDDD")),
            xaxis_title="Risk Category", yaxis_title="Average Salary (USD)",
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### Role × Industry Heatmap (AI Replacement Score)")
    pivot = heatmap_df.pivot_table(
        index="job_role", columns="industry", values="ai_replacement_score", aggfunc="mean"
    ).round(1)
    fig3 = px.imshow(
        pivot,
        color_continuous_scale="RdYlGn_r",
        aspect="auto",
        labels={"color": "Avg Score"},
    )
    fig3.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#DDDDDD", margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Pipeline & API
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.subheader("ML Pipeline & Deployment")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Training Pipeline")
        st.markdown("""
        ```
        ┌─────────────────────────┐
        │  Synthetic Data Gen     │  15,000 rows
        │  ai_job_replacement…csv │  (train_demo.py)
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │  DataValidator          │  Schema + type checks
        │  DataPreprocessor       │  Missing values, splits
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │  FeatureEngineer        │  Interaction, Domain,
        │                         │  Temporal features
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │  GBRegressor (n=300)    │  lr=0.08, depth=5
        │  R²=0.9497, RMSE=3.106  │
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │  models/demo_bundle.pkl │  Analytics + artefacts
        └─────────────────────────┘
        ```
        """)

    with col2:
        st.markdown("#### FastAPI Deployment Endpoints")
        st.markdown("""
        | Method | Endpoint | Description |
        |--------|----------|-------------|
        | `GET` | `/health` | Service health check |
        | `POST` | `/predict` | Single job prediction |
        | `POST` | `/predict_batch` | Batch predictions (JSON array) |
        | `GET` | `/model-info` | Model metadata & feature list |
        """)
        st.markdown("#### Docker")
        st.code(
            "docker build -t ai-job-displacement .\n"
            "docker run -p 8000:8000 ai-job-displacement",
            language="bash",
        )
        st.markdown("#### Local FastAPI")
        st.code("uvicorn deployment.app:app --reload --port 8000", language="bash")

    st.markdown("#### Project Structure")
    st.code("""
ai-job-displacement/
├── src/
│   ├── config.py           ← Config class, feature lists
│   ├── data_loader.py      ← DataValidator, DataLoader
│   ├── preprocessing.py    ← DataPreprocessor
│   ├── feature_engineering.py ← FeatureEngineer
│   ├── model_training.py   ← ModelTrainer
│   ├── evaluation.py       ← ModelEvaluator
│   └── utils.py
├── deployment/
│   └── app.py              ← FastAPI application
├── models/
│   ├── demo_bundle.pkl     ← Streamlit demo artefacts
│   └── model_artifacts/    ← feature_names.json, etc.
├── tests/                  ← 14 unit tests (pytest)
├── train_demo.py           ← Build demo_bundle.pkl
├── train_model.py          ← Full training pipeline
├── streamlit_app.py        ← This dashboard
└── .github/workflows/ci.yml ← CI: Python 3.11 & 3.12
    """, language="text")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — 🔮 Predict
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.subheader("🔮 Predict AI Replacement Score")
    st.markdown(
        "Fill in the details below to estimate how strongly a job role may be "
        "displaced by AI automation."
    )

    col_cat, col_num = st.columns([1, 2])

    with col_cat:
        st.markdown("##### Job Characteristics")
        p_role = st.selectbox("Job Role", bundle["job_roles"])
        p_industry = st.selectbox("Industry", bundle["industries"])
        p_country = st.selectbox("Country", bundle["countries"])
        p_year = st.slider("Year", 2020, 2030, 2024)
        p_edu = st.slider("Education Requirement Level", 1, 5, 3,
                          help="1=No degree, 5=PhD")

    with col_num:
        st.markdown("##### Quantitative Indicators")
        p_auto_risk = st.slider("Automation Risk (%)", 0.0, 100.0, 50.0, 0.5)
        p_ai_adopt = st.slider("AI Adoption Level (0–1)", 0.0, 1.0, 0.5, 0.01)
        p_skill_gap = st.slider("Skill Gap Index (0–1)", 0.0, 1.0, 0.3, 0.01)
        p_remote = st.slider("Remote Feasibility (0–1)", 0.0, 1.0, 0.5, 0.01)
        p_skill_growth = st.slider("Skill Demand Growth (%)", -10.0, 50.0, 10.0, 0.5)
        p_sal_before = st.number_input("Salary Before AI (USD)", 20_000, 500_000, 80_000, 5_000)
        p_sal_change = st.slider("Salary Change (%)", -40.0, 30.0, 0.0, 0.5)

    st.divider()
    if st.button("🔮 Predict AI Replacement Score", type="primary", use_container_width=True):
        model = bundle["model"]
        le_role = bundle["le_role"]
        le_industry = bundle["le_industry"]
        le_country = bundle["le_country"]

        # Encode
        try:
            role_enc = le_role.transform([p_role])[0]
        except ValueError:
            role_enc = 0
        try:
            ind_enc = le_industry.transform([p_industry])[0]
        except ValueError:
            ind_enc = 0
        try:
            cnt_enc = le_country.transform([p_country])[0]
        except ValueError:
            cnt_enc = 0

        X_pred = pd.DataFrame([{
            "year": p_year,
            "automation_risk_percent": p_auto_risk,
            "skill_gap_index": p_skill_gap,
            "salary_before_usd": p_sal_before,
            "salary_change_percent": p_sal_change,
            "skill_demand_growth_percent": p_skill_growth,
            "remote_feasibility_score": p_remote,
            "ai_adoption_level": p_ai_adopt,
            "education_requirement_level": p_edu,
            "job_role_enc": role_enc,
            "industry_enc": ind_enc,
            "country_enc": cnt_enc,
        }])

        score = float(np.clip(model.predict(X_pred)[0], 0, 100))
        label, colour = risk_badge(score)

        r1, r2_col, r3 = st.columns([1, 2, 1])
        with r2_col:
            st.markdown(
                f"""
                <div style='text-align:center; padding:24px; border-radius:12px;
                            background:rgba(255,255,255,0.05);'>
                    <h2 style='color:{colour}; font-size:56px; margin:0;'>
                        {score:.1f}
                    </h2>
                    <p style='color:#AAAAAA; font-size:18px;'>AI Replacement Score (0 – 100)</p>
                    <span style='background:{colour}; color:#111; padding:6px 20px;
                                 border-radius:20px; font-weight:bold; font-size:18px;'>
                        {label} RISK
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("")
        st.markdown("##### What this means for your role")
        recommendations = {
            "LOW":    [
                "✅ Your role shows strong resilience against immediate AI displacement.",
                "💡 Focus on deepening domain expertise and creative problem-solving.",
                "📚 Stay current with AI tools as productivity enhancers, not replacements.",
            ],
            "MEDIUM": [
                "⚠️ Moderate automation risk — proactive upskilling is recommended.",
                "💡 Identify which parts of your role AI can assist vs. fully automate.",
                "🎓 Pursue certifications in data literacy or AI-adjacent skills.",
            ],
            "HIGH":   [
                "🚨 High displacement risk — significant transformation likely within 3–5 years.",
                "💡 Pivot toward oversight, strategy, and roles requiring human judgment.",
                "🔄 Consider retraining in roles with lower automation risk (engineering, healthcare).",
            ],
        }
        for tip in recommendations[label]:
            st.markdown(tip)

        with st.expander("View input features used for prediction"):
            st.dataframe(X_pred.T.rename(columns={0: "value"}), use_container_width=True)
