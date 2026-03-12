# Changelog

## [1.0.0] - 2026-03-12

### Added
- `train_demo.py` -- synthesizes 15,000-row dataset + trains GradientBoostingRegressor (R2=0.9497, RMSE=3.106)
- `streamlit_app.py` -- 5-tab dashboard: Overview, Model Results, Analytics, Pipeline & API, Predict
- `.streamlit/config.toml` -- dark red theme
- `requirements-ci.txt` -- CI-only dependencies
- `.python-version` + `runtime.txt` -- pin Python 3.11 for Streamlit Cloud
- Self-healing bundle: auto-rebuilds models/demo_bundle.pkl if pickle load fails

### Changed
- `requirements.txt` -- updated to modern >= lower bounds (scikit-learn>=1.8, plotly>=6.0, streamlit>=1.36)
- `.github/workflows/ci.yml` -- Python 3.11/3.12 matrix, ruff, codecov@v5, Docker buildx@v3
- `README.md` -- full rewrite with schema table, results, quick-start, project structure

### Fixed
- `src/data_loader.py` -- corrected error messages; added load_raw_data() method
- `src/feature_engineering.py` -- added risk_exposure, education_barrier, create_temporal_features()
- `src/model_training.py` -- added train_baseline_model() with RandomForestClassifier
- `src/preprocessing.py` -- added split_data() returning X_train, X_test, y_train, y_test
- All 14 unit tests now pass (was 4/14)
