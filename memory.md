# Cybersecurity Anomaly Detection (UEBA) - Project Memory & Context

## 1. Project Context
**Goal**: Build a complete AI-Powered Behavioral Anomaly Detection (User and Entity Behavior Analytics - UEBA) system from scratch for the Honeywell Internship Challenge.
**Dataset**: `Synthetic_Cybersecurity_Access_Logs_v2.xlsx` (~100,000 events, 30 columns, 500 entities including Users, Service Accounts, and Edge Devices, with a known 2% anomaly rate).
**Constraints**: 
- Must be an original implementation (no cloned/copied repositories).
- Enterprise-grade production quality (classes, typing, structured logging, modular architecture).
- Handled sequentially in 10 phases.
- **Agent Operational Rule**: Gemini agents strictly only update documentation/plan files (`implementation_plan.md` and `memory.md`). Gemini agents will NOT perform building, coding, or code editing tasks.

## 2. Directory Structure & File Roles

### Root
- `main.py`: The central orchestrator that wires together all the pipeline phases sequentially.
- `requirements.txt`: Python package dependencies (pandas, scikit-learn, xgboost, lightgbm, shap, streamlit, pyyaml, etc.).
- `memory.md`: This file, tracking the entire project context, structure, and progress.

### `config/`
- `config.yaml`: The central source of truth for all configurations (column names, validation boundaries, encoding maps, missing value strategies, modeling hyperparameters, and thresholds).

### `utils/`
- `logger.py`: Provides standardized, thread-safe logging infrastructure used across all modules.

### `preprocessing/`
- `data_validator.py` (Phase 1): Validates the raw dataset for schema integrity, missing values, timestamp consistency, and logical bound checks before ingestion.
- `data_preprocessor.py` (Phase 2): Cleans, encodes, and standard-scales the data into a model-ready numeric matrix. Persists scalers/encoders.

### `feature_engineering/`
- `feature_engineer.py` (Phase 3): Generates 27 advanced behavioral features (expanding statistics, Z-scores, Shannon entropy for lateral movement, streaks, etc.) using highly optimized vectorized pandas operations.

### `models/`
- `baseline/baseline_profiler.py` (Phase 4): Builds statistical normal-behavior profiles for every entity and department (for cold-starts). Computes deviation scores for current events vs. historical profiles.
- `anomaly_detection/anomaly_detector.py` (Phase 5): An unsupervised Isolation Forest model trained to detect outlier events. Outputs continuous risk scores (0-100) and binary predictions.
- `classification/attack_classifier.py` (Phase 6): A supervised multi-class model (comparing Random Forest, XGBoost, LightGBM) trained *only* on anomalous events to classify them into the 7 known attack types (e.g., Brute Force, Lateral Movement).

### `data/`
- `enriched_dataset.csv`: Output of Phase 3 (Feature Engineering). Contains raw data + engineered features.
- `processed_dataset.csv`: Output of Phase 2 (Preprocessing). The fully encoded/scaled numeric matrix ready for ML models.

### `reports/`
- `data_validation_report.md`: Markdown report generated automatically by Phase 1 detailing data quality checks.

### `saved_models/`
- Contains joblib files generated during preprocessing (`label_encoders.joblib`, `ordinal_encoder.joblib`, `scaler.joblib`) and will contain fitted ML models.

### `logs/`
- `ueba_system.log`: Execution logs output by the `utils/logger.py`.

### `explainability/`
- `explainability_engine.py` (Phase 7): SHAP-based explainability engine. Loads saved Isolation Forest and LightGBM models. Provides per-event SHAP feature attributions with natural-language narratives for SOC analysts and global feature importance rankings.

### `dashboard/`
- `app.py` (Phase 8): Interactive Streamlit SOC dashboard with 4 pages: Executive Summary (KPIs, attack distribution, risk histogram, department anomalies, hourly timeline, global feature importance), Alert Triage (filterable alert table with risk score progress bars), Entity Profiler (profile comparison, radar chart, event timeline), Explainability (per-alert SHAP drill-down with contribution charts and narratives). Launch: `streamlit run dashboard/app.py`.

### `tests/`
- `test_pipeline.py` (Phase 9): Comprehensive pytest suite with 48 tests across 7 classes: TestDataValidation (7), TestFeatureEngineering (7), TestBaselineProfiling (5), TestPreprocessing (6), TestAnomalyDetection (7), TestAttackClassification (6), TestExplainability (5), TestPipelineIntegration (5). All tests pass. Run: `pytest tests/test_pipeline.py -v`.

---

## 3. Work Completed
- **Phase 1 (Validation)**: Completely developed, successfully runs. Discovered expected missing values in `browser` and fixed negative `session_duration_min` anomalies.
- **Phase 2 (Preprocessing)**: Completely developed, successfully runs. Handles data clamping, date extraction, scaling, and categorical encoding. Produces 111-column feature matrix.
- **Phase 3 (Feature Engineering)**: Completely developed, successfully runs. Vectorized computations reduce 100k row processing time to ~8 seconds. Expands dataset from 30 to 57 columns (27 new features).
- **Phase 4 (Baseline Profiling)**: Completely developed, wired, and executed. Built 500 entity profiles + 17 department profiles. 0 cold-start entities (all had >= 5 events). Computes 6 profile deviation features. Runs on raw enriched data BEFORE preprocessing.
- **Phase 5 (Anomaly Detection)**: Completely developed, wired, and executed. Isolation Forest (200 trees, contamination=0.02). Detected 2,000 anomalies. Metrics: Precision=0.4305, Recall=0.4305, F1=0.4305, ROC AUC=0.8459, FPR=0.0116, Top-1% Recall=0.3225.
- **Phase 6 (Attack Classification)**: Completely developed, wired, and executed. Compared 3 models via 5-fold Stratified CV: RandomForest (F1=0.9982), XGBoost (F1=0.9970), LightGBM (F1=0.9995). Best model: LightGBM. Perfect classification on training data (macro F1=1.0000) across all 7 attack types.
- **Phase 7 (Explainability)**: Completely developed, wired, and executed. Uses SHAP TreeExplainer for both Isolation Forest and LightGBM. Generates per-event local attributions with human-readable narratives (e.g., "Login Status -> +4.1554"). Global feature importance ranking saved. Handles false-positive anomalies gracefully by selecting true-positive samples. Top global feature: `time_since_last_login_min`. Saved `explainability_summary.joblib`.
- **Phase 8 (SOC Dashboard)**: Completely developed and verified. Streamlit + Plotly dashboard with 4 pages: Executive Summary (5 KPI cards, 4 interactive charts, global feature importance), Alert Triage (filterable table with risk score progress bars, entity/department/attack type filters), Entity Profiler (behavioral profile vs department baseline, radar chart, event timeline scatter), Explainability (per-alert SHAP attribution bar chart, natural-language narrative, raw event details). Serves on `http://localhost:8501`. Premium dark theme with Inter font, glassmorphism cards, and purple gradient color palette.
- **Phase 9 (Evaluation & Testing)**: Completely developed and verified. 53 pytest tests across 9 test suites covering all pipeline phases, risk engine, real-time pipeline, and integration tests. All 53 tests pass in ~2.3 minutes.
- **Enhancement 1 (Risk Scoring Engine)**: Completely developed and integrated. Fuses anomaly confidence with 7 context parameters into a 0-100 score. Integrated into `main.py` and `dashboard/app.py`.
- **Enhancement 2 (Real-Time Inference Pipeline)**: Completely developed and verified. Low-latency API class (`UEBARealTimePipeline`) in `inference/realtime_pipeline.py` for scoring single raw JSON events in real-time. Added `.transform()` to `DataPreprocessor`. Tested in `tests/test_realtime_pipeline.py`.
- **Enhancement 3 (Configuration-Driven System)**: Completely developed and verified. Model hyperparameters for Isolation Forest and Attack Classifiers (Random Forest, XGBoost, LightGBM) plus candidate model toggles are now parsed directly from `config/config.yaml`. Full test suite re-verified (53 passed).
- **Enhancement 4 (Concept Drift Support)**: Completely developed and verified. Created `models/concept_drift/drift_detector.py` to monitor distribution shifts using the KS-test. Integrated into `main.py` (Phase 8 to build `drift_baseline.joblib`) and added a new page to `dashboard/app.py`. Verified via `tests/test_concept_drift.py`.
- **Enhancement 5 (Incident Response Engine)**: Completely developed and verified. A lightweight SOAR engine mapping anomaly types and risk levels to automated playbook actions. Integrated directly into `dashboard/app.py` for alert triage drill-downs and `inference/realtime_pipeline.py` for automated downstream mitigation. Verified via `tests/test_incident_response.py`.

### Saved Artifacts (saved_models/)
- `entity_profiles.joblib`: 500 entity behaviour profiles
- `department_profiles.joblib`: 17 department aggregate profiles
- `isolation_forest.joblib`: Fitted Isolation Forest model
- `attack_classifier.joblib`: Fitted LightGBM classifier
- `attack_label_encoder.joblib`: Label encoder for 7 attack types
- `model_comparison.joblib`: Cross-validation comparison results
- `label_encoders.joblib`, `ordinal_encoder.joblib`, `scaler.joblib`: Preprocessing transformers
- `explainability_summary.joblib`: Global SHAP feature importance summary

---

## 4. Work In Progress / Immediate Next Steps
- **Phase 10 (Documentation)**: Final README, API docs, code clean-up.

---

## 5. Future Work (Pending Phases)
- **Phase 10 (Documentation)**: Final codebase review, API docs, inline comments, and README polishing.
