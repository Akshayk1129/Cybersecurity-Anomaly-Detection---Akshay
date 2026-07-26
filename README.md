# 🛡️ Cybersecurity Anomaly Detection — AI-Powered UEBA System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

An enterprise-grade **User and Entity Behavior Analytics (UEBA)** system built from scratch for cybersecurity threat detection. The pipeline ingests 100,000 synthetic access logs, engineers 27 behavioral features, detects anomalies with Isolation Forest, classifies attacks into 7 categories using LightGBM, and presents results through an interactive SOC analyst dashboard.

> **Built for the Honeywell Internship Challenge**

---

## 🚀 Key Capabilities

| Capability | Description |
|---|---|
| **Anomaly Detection** | Isolation Forest (200 trees) with ROC AUC **0.8459** and FPR **1.16%** |
| **Attack Classification** | LightGBM multi-class classifier with macro F1 **0.9995** across 7 attack types |
| **Risk Scoring** | Enterprise risk engine fusing anomaly confidence with 7 contextual parameters (0–100 score) |
| **Explainability** | SHAP TreeExplainer with per-event natural-language narratives for SOC analysts |
| **Concept Drift** | KS-test monitoring of behavioral feature distributions to flag model degradation |
| **Incident Response** | SOAR playbook engine mapping attack types × risk levels → automated mitigation actions |
| **Real-Time Inference** | Low-latency API for scoring single JSON events through the full pipeline |
| **Interactive Dashboard** | 5-page Streamlit + Plotly SOC portal with dark theme and drill-down investigation |

---

## 📐 Architecture

```
main.py (Orchestrator)
├── Phase 1: DataValidator          → Schema & data quality checks
├── Phase 3: FeatureEngineer        → 27 behavioral features (entropy, z-scores, streaks)
├── Phase 4: BaselineProfiler       → 500 entity + 17 department statistical profiles
├── Phase 2: DataPreprocessor       → Encoding, scaling → 111-column numeric matrix
├── Phase 5: AnomalyDetector        → Isolation Forest unsupervised detection
├── Phase 6: AttackClassifier       → LightGBM supervised 7-class classification
├── Phase 6b: RiskScoringEngine     → Contextual 0–100 risk scoring
├── Phase 7: ExplainabilityEngine   → SHAP attributions + narratives
└── Phase 8: DriftDetector          → KS-test baseline fitting
```

---

## 🗂️ Project Structure

```
Cybersecurity-Anomaly-Detection/
├── config/
│   └── config.yaml              # Central configuration (paths, hyperparameters, playbooks)
├── dashboard/
│   └── app.py                   # Streamlit SOC Dashboard (5 pages)
├── data/
│   ├── enriched_dataset.csv     # Feature-engineered dataset
│   └── processed_dataset.csv    # Fully encoded/scaled numeric matrix
├── explainability/
│   └── explainability_engine.py # SHAP-based explainability
├── feature_engineering/
│   └── feature_engineer.py      # 27 behavioral feature generators
├── incident_response/
│   └── response_engine.py       # SOAR playbook engine
├── inference/
│   └── realtime_pipeline.py     # Real-time single-event scoring API
├── models/
│   ├── anomaly_detection/       # Isolation Forest
│   ├── baseline/                # Entity & department profiling
│   ├── classification/          # LightGBM attack classifier
│   ├── concept_drift/           # KS-test drift detector
│   └── risk_scoring/            # Enterprise risk engine
├── preprocessing/
│   ├── data_validator.py        # Phase 1: Data quality validation
│   └── data_preprocessor.py     # Phase 2: Encoding & scaling
├── saved_models/                # Pre-trained model artifacts (.joblib)
├── tests/                       # 59 pytest tests across 11 suites
├── utils/
│   └── logger.py                # Structured logging infrastructure
├── main.py                      # Pipeline orchestrator
└── requirements.txt             # Python dependencies
```

---

## ⚡ Quick Start

### Local Development

```bash
# Clone the repository
git clone https://github.com/<your-username>/Cybersecurity-Anomaly-Detection.git
cd Cybersecurity-Anomaly-Detection

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline (trains all models)
python main.py

# Launch the dashboard
streamlit run dashboard/app.py
```

### Run Tests

```bash
pytest tests/ -v
```

---

## 🔬 Attack Types Detected

| # | Attack Type | Description |
|---|---|---|
| 1 | **Brute Force** | Rapid repeated authentication attempts |
| 2 | **Credential Stuffing** | Reuse of compromised credentials across services |
| 3 | **Device Spoofing** | Impersonation of trusted device fingerprints |
| 4 | **Impossible Travel** | Geographically implausible login sequences |
| 5 | **Insider Drift** | Gradual behavioral deviation from established baseline |
| 6 | **Lateral Movement** | Sequential access across network segments |
| 7 | **Low-and-Slow Exfiltration** | Covert data extraction below detection thresholds |

---

## 🧪 Testing

The test suite includes **59 tests** across 11 test classes covering all pipeline phases, enhancements, and integration scenarios:

```
tests/
├── test_pipeline.py          # 48 tests (Phases 1–7 + integration)
├── test_risk_scoring.py      # Risk engine validation
├── test_realtime_pipeline.py # Real-time inference tests
├── test_concept_drift.py     # KS-test drift detection tests
└── test_incident_response.py # SOAR playbook routing tests
```

---

## 🛠️ Tech Stack

- **Python 3.10+** — Core language
- **Pandas / NumPy** — Data processing
- **scikit-learn** — Isolation Forest, preprocessing
- **LightGBM / XGBoost** — Gradient boosting classifiers
- **SHAP** — Model explainability
- **Streamlit + Plotly** — Interactive dashboard
- **SciPy** — Statistical testing (KS-test for drift)
- **PyYAML** — Configuration management

---

## 📄 License

This project is licensed under the MIT License.
