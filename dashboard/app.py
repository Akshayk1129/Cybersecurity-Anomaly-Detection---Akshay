# =============================================================================
# Cybersecurity Anomaly Detection - UEBA System
# Module: SOC Analyst Dashboard (Phase 8)
# =============================================================================
"""
Interactive Security Operations Center (SOC) Dashboard.

Provides real-time-style investigation views for security analysts:
  - Executive threat summary with KPIs
  - Alert triage console with risk-score filtering
  - Entity behavioural profile explorer
  - SHAP explainability drill-down per alert

Launch:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Path Setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from explainability.explainability_engine import ExplainabilityEngine
from models.risk_scoring.risk_engine import RiskScoringEngine
from incident_response.response_engine import IncidentResponseEngine

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="UEBA - Cybersecurity Anomaly Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS - Premium Dark Theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global ── */
    .stApp {
        font-family: 'Inter', sans-serif;
        background: #0B0F19;
        color: #CBD5E1;
    }
    html, body, [data-testid="stAppViewContainer"] {
        background: #0B0F19 !important;
    }

    /* Hide default Streamlit header bar for cleaner look */
    header[data-testid="stHeader"] {
        background: rgba(11, 15, 25, 0.8) !important;
        backdrop-filter: blur(12px) !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0D1117 0%, #111827 100%) !important;
        border-right: 1px solid rgba(99, 102, 241, 0.15);
    }
    [data-testid="stSidebar"] .stMarkdown h2 {
        font-size: 1.15rem !important;
    }
    [data-testid="stSidebar"] .stRadio label {
        font-size: 0.92rem !important;
        padding: 6px 0 !important;
    }

    /* ── Metric Cards ── */
    .metric-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.55), rgba(15, 23, 42, 0.75));
        border: 1px solid rgba(99, 102, 241, 0.18);
        border-radius: 14px;
        padding: 20px 16px;
        margin: 4px 0;
        text-align: center;
        min-height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        transition: all 0.25s ease;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
    }
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: rgba(99, 102, 241, 0.45);
        box-shadow: 0 8px 24px rgba(99, 102, 241, 0.18);
    }
    .metric-card h3 {
        color: #94A3B8;
        font-size: 0.65rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.4px;
        margin: 0 0 6px 0;
        white-space: nowrap;
    }
    .metric-card .metric-value {
        font-size: 1.6rem;
        font-weight: 800;
        margin: 0;
        white-space: nowrap;
        background: linear-gradient(135deg, #818CF8, #A78BFA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.2;
    }

    /* ── Section Headers ── */
    .section-header {
        color: #F1F5F9;
        font-size: 1.05rem;
        font-weight: 600;
        margin-top: 28px;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(148, 163, 184, 0.15);
    }

    /* ── Page Titles ── */
    .stApp h1 {
        font-weight: 800 !important;
        font-size: 1.8rem !important;
        color: #F8FAFC !important;
        letter-spacing: -0.02em;
    }
    .stApp h2 {
        font-weight: 700 !important;
        font-size: 1.4rem !important;
        color: #F1F5F9 !important;
    }

    /* ── Blockquote Descriptions ── */
    .stApp blockquote {
        border-left: 3px solid #6366F1;
        background: rgba(30, 41, 59, 0.35);
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0 20px 0;
        font-size: 0.88rem;
        color: #94A3B8;
        line-height: 1.6;
    }

    /* ── Narrative / Explainability Box ── */
    .narrative-box {
        background: rgba(30, 41, 59, 0.5);
        border-left: 3px solid #6366F1;
        border-radius: 6px;
        padding: 18px 22px;
        font-size: 0.9rem;
        line-height: 1.7;
        color: #CBD5E1;
        margin: 8px 0;
    }

    /* ── Data Tables ── */
    [data-testid="stDataFrame"] {
        border-radius: 10px !important;
        overflow: hidden;
    }

    /* ── Buttons ── */
    .stButton>button, .stDownloadButton>button {
        background: linear-gradient(135deg, #4F46E5, #7C3AED) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        padding: 0.45rem 1.1rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2) !important;
    }
    .stButton>button:hover, .stDownloadButton>button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 16px rgba(99, 102, 241, 0.35) !important;
        background: linear-gradient(135deg, #6366F1, #8B5CF6) !important;
    }

    /* ── Info / Warning / Error boxes ── */
    .stAlert {
        border-radius: 10px !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
    }

    /* ── Selectbox / Slider / Inputs ── */
    .stSelectbox label, .stSlider label, .stRadio label {
        font-weight: 500 !important;
        color: #94A3B8 !important;
        font-size: 0.85rem !important;
    }

    /* ── Metric (st.metric) containers ── */
    [data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(99, 102, 241, 0.12);
        border-radius: 12px;
        padding: 14px 18px;
    }

    /* ── Custom Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(99, 102, 241, 0.35); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(99, 102, 241, 0.6); }

    /* ── Plotly charts transparent bg ── */
    .js-plotly-plot .plotly .main-svg { background: transparent !important; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data Loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading enriched dataset...")
def load_enriched_data() -> pd.DataFrame:
    """Load the enriched dataset with original readable columns."""
    path = PROJECT_ROOT / "data" / "enriched_dataset.csv"
    df = pd.read_csv(str(path), parse_dates=["timestamp"])
    return df


@st.cache_data(show_spinner="Loading processed features...")
def load_processed_data() -> pd.DataFrame:
    """Load the fully processed numeric feature matrix."""
    path = PROJECT_ROOT / "data" / "processed_dataset.csv"
    return pd.read_csv(str(path))


@st.cache_resource(show_spinner="Loading saved models...")
def load_models() -> dict:
    """Load all saved models and artifacts."""
    models_dir = PROJECT_ROOT / "saved_models"
    assets = {}
    for name in [
        "isolation_forest", "attack_classifier", "attack_label_encoder",
        "entity_profiles", "department_profiles", "model_comparison",
        "explainability_summary", "scaler",
    ]:
        path = models_dir / f"{name}.joblib"
        if path.exists():
            assets[name] = joblib.load(str(path))
    return assets


@st.cache_resource(show_spinner="Initializing SHAP explainers...")
def load_explainability_engine() -> ExplainabilityEngine:
    """Initialize the ExplainabilityEngine (loads models + SHAP)."""
    return ExplainabilityEngine(config_path="config/config.yaml")


@st.cache_resource(show_spinner="Initializing Risk Scoring Engine...")
def load_risk_engine() -> RiskScoringEngine:
    return RiskScoringEngine(config_path="config/config.yaml")


@st.cache_resource(show_spinner="Initializing Incident Response Engine...")
def load_response_engine() -> IncidentResponseEngine:
    return IncidentResponseEngine(config_path="config/config.yaml")


@st.cache_data(show_spinner="Computing anomaly and risk scores...")
def compute_risk_results(_iforest, _risk_engine, enriched_df: pd.DataFrame, processed_df: pd.DataFrame) -> pd.DataFrame:
    """Compute anomaly predictions and run the Enterprise Risk Scoring Engine."""
    # Extract feature columns (exclude label and anomaly_type)
    feature_cols = [c for c in processed_df.columns if c not in ("label", "anomaly_type", "entity_id")]
    X = processed_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    predictions = _iforest.predict(X)
    raw_scores = _iforest.decision_function(X)

    # Normalise to 0-100 base anomaly score (lower decision_function = higher risk)
    min_s, max_s = raw_scores.min(), raw_scores.max()
    if max_s - min_s > 0:
        base_anomaly_scores = 100 * (1 - (raw_scores - min_s) / (max_s - min_s))
    else:
        base_anomaly_scores = np.zeros_like(raw_scores)

    attack_preds = enriched_df["anomaly_type"].values if "anomaly_type" in enriched_df.columns else None
    
    # Run Enterprise Risk Engine
    risk_results = _risk_engine.run(enriched_df, base_anomaly_scores, attack_preds)
    risk_results["prediction"] = predictions
    
    return risk_results


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------
def main() -> None:
    """Render the SOC Analyst Dashboard."""

    # --- Load Data ---
    enriched_df = load_enriched_data()
    processed_df = load_processed_data()
    assets = load_models()
    explainer = load_explainability_engine()
    risk_engine = load_risk_engine()
    response_engine = load_response_engine()

    iforest = assets.get("isolation_forest")
    attack_le = assets.get("attack_label_encoder")
    entity_profiles = assets.get("entity_profiles", {})
    dept_profiles = assets.get("department_profiles", {})
    model_comparison = assets.get("model_comparison", {})
    explainability_summary = assets.get("explainability_summary", {})

    # Compute anomaly scores & run risk engine
    risk_df = compute_risk_results(iforest, risk_engine, enriched_df, processed_df)

    # Merge risk results with enriched data
    merged = enriched_df.copy()
    merged["risk_score"] = np.round(risk_df["risk_score"].values, 2)
    merged["risk_level"] = risk_df["risk_level"].values
    merged["risk_contributors"] = risk_df["risk_contributors"].values
    merged["is_anomaly"] = (risk_df["prediction"].values == -1).astype(int)

    # ===================================================================
    # SIDEBAR
    # ===================================================================
    with st.sidebar:
        st.markdown("## 🛡️ UEBA Dashboard")
        st.caption("Enterprise Security Operations Center")
        st.markdown("---")

        page = st.radio(
            "Navigation",
            ["📊 Executive Summary", "🚨 Alert Triage", "👤 Entity Profiler", "🔍 Explainability", "📈 Concept Drift"],
            index=0,
            label_visibility="collapsed",
        )
        # Strip emoji prefix for page matching
        page = page.split(" ", 1)[1]

        st.markdown("---")
        st.markdown("### 🎛️ Filters")

        # Risk score threshold
        risk_threshold = st.slider(
            "Minimum Risk Score", 0, 100, 50, step=5,
            help="Show alerts with risk score >= this value.",
        )

        # Entity type filter
        entity_types = ["All"] + sorted(enriched_df["entity_type"].dropna().unique().tolist())
        selected_entity_type = st.selectbox("Entity Type", entity_types)

        # Department filter
        departments = ["All"] + sorted(enriched_df["department"].dropna().unique().tolist())
        selected_dept = st.selectbox("Department", departments)

        # Attack type filter
        attack_types = ["All"] + sorted(enriched_df[enriched_df["label"] == "Anomaly"]["anomaly_type"].dropna().unique().tolist())
        selected_attack = st.selectbox("Attack Type", attack_types)

        st.markdown("---")
        st.caption(f"📊 {len(enriched_df):,} events · {enriched_df['entity_id'].nunique()} entities")

    # --- Apply Filters ---
    filtered = merged.copy()
    if selected_entity_type != "All":
        filtered = filtered[filtered["entity_type"] == selected_entity_type]
    if selected_dept != "All":
        filtered = filtered[filtered["department"] == selected_dept]
    if selected_attack != "All":
        filtered = filtered[filtered["anomaly_type"] == selected_attack]

    # ===================================================================
    # PAGE: Executive Summary
    # ===================================================================
    if page == "Executive Summary":
        st.markdown("# Executive Threat Summary")
        st.markdown("> **Welcome to the UEBA Dashboard.** This page provides a high-level overview of the current security posture, showing total events analyzed, anomalies detected by the Isolation Forest model, and the global breakdown of attack types.")

        anomaly_count = int(merged["is_anomaly"].sum())
        high_risk = int((merged["risk_score"] >= 75).sum())
        attack_classes = enriched_df[enriched_df["label"] == "Anomaly"]["anomaly_type"].nunique()
        best_model = model_comparison.get("best_model", "LightGBM") if isinstance(model_comparison, dict) else "LightGBM"

        # KPI Row — 4 uniform numeric cards
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="metric-card"><h3>Total Events</h3><p class="metric-value">{len(merged):,}</p></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card"><h3>Anomalies</h3><p class="metric-value">{anomaly_count:,}</p></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-card"><h3>High-Risk Alerts</h3><p class="metric-value">{high_risk:,}</p></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="metric-card"><h3>Attack Classes</h3><p class="metric-value">{attack_classes}</p></div>', unsafe_allow_html=True)

        # Best model banner
        st.markdown(f'<div style="text-align:center;color:#94A3B8;font-size:0.82rem;margin:8px 0 4px 0;">Best Classifier: <strong style="color:#A78BFA;">{best_model}</strong></div>', unsafe_allow_html=True)

        # --- Charts Row 1 ---
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown('<p class="section-header">Attack Type Distribution</p>', unsafe_allow_html=True)
            anomalies = enriched_df[enriched_df["label"] == "Anomaly"]
            attack_counts = anomalies["anomaly_type"].value_counts().reset_index()
            attack_counts.columns = ["Attack Type", "Count"]
            fig_atk = px.bar(
                attack_counts, x="Attack Type", y="Count",
                color="Count",
                color_continuous_scale=["#6366f1", "#a855f7", "#ec4899"],
                template="plotly_dark",
            )
            fig_atk.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=10, b=40),
                height=350,
                showlegend=False,
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_atk, use_container_width=True)

        with col_right:
            st.markdown('<p class="section-header">Risk Score Distribution</p>', unsafe_allow_html=True)
            fig_risk = px.histogram(
                merged, x="risk_score", nbins=50,
                color_discrete_sequence=["#818cf8"],
                template="plotly_dark",
                labels={"risk_score": "Risk Score (0-100)"},
            )
            fig_risk.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=10, b=40),
                height=350,
                showlegend=False,
            )
            st.plotly_chart(fig_risk, use_container_width=True)

        # --- Charts Row 2 ---
        col_l2, col_r2 = st.columns(2)

        with col_l2:
            st.markdown('<p class="section-header">Anomalies by Department</p>', unsafe_allow_html=True)
            dept_anom = merged[merged["is_anomaly"] == 1].groupby("department").size().reset_index(name="Anomalies")
            dept_anom = dept_anom.sort_values("Anomalies", ascending=True)
            fig_dept = px.bar(
                dept_anom, y="department", x="Anomalies", orientation="h",
                color="Anomalies",
                color_continuous_scale=["#22c55e", "#eab308", "#ef4444"],
                template="plotly_dark",
            )
            fig_dept.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=10, b=20, l=10),
                height=400,
                coloraxis_showscale=False,
                yaxis_title="",
            )
            st.plotly_chart(fig_dept, use_container_width=True)

        with col_r2:
            st.markdown('<p class="section-header">Anomaly Timeline (Hourly)</p>', unsafe_allow_html=True)
            hourly = merged[merged["is_anomaly"] == 1].groupby("login_hour").size().reset_index(name="Anomalies")
            fig_time = px.area(
                hourly, x="login_hour", y="Anomalies",
                color_discrete_sequence=["#a855f7"],
                template="plotly_dark",
                labels={"login_hour": "Hour of Day"},
            )
            fig_time.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=10, b=40),
                height=400,
            )
            st.plotly_chart(fig_time, use_container_width=True)

        # --- Global Feature Importance ---
        st.markdown('<p class="section-header">Global Feature Importance (LightGBM)</p>', unsafe_allow_html=True)
        if explainability_summary and "global_top_features" in explainability_summary:
            feat_imp = explainability_summary["global_top_features"]
            feat_df = pd.DataFrame(list(feat_imp.items()), columns=["Feature", "Importance"])
            feat_df = feat_df.sort_values("Importance", ascending=True)
            fig_feat = px.bar(
                feat_df, y="Feature", x="Importance", orientation="h",
                color="Importance",
                color_continuous_scale=["#6366f1", "#c084fc"],
                template="plotly_dark",
            )
            fig_feat.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=10, b=20, l=10),
                height=450,
                coloraxis_showscale=False,
                yaxis_title="",
            )
            st.plotly_chart(fig_feat, use_container_width=True)

    # ===================================================================
    # PAGE: Alert Triage
    # ===================================================================
    elif page == "Alert Triage":
        st.markdown("# Alert Triage & Investigation Console")
        st.markdown("> **Investigate active threats.** Use the filters on the left to drill down into specific alerts. Expand any row to see automated playbook actions and context.")

        alerts = filtered[filtered["risk_score"] >= risk_threshold].copy()
        alerts = alerts.sort_values("risk_score", ascending=False)

        st.markdown(f"Showing **{len(alerts):,}** alerts with risk score >= **{risk_threshold}**")

        if not alerts.empty:
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                alerts.to_excel(writer, index=False, sheet_name='Threat Alerts')
            
            st.download_button(
                label="📥 Download Triage Report (Excel)",
                data=buffer.getvalue(),
                file_name="SOC_Alert_Triage_Report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        if alerts.empty:
            st.info("No alerts match the current filter criteria. Try lowering the risk threshold.")
        else:
            display_cols = [
                "event_id", "entity_id", "entity_type", "department",
                "anomaly_type", "risk_score", "risk_level", "risk_contributors",
                "login_hour", "is_off_hours",
                "geo_location", "resource_accessed", "resource_sensitivity",
                "session_duration_min", "bytes_uploaded", "bytes_downloaded",
                "failed_attempts", "geo_velocity_kmph",
            ]
            existing = [c for c in display_cols if c in alerts.columns]

            st.dataframe(
                alerts[existing].head(500),
                use_container_width=True,
                height=500,
                column_config={
                    "risk_score": st.column_config.ProgressColumn(
                        "Risk Score", min_value=0, max_value=100, format="%d",
                    ),
                },
            )

            # Summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Alerts Shown", f"{len(alerts):,}")
            with col2:
                st.metric("Avg Risk Score", f"{alerts['risk_score'].mean():.1f}")
            with col3:
                unique_entities = alerts["entity_id"].nunique()
                st.metric("Unique Entities", f"{unique_entities:,}")

    # ===================================================================
    # PAGE: Entity Profiler
    # ===================================================================
    elif page == "Entity Profiler":
        st.markdown("# Entity Behavioral Profiler")
        st.markdown("> **Analyze individual entity behavior.** Search for any user or service account to compare their current activity against their historical profile and their peer department baseline.")

        entity_ids = sorted(enriched_df["entity_id"].unique().tolist())
        selected_entity = st.selectbox("Select Entity", entity_ids, index=0)

        entity_events = merged[merged["entity_id"] == selected_entity].copy()
        entity_anomalies = entity_events[entity_events["is_anomaly"] == 1]

        # Entity Info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            etype = entity_events["entity_type"].iloc[0] if len(entity_events) > 0 else "N/A"
            st.metric("Entity Type", etype)
        with col2:
            dept = entity_events["department"].iloc[0] if len(entity_events) > 0 else "N/A"
            st.metric("Department", dept)
        with col3:
            st.metric("Total Events", f"{len(entity_events):,}")
        with col4:
            st.metric("Anomalies", f"{len(entity_anomalies):,}")

        st.markdown("---")

        # Entity Profile vs Department Baseline
        profile = entity_profiles.get(selected_entity, {})
        dept_name = entity_events["department"].iloc[0] if len(entity_events) > 0 else None
        dept_profile = dept_profiles.get(dept_name, {}) if dept_name else {}

        if profile:
            st.markdown('<p class="section-header">Behavioral Profile vs Department Baseline</p>', unsafe_allow_html=True)

            comparison_data = []
            metric_labels = {
                "session_mean": "Avg Session (min)",
                "bytes_up_mean": "Avg Bytes Uploaded",
                "bytes_down_mean": "Avg Bytes Downloaded",
                "failed_ratio": "Failed Login Rate",
            }
            for key, label in metric_labels.items():
                entity_val = profile.get(key, 0)
                dept_val = dept_profile.get(key, 0)
                comparison_data.append({
                    "Metric": label,
                    "Entity": round(entity_val, 2),
                    "Department Avg": round(dept_val, 2),
                    "Deviation (%)": round(((entity_val - dept_val) / dept_val * 100) if dept_val != 0 else 0, 1),
                })

            comp_df = pd.DataFrame(comparison_data)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

            # Radar Chart
            categories = list(metric_labels.values())
            entity_vals = [profile.get(k, 0) for k in metric_labels.keys()]
            dept_vals = [dept_profile.get(k, 0) for k in metric_labels.keys()]

            # Normalise for radar
            max_vals = [max(abs(e), abs(d), 1) for e, d in zip(entity_vals, dept_vals)]
            entity_norm = [e / m for e, m in zip(entity_vals, max_vals)]
            dept_norm = [d / m for d, m in zip(dept_vals, max_vals)]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=entity_norm + [entity_norm[0]],
                theta=categories + [categories[0]],
                name=selected_entity,
                fill="toself",
                fillcolor="rgba(99, 102, 241, 0.2)",
                line=dict(color="#6366f1", width=2),
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=dept_norm + [dept_norm[0]],
                theta=categories + [categories[0]],
                name=f"Dept: {dept_name}",
                fill="toself",
                fillcolor="rgba(168, 85, 247, 0.15)",
                line=dict(color="#a855f7", width=2, dash="dash"),
            ))
            fig_radar.update_layout(
                polar=dict(bgcolor="rgba(0,0,0,0)"),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=400,
                margin=dict(t=30, b=30),
                legend=dict(orientation="h", yanchor="bottom", y=-0.15),
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # Entity Event Timeline
        if not entity_events.empty:
            st.markdown('<p class="section-header">Event Timeline</p>', unsafe_allow_html=True)
            timeline = entity_events.sort_values("timestamp")
            fig_tl = px.scatter(
                timeline, x="timestamp", y="risk_score",
                color="is_anomaly",
                color_discrete_map={0: "#6366f1", 1: "#ef4444"},
                size="risk_score", size_max=12,
                hover_data=["anomaly_type", "resource_accessed", "geo_location"],
                template="plotly_dark",
                labels={"is_anomaly": "Anomaly", "risk_score": "Risk Score"},
            )
            fig_tl.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                height=350,
                margin=dict(t=10, b=40),
            )
            st.plotly_chart(fig_tl, use_container_width=True)

    # ===================================================================
    # PAGE: Explainability
    # ===================================================================
    elif page == "Explainability":
        st.markdown("# Alert Explainability & SHAP Analysis")
        st.markdown("> **Understand the 'Why'.** Select a high-risk alert to see a simple English analyst report and the exact SHAP feature contributions that caused the AI to flag the event.")

        # Find anomalous events
        anomalous_events = merged[merged["is_anomaly"] == 1].copy()
        anomalous_events = anomalous_events.sort_values("risk_score", ascending=False)

        if anomalous_events.empty:
            st.info("No anomalies detected to explain.")
        else:
            # Let analyst pick an alert to investigate
            top_alerts = anomalous_events.head(100)
            alert_options = [
                f"{row['event_id']} | {row['entity_id']} | Risk: {row['risk_score']} ({row.get('risk_level', 'Unknown')})"
                for _, row in top_alerts.iterrows()
            ]
            selected_alert = st.selectbox("Select Alert to Investigate", alert_options, index=0)

            # Parse selected
            selected_event_id = selected_alert.split(" | ")[0]
            alert_row = merged[merged["event_id"].astype(str) == selected_event_id]

            if not alert_row.empty:
                row = alert_row.iloc[0]

                # Alert context
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Entity", row["entity_id"])
                with col2:
                    st.metric("Attack Type", row.get("anomaly_type", "Unknown"))
                with col3:
                    st.metric("Risk Score", f"{row['risk_score']:.0f} ({row.get('risk_level', 'Unknown')})")
                with col4:
                    st.metric("Department", row.get("department", "N/A"))
                    
                st.markdown("**Risk Contributors:**")
                st.info(row.get("risk_contributors", "None identified"))
                
                st.markdown("**Automated SOAR Actions:**")
                attack_type = str(row.get("anomaly_type", "Normal"))
                if attack_type == "nan": attack_type = "Normal"
                
                risk_lvl = row.get("risk_level", "Low")
                actions = response_engine.get_actions(attack_type, risk_lvl)
                if actions:
                    for action in actions:
                        st.markdown(f"- 🚀 `{action}`")
                else:
                    st.markdown("- _No automated actions defined._")

                st.markdown("---")

                # Get processed features for this event
                event_idx = alert_row.index[0]
                feature_cols = [c for c in processed_df.columns if c not in ("label", "anomaly_type", "entity_id")]
                event_features = processed_df.loc[[event_idx], feature_cols].copy()
                event_features = event_features.replace([np.inf, -np.inf], np.nan).fillna(0)

                attack_type = str(row.get("anomaly_type", ""))
                if attack_type == "Normal" or attack_type == "nan":
                    attack_type = None

                explanation = explainer.explain_event(
                    event_features,
                    top_k=10,
                    predicted_attack_type=attack_type,
                )

                # Generate Analyst Summary
                st.markdown('<p class="section-header">Analyst Report</p>', unsafe_allow_html=True)
                top_features_list = explanation.get("top_features", [])
                if top_features_list:
                    top_1 = top_features_list[0]["human_name"].lower()
                    reason_text = f"primarily due to anomalous **{top_1}**"
                    if len(top_features_list) > 1:
                        top_2 = top_features_list[1]["human_name"].lower()
                        reason_text += f", combined with unusual **{top_2}**"
                    
                    atk_label = str(row.get("anomaly_type", "General Anomaly"))
                    if atk_label == "nan" or atk_label == "": atk_label = "General Anomaly"
                    
                    rsk_label = row.get("risk_level", "Unknown")
                    
                    analyst_summary = (
                        f"This event was flagged {reason_text}. "
                        f"The behavioral pattern strongly correlates with a **{atk_label}** attack and poses a **{rsk_label}** risk to the organization. "
                        f"Review the detailed SHAP contributions and automated SOAR actions to proceed with mitigation."
                    )
                    st.info(analyst_summary)

                # Narrative (Detailed Breakdown)
                st.markdown('<p class="section-header">Detailed Investigation Narrative</p>', unsafe_allow_html=True)
                narrative = explanation.get("narrative", "No explanation available.")
                st.markdown(f'<div class="narrative-box">{narrative}</div>', unsafe_allow_html=True)

                # Feature contribution chart
                st.markdown('<p class="section-header">Feature Contribution Breakdown</p>', unsafe_allow_html=True)
                top_features = explanation.get("top_features", [])
                if top_features:
                    contrib_df = pd.DataFrame(top_features)
                    contrib_df = contrib_df.sort_values("shap_value", ascending=True)

                    colors = ["#ef4444" if v > 0 else "#22c55e" for v in contrib_df["shap_value"]]
                    fig_shap = go.Figure(go.Bar(
                        x=contrib_df["shap_value"],
                        y=contrib_df["human_name"],
                        orientation="h",
                        marker_color=colors,
                        text=[f"{v:+.4f}" for v in contrib_df["shap_value"]],
                        textposition="auto",
                    ))
                    fig_shap.update_layout(
                        template="plotly_dark",
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        height=max(250, len(top_features) * 40),
                        margin=dict(t=10, b=30, l=10),
                        xaxis_title="SHAP Value (Impact on Prediction)",
                        yaxis_title="",
                    )
                    st.plotly_chart(fig_shap, use_container_width=True)

                # Event details table
                st.markdown('<p class="section-header">Raw Event Details</p>', unsafe_allow_html=True)
                detail_cols = [
                    "event_id", "entity_id", "entity_type", "department",
                    "timestamp", "login_hour", "is_off_hours", "geo_location",
                    "resource_accessed", "resource_type", "resource_sensitivity",
                    "auth_method", "login_status", "session_duration_min",
                    "bytes_uploaded", "bytes_downloaded", "failed_attempts",
                    "geo_velocity_kmph", "new_device", "new_location",
                    "protocol", "os", "browser", "privilege_level",
                ]
                existing = [c for c in detail_cols if c in alert_row.columns]
                st.dataframe(alert_row[existing].T.rename(columns={event_idx: "Value"}), use_container_width=True)
    # ===================================================================
    # PAGE: Concept Drift
    # ===================================================================
    elif page == "Concept Drift":
        st.markdown("## System Health & Concept Drift")
        st.markdown("> **Monitor AI Reliability.** This page tracks whether the statistical behavior of the network is shifting over time. If a distribution shift is detected (Concept Drift), it alerts the team that the Machine Learning models may need retraining.")

        
        drift_model_path = PROJECT_ROOT / "saved_models/drift_baseline.joblib"
        if not drift_model_path.exists():
            st.warning("Drift baseline not found. Run the batch pipeline first to fit the drift detector.")
        else:
            drift_detector = joblib.load(str(drift_model_path))
            
            st.info(f"Tracking **{len(drift_detector.tracked_features)}** behavioral features using the Kolmogorov-Smirnov (KS) test (p-value threshold = {drift_detector.p_value_threshold}).")
            
            with st.spinner("Detecting drift on current dataset..."):
                # Pass the features with risk_score merged in
                current_df = processed_df.copy()
                current_df["risk_score"] = merged["risk_score"].values
                drift_results = drift_detector.detect(current_df)
                
            drifting_features = sum(1 for res in drift_results.values() if res["is_drifting"])
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="metric-card"><h3>Features Tracked</h3><p class="metric-value">{}</p></div>'.format(len(drift_results)), unsafe_allow_html=True)
            with col2:
                status_color = "#ef4444" if drifting_features > 0 else "#22c55e"
                st.markdown(f'<div class="metric-card"><h3>Drifting Features</h3><p class="metric-value" style="color: {status_color}">{drifting_features}</p></div>', unsafe_allow_html=True)
                
            if drifting_features > 0:
                st.error("⚠️ Concept drift detected! Model retraining is recommended.")
            else:
                st.success("✅ No significant distribution shifts detected. Model is healthy.")
                
            st.markdown('<p class="section-header">Drift Analysis Details</p>', unsafe_allow_html=True)
            
            # Format results into a dataframe
            drift_rows = []
            for feat, metrics in drift_results.items():
                drift_rows.append({
                    "Feature": feat,
                    "Reference Mean": f"{metrics['reference_mean']:.4f}",
                    "Current Mean": f"{metrics['current_mean']:.4f}",
                    "KS Statistic": f"{metrics['ks_statistic']:.4f}",
                    "p-value": f"{metrics['p_value']:.4e}",
                    "Drifting?": "❌ Yes" if metrics["is_drifting"] else "✅ No"
                })
                
            drift_df = pd.DataFrame(drift_rows)
            st.dataframe(drift_df, use_container_width=True)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
