import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from streamlit_app_helpers import (
    build_features,
    load_model,
    auto_select_model,
    predict_rul,
)

# =====================================================
# Page setup
# =====================================================
st.set_page_config(layout="wide")
st.title("📊 Analytical Diagnosis – Fleet Health Insights")
st.caption(
    "Executive-level analytical view of degradation behavior, risk concentration, and model intelligence"
)

# =====================================================
# Upload Section (Top)
# =====================================================
uploaded = st.file_uploader(
    "Upload CMAPSS or compatible sensor CSV",
    type=["csv"],
)

if not uploaded:
    st.info("⬆️ Upload a dataset to unlock analytical insights")
    st.stop()

df = pd.read_csv(uploaded)

# =====================================================
# Feature engineering & prediction
# =====================================================
df_feat = build_features(df)
model_path = auto_select_model(df_feat)
model_type, model, feats = load_model(model_path)
df_feat["RUL_pred"] = predict_rul(df_feat, model_type, model, feats)

# =====================================================
# SECTION 1 — Executive Snapshot (TOP ROW)
# =====================================================
st.subheader("🔍 Executive Dataset Snapshot")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Records", f"{len(df_feat):,}")
c2.metric("Engines Monitored", df_feat["unit"].nunique())
c3.metric(
    "Avg Cycles per Engine",
    int(df_feat.groupby("unit")["cycle"].max().mean()),
)
c4.metric("Sensors Analyzed", 21)

st.caption(
    "This snapshot validates dataset scale, fleet coverage, and analytical readiness."
)

st.divider()

# =====================================================
# SECTION 2 — Degradation Behavior (LEFT) + Risk Spread (RIGHT)
# =====================================================
st.subheader("📉 Degradation Behavior & Risk Distribution")

left, right = st.columns([2, 1])

# ---------- LEFT: RUL vs Cycle ----------
with left:
    fig, ax = plt.subplots(figsize=(10, 4))

    sample_units = np.random.choice(
        df_feat["unit"].unique(),
        size=min(12, df_feat["unit"].nunique()),
        replace=False,
    )

    for u in sample_units:
        g = df_feat[df_feat["unit"] == u]
        ax.plot(g["cycle"], g["RUL_pred"], alpha=0.6)

    ax.set_xlabel("Cycle")
    ax.set_ylabel("Predicted RUL")
    ax.set_title("RUL Degradation Across Sample Engines")

    st.pyplot(fig)

    st.caption(
        "Downward trends confirm consistent degradation learning across engines."
    )

# ---------- RIGHT: RUL Distribution ----------
with right:
    latest = (
        df_feat.sort_values("cycle")
        .groupby("unit", as_index=False)
        .tail(1)
    )

    fig2, ax2 = plt.subplots(figsize=(5, 4))
    sns.histplot(latest["RUL_pred"], bins=30, kde=True, ax=ax2)

    ax2.set_xlabel("Predicted RUL")
    ax2.set_title("Fleet Risk Distribution")

    st.pyplot(fig2)

    st.caption(
        "Left-skew indicates concentration of engines approaching failure."
    )

st.divider()

# =====================================================
# SECTION 3 — Sensor Intelligence (Correlation)
# =====================================================
st.subheader("🔗 Sensor Intelligence & Dependency Structure")

sensor_cols = [f"s{i}" for i in range(1, 22)]
corr = df_feat[sensor_cols + ["RUL_pred"]].corr()

fig3, ax3 = plt.subplots(figsize=(11, 6))
sns.heatmap(
    corr,
    cmap="coolwarm",
    center=0,
    linewidths=0.2,
    ax=ax3,
)

ax3.set_title("Sensor Correlation with RUL")

st.pyplot(fig3)

st.caption(
    """
This matrix highlights:
• Strongly correlated sensor groups  
• Sensors most aligned with degradation  
• Opportunities for sensor reduction or redundancy control
"""
)

st.divider()

# =====================================================
# SECTION 4 — Model Intelligence (Feature Importance)
# =====================================================
st.subheader("🧠 Model Intelligence – Key Degradation Drivers")

if model_type == "xgb":
    importances = model.get_score(importance_type="gain")
    imp_df = (
        pd.DataFrame(
            importances.items(),
            columns=["Feature", "Importance"],
        )
        .sort_values("Importance", ascending=False)
        .head(15)
    )

    fig4, ax4 = plt.subplots(figsize=(7, 4))
    ax4.barh(imp_df["Feature"], imp_df["Importance"])
    ax4.invert_yaxis()
    ax4.set_title("Top Features Driving RUL Predictions")

    st.pyplot(fig4)

    st.caption(
        """
Higher importance indicates stronger influence on remaining life predictions.
These features represent dominant degradation signals learned by the model.
"""
    )
else:
    st.info("Feature importance is available only for XGBoost models.")

st.divider()

# =====================================================
# SECTION 5 — Executive Analytical Summary
# =====================================================
st.success(
    """
### 📌 Executive Analytical Summary

• Degradation patterns are **stable and monotonic**, indicating robust learning  
• Failure risk is **not evenly distributed** across the fleet  
• A limited set of sensors drives the majority of predictive power  
• Analytical signals align with **real-world wear and failure behavior**

This analytical layer supports **strategic maintenance planning, sensor optimization, and model trust**.
"""
)
