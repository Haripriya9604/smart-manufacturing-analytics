import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np

from streamlit_app_helpers import (
    build_features,
    load_model,
    auto_select_model,
    predict_rul,
    add_confidence_interval,
    add_failure_date,
)

# =====================================================
# Page Config
# =====================================================
st.set_page_config(
    page_title="Industrial Control Room – RUL Monitoring",
    layout="wide",
)

st.title("🏭 Industrial Control Room – Predictive Maintenance Dashboard")
st.caption(
    "Fleet-level Remaining Useful Life (RUL) monitoring, risk prioritization, "
    "and maintenance planning"
)

# =====================================================
# File Upload
# =====================================================
uploaded = st.file_uploader(
    "Upload CMAPSS or compatible sensor CSV",
    type=["csv"],
)

if not uploaded:
    st.info("⬆️ Upload a dataset to start monitoring fleet health")
    st.stop()

df = pd.read_csv(uploaded)

st.subheader("📄 Uploaded Data Preview")
st.dataframe(df.head(), use_container_width=True)

# =====================================================
# Feature Engineering
# =====================================================
df_feat = build_features(df)

# =====================================================
# Model Selection & Prediction
# =====================================================
model_path = auto_select_model(df_feat)
model_type, model, feats = load_model(model_path)

df_feat["RUL_pred"] = predict_rul(df_feat, model_type, model, feats)

# =====================================================
# Confidence Interval
# =====================================================
df_feat = add_confidence_interval(
    df_feat,
    pred_col="RUL_pred",
    sigma=12,     # validated residual std
    z=1.96        # 95% CI
)

# =====================================================
# Latest state per engine
# =====================================================
latest = (
    df_feat.sort_values("cycle")
    .groupby("unit", as_index=False)
    .tail(1)
    .copy()
)

# =====================================================
# Failure Date Prediction
# =====================================================
latest = add_failure_date(latest)

# =====================================================
# Severity Classification
# =====================================================
def severity_bucket(rul):
    if rul < 0:
        return "Overdue"
    elif rul <= 30:
        return "Critical"
    elif rul <= 80:
        return "Warning"
    else:
        return "Healthy"

latest["Severity"] = latest["RUL_pred"].apply(severity_bucket)

# =====================================================
# KPI SUMMARY
# =====================================================
latest["RUL_safe"] = latest["RUL_pred"].clip(lower=0)

avg_rul = latest["RUL_safe"].mean()
overdue_count = (latest["RUL_pred"] < 0).sum()
severity_counts = latest["Severity"].value_counts()

st.subheader("⏳ Fleet Health Summary")

k1, k2, k3, k4 = st.columns(4)

k1.metric("Average Remaining Life", f"{avg_rul:.0f} cycles")
k2.metric("⚠️ Engines Overdue", int(overdue_count))
k3.metric("Critical (≤30 cycles)", int(severity_counts.get("Critical", 0)))
k4.metric("Healthy Engines", int(severity_counts.get("Healthy", 0)))

st.info(
    f"""
**How to interpret this summary:**

• Fleet can operate **~{avg_rul:.0f} more cycles on average**  
• **{overdue_count} engines** are already beyond predicted failure  
• *Critical* engines require urgent maintenance  
• *Healthy* engines are safe for now
"""
)

# =====================================================
# FAILURE WINDOW KPI
# =====================================================
st.subheader("🗓 Predicted Failure Window")

soonest = latest.sort_values("Days_to_Failure").iloc[0]

f1, f2, f3 = st.columns(3)

f1.metric("Nearest Failure Date", soonest["Failure_Date"].strftime("%d %b %Y"))
f2.metric("Highest Risk Engine", f"Engine {int(soonest['unit'])}")
f3.metric("Days Remaining", f"{int(soonest['Days_to_Failure'])} days")

st.warning(
    f"""
**Failure Outlook**

• Engine **{int(soonest['unit'])}** is expected to fail first  
• Estimated date: **{soonest['Failure_Date'].strftime('%d %b %Y')}**  
• Maintenance should be completed **before this date**
"""
)

# =====================================================
# RUL DEGRADATION + CONFIDENCE BAND
# =====================================================
st.subheader("📉 RUL Degradation with Uncertainty")

fig, ax = plt.subplots(figsize=(12, 4))

for _, g in df_feat.groupby("unit"):
    ax.plot(g["cycle"], g["RUL_pred"], alpha=0.25, color="steelblue")
    ax.fill_between(
        g["cycle"],
        g["RUL_lower"],
        g["RUL_upper"],
        color="steelblue",
        alpha=0.08,
    )

ax.axhspan(0, 30, color="red", alpha=0.1, label="Immediate Maintenance")
ax.axhspan(30, 80, color="orange", alpha=0.1, label="Planned Maintenance")
ax.axhline(0, color="red", linestyle="--")

ax.set_xlabel("Cycle")
ax.set_ylabel("Predicted RUL (cycles)")
ax.set_title("Remaining Useful Life with Confidence Interval")
ax.legend(loc="upper right")

st.pyplot(fig)

st.caption(
    """
• Solid line = predicted RUL  
• Shaded band = 95% confidence interval  
• Wider band = higher uncertainty  
• Red zone indicates immediate maintenance risk
"""
)

# =====================================================
# INTERACTIVE FAILURE TIMELINE (GANTT)
# =====================================================
st.subheader("📆 Fleet Failure Timeline (Maintenance Planning View)")

timeline_df = latest.copy()
timeline_df["Start"] = pd.Timestamp.today().normalize()
timeline_df["Failure_Date"] = pd.to_datetime(timeline_df["Failure_Date"])
timeline_df["Engine"] = timeline_df["unit"].astype(str)

timeline_df = timeline_df[timeline_df["Failure_Date"] >= timeline_df["Start"]]

if timeline_df.empty:
    st.warning("No upcoming failures detected in the selected horizon.")
else:
    fig_gantt = px.timeline(
        timeline_df,
        x_start="Start",
        x_end="Failure_Date",
        y="Engine",
        color="Severity",
        color_discrete_map={
            "Overdue": "red",
            "Critical": "orange",
            "Warning": "gold",
            "Healthy": "green",
        },
        hover_data={
            "RUL_pred": ":.1f",
            "Days_to_Failure": True,
            "Failure_Date": True,
        },
    )

    fig_gantt.update_layout(
        height=600,
        xaxis_title="Calendar Date",
        yaxis_title="Engine ID",
        title="Predicted Engine Failure Timeline",
    )

    fig_gantt.update_yaxes(autorange="reversed")

    st.plotly_chart(fig_gantt, use_container_width=True)

st.caption(
    """
Each bar represents one engine.
Shorter bars indicate urgent maintenance priority.
"""
)

# =====================================================
# TOP RISK TABLE
# =====================================================
st.subheader("🚨 Top Risk Engines – Maintenance Priority")

action_map = {
    "Overdue": "STOP immediately",
    "Critical": "Schedule maintenance now",
    "Warning": "Monitor closely",
    "Healthy": "No action required",
}

latest["Recommended Action"] = latest["Severity"].map(action_map)

risk_table = (
    latest.sort_values("RUL_pred")
    .loc[:, [
        "unit",
        "RUL_pred",
        "RUL_lower",
        "RUL_upper",
        "Days_to_Failure",
        "Failure_Date",
        "Severity",
        "Recommended Action",
    ]]
    .rename(columns={
        "unit": "Engine ID",
        "RUL_pred": "Predicted RUL",
        "RUL_lower": "Lower Bound",
        "RUL_upper": "Upper Bound",
    })
)

st.dataframe(
    risk_table.head(15),
    use_container_width=True,
    hide_index=True,
)

st.success(
    """
### 🧠 Operational Insight

• Predictions include **uncertainty bounds**, not guesses  
• Failure dates convert ML output into **real maintenance schedules**  
• Prioritize **Overdue → Critical → Warning** engines  

This dashboard reflects **OEM-grade predictive maintenance systems**.
"""
)
