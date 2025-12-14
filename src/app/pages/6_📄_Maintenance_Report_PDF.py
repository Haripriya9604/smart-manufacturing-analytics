import streamlit as st
import pandas as pd
import os
from fpdf import FPDF

from streamlit_app_helpers import (
    build_features,
    load_model,
    auto_select_model,
    predict_rul,
)

# ===============================
# Helpers (CRITICAL)
# ===============================
def safe_text(text: str) -> str:
    replacements = {
        "–": "-",
        "—": "-",
        "’": "'",
        "“": '"',
        "”": '"',
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text


def pdf_multiline(pdf, text, line_height=7):
    """Safe wrapper for multi_cell"""
    usable_width = pdf.w - pdf.l_margin - pdf.r_margin
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(
        usable_width,
        line_height,
        safe_text(text)
    )


# ===============================
# Page setup
# ===============================
st.set_page_config(layout="wide")
st.title("📄 Auto-Generated Maintenance Report")
st.caption("Enterprise-grade PDF report for predictive maintenance planning")

# ===============================
# File upload
# ===============================
uploaded = st.file_uploader(
    "Upload CMAPSS or compatible sensor CSV",
    type=["csv"]
)
if not uploaded:
    st.info("⬆️ Upload a dataset to generate a maintenance report")
    st.stop()

df = pd.read_csv(uploaded)

# ===============================
# Feature engineering & prediction
# ===============================
df_feat = build_features(df)

model_path = auto_select_model(df_feat)
model_type, model, feats = load_model(model_path)

df_feat["RUL_pred"] = predict_rul(df_feat, model_type, model, feats)

latest = (
    df_feat.sort_values("cycle")
    .groupby("unit", as_index=False)
    .tail(1)
    .copy()
)

latest["RUL_safe"] = latest["RUL_pred"].clip(lower=0)

# ===============================
# Severity & actions
# ===============================
def severity_bucket(rul):
    if rul < 0:
        return "OVERDUE"
    elif rul <= 30:
        return "CRITICAL"
    elif rul <= 80:
        return "WARNING"
    return "HEALTHY"


action_map = {
    "OVERDUE": "STOP IMMEDIATELY",
    "CRITICAL": "SCHEDULE MAINTENANCE NOW",
    "WARNING": "MONITOR CLOSELY",
    "HEALTHY": "NO ACTION REQUIRED",
}

latest["Severity"] = latest["RUL_pred"].apply(severity_bucket)
latest["Action"] = latest["Severity"].map(action_map)

# ===============================
# PDF generation
# ===============================
if st.button("📄 Generate Maintenance PDF Report"):

    # -------- Font path --------
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    FONT_PATH = os.path.join(BASE_DIR, "assets", "DejaVuSans.ttf")

    if not os.path.exists(FONT_PATH):
        st.error("❌ Font file not found: src/app/assets/DejaVuSans.ttf")
        st.stop()

    # -------- PDF setup --------
    pdf = FPDF()
    pdf.add_page()
    pdf.add_font("DejaVu", "", FONT_PATH, uni=True)

    # -------- Title --------
    pdf.set_font("DejaVu", size=14)
    pdf.cell(0, 10, "Predictive Maintenance Report", ln=True)
    pdf.ln(3)

    # -------- Fleet summary --------
    avg_rul = latest["RUL_safe"].mean()
    overdue = (latest["RUL_pred"] < 0).sum()
    critical = (latest["Severity"] == "CRITICAL").sum()
    healthy = (latest["Severity"] == "HEALTHY").sum()

    pdf.set_font("DejaVu", size=10)

    summary_text = f"""
Fleet Health Summary
--------------------
Average Remaining Life : {avg_rul:.0f} cycles
Engines Overdue        : {overdue}
Critical Engines       : {critical}
Healthy Engines        : {healthy}

Interpretation:
- Overdue engines have exceeded predicted failure limits
- Critical engines require immediate maintenance planning
- Healthy engines are safe to operate
""".strip()

    pdf_multiline(pdf, summary_text)
    pdf.ln(4)

    # -------- Engine-level actions --------
    pdf.set_font("DejaVu", size=11)
    pdf.cell(0, 8, "Engine-Level Maintenance Actions", ln=True)
    pdf.ln(2)

    pdf.set_font("DejaVu", size=9)

    for _, row in latest.sort_values("RUL_pred").iterrows():
        engine_text = (
            f"Engine {int(row['unit'])} | "
            f"RUL: {row['RUL_pred']:.1f} cycles | "
            f"Status: {row['Severity']} | "
            f"Action: {row['Action']}"
        )
        pdf_multiline(pdf, engine_text, line_height=6)

    # -------- Save & download --------
    output_path = "maintenance_report.pdf"
    pdf.output(output_path)

    with open(output_path, "rb") as f:
        st.download_button(
            "⬇️ Download Maintenance Report (PDF)",
            f,
            file_name="maintenance_report.pdf",
            mime="application/pdf",
        )

    st.success("✅ Maintenance report generated successfully")
