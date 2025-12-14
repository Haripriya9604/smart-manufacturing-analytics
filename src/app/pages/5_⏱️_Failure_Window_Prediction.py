import streamlit as st
import pandas as pd

from streamlit_app_helpers import build_features, load_model, auto_select_model, predict_rul

st.set_page_config(layout="wide")
st.title("⏱️ Failure Window Prediction")
st.caption("When will engines likely fail based on current usage rate")

uploaded = st.file_uploader("Upload CMAPSS CSV", type=["csv"])
if not uploaded:
    st.stop()

df = pd.read_csv(uploaded)
df_feat = build_features(df)

model_path = auto_select_model(df_feat)
model_type, model, feats = load_model(model_path)
df_feat["RUL_pred"] = predict_rul(df_feat, model_type, model, feats)

latest = df_feat.sort_values("cycle").groupby("unit").tail(1)

cycles_per_day = st.slider(
    "Estimated operating cycles per day",
    min_value=1,
    max_value=200,
    value=50,
)

latest["Days_to_Failure"] = latest["RUL_pred"] / cycles_per_day

st.subheader("📅 Estimated Failure Window")

display = latest[
    ["unit", "RUL_pred", "Days_to_Failure"]
].rename(columns={
    "unit": "Engine ID",
    "RUL_pred": "Remaining Cycles",
})

st.dataframe(display, use_container_width=True)

st.info(
    """
**Interpretation:**
• Negative days → engine already overdue  
• Low days → urgent maintenance  
• Higher days → schedule planning possible
"""
)
