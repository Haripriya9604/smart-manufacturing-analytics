import streamlit as st
import pandas as pd

from streamlit_app_helpers import build_features, load_model, predict_rul

st.set_page_config(layout="wide")
st.title("🎯 Model Comparison – FD vs Global")

uploaded = st.file_uploader("Upload CMAPSS CSV", type=["csv"])
if not uploaded:
    st.stop()

df = pd.read_csv(uploaded)
df_feat = build_features(df)

fd_model = load_model("rul_model_FD004.joblib")
global_model = load_model("rul_model_global.joblib")

df_feat["RUL_FD"] = predict_rul(df_feat, *fd_model)
df_feat["RUL_Global"] = predict_rul(df_feat, *global_model)

latest = df_feat.sort_values("cycle").groupby("unit").tail(1)

comparison = latest[
    ["unit", "RUL_FD", "RUL_Global"]
].rename(columns={"unit": "Engine ID"})

st.dataframe(comparison, use_container_width=True)

st.info(
    """
**Why this matters:**
• FD-specific models adapt to operating conditions  
• Global models generalize but may miss nuances  
• This comparison justifies model selection strategy
"""
)
