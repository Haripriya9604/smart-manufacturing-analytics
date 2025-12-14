import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from streamlit_app_helpers import build_features, load_model, auto_select_model, predict_rul

st.set_page_config(layout="wide")
st.title("🔍 Engine Drill-Down Analysis")
st.caption("Detailed lifecycle and failure risk for an individual engine")

uploaded = st.file_uploader("Upload CMAPSS CSV", type=["csv"])
if not uploaded:
    st.stop()

df = pd.read_csv(uploaded)
df_feat = build_features(df)

model_path = auto_select_model(df_feat)
model_type, model, feats = load_model(model_path)
df_feat["RUL_pred"] = predict_rul(df_feat, model_type, model, feats)

engine_id = st.selectbox(
    "Select Engine ID",
    sorted(df_feat["unit"].unique())
)

engine_df = df_feat[df_feat["unit"] == engine_id]

latest_rul = engine_df.sort_values("cycle").iloc[-1]["RUL_pred"]

st.metric(
    "Current Predicted RUL",
    f"{latest_rul:.1f} cycles",
)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(engine_df["cycle"], engine_df["RUL_pred"], label="Predicted RUL")
ax.axhline(0, color="red", linestyle="--")
ax.axhline(30, color="orange", linestyle="--")
ax.set_xlabel("Cycle")
ax.set_ylabel("RUL")
ax.set_title(f"Engine {engine_id} – Remaining Useful Life")
ax.legend()

st.pyplot(fig)

if latest_rul < 0:
    st.error("🚨 This engine has exceeded its predicted failure point. STOP immediately.")
elif latest_rul <= 30:
    st.warning("⚠️ This engine is in a critical state. Schedule maintenance now.")
else:
    st.success("✅ This engine is operating within safe limits.")
