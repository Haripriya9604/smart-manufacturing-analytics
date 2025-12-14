import streamlit as st

st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔧 Predictive Maintenance – CMAPSS RUL System")
st.markdown("""
Welcome to the **Predictive Maintenance Dashboard**.

Use the sidebar to navigate:
- 🏭 **Control Room Dashboard** → Fleet overview  
- 🔍 **Engine Drill-Down** → Inspect a single engine  
- 🧠 **Model Explainability** → SHAP insights and feature contributions  
""")
