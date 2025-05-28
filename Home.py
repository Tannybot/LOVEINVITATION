import streamlit as st

st.set_page_config(
    page_title="Decision Tree Analysis",
    page_icon="🌳",
    layout="wide"
)

st.title("🌳 Decision Tree Analysis Dashboard")
st.sidebar.success("Select a page above.")

st.markdown("""
## Welcome to the Decision Tree Analysis Dashboard!

This application provides a comprehensive interface for:
1. 📝 Project Description & Documentation
2. 🤖 Model Training & Evaluation
3. 📊 Visualization & Insights

Please select a page from the sidebar to begin.
""")
