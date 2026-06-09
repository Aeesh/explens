import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(
    page_title="ExpLens",
    page_icon="🔭",
    layout="wide"
)

with st.sidebar:
    st.title("🔭 ExpLens")
    st.markdown(
        "Automated ML experiment narratives with "
        "**consistency verification**."
    )
    st.markdown("---")
    st.markdown("**How it works**")
    st.markdown("1. Connect to a WandB run or upload a log file")
    st.markdown("2. Facts are extracted from your metrics")
    st.markdown("3. An LLM generates a structured narrative")
    st.markdown("4. Claims are verified against your actual numbers")
    st.markdown("5. Download the report as markdown")
    st.markdown("---")
    st.markdown("[💻 GitHub](https://github.com/aeesh/explens)")

