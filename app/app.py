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

st.title("ExpLens — ML Experiment Report Generator")
st.markdown(
    "Generate a structured, factually verified narrative from your "
    "training run. Paste a WandB run path or upload a log file."
)

tab1, tab2 = st.tabs(["WandB Run", "Upload Log File"])

with tab1:
    run_path = st.text_input(
        "WandB run path:",
        placeholder="entity/project/run_id",
        help="Find this in your WandB dashboard URL"
    )
    generate_wandb = st.button("Generate Report", key="wandb_btn",
                               type="primary")

    if generate_wandb and run_path.strip():
        with st.spinner("Loading run data..."):
            try:
                from src.connectors.wandb_connector import load_run
                run = load_run(run_path.strip())
                st.success(f"Loaded: {run.run_name} ({run.n_steps} steps)")
            except Exception as e:
                st.error(f"Failed to load run: {e}")
                st.stop()

        _run_analysis(run)

with tab2:
    uploaded = st.file_uploader("Upload training log (CSV or JSON)")
    run_name = st.text_input("Run name:", placeholder="my_experiment")
    generate_local = st.button("Generate Report", key="local_btn",
                               type="primary")

    if generate_local and uploaded:
        import tempfile, os
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(uploaded.name)[1]
        ) as f:
            f.write(uploaded.read())
            tmp_path = f.name

        with st.spinner("Loading..."):
            try:
                if uploaded.name.endswith(".csv"):
                    from src.connectors.local_connector import load_from_csv
                    run = load_from_csv(tmp_path, run_name=run_name or "local_run")
                else:
                    from src.connectors.local_connector import load_from_json
                    run = load_from_json(tmp_path)
                st.success(f"Loaded: {run.run_name}")
            except Exception as e:
                st.error(f"Failed to load file: {e}")
                st.stop()

        _run_analysis(run)