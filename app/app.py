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


def _run_analysis(run):
    """Shared analysis and display logic."""
    import tempfile

    with st.spinner("Extracting facts..."):
        from src.analysis.extractor import extract_facts
        from src.analysis.patterns import detect_patterns
        facts = extract_facts(run)
        patterns = detect_patterns(facts)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Epochs", run.n_epochs)
    with col2:
        if facts.best_val_loss:
            st.metric("Best Val Loss", f"{facts.best_val_loss:.4f}")
    with col3:
        if facts.best_val_accuracy:
            st.metric("Best Val Acc", f"{facts.best_val_accuracy:.1%}")
    with col4:
        st.metric("Overfitting", facts.overfit_severity.title())

    with st.spinner("Generating narrative..."):
        from src.narrator.generator import ExperimentNarrator
        narrator = ExperimentNarrator()
        result = narrator.generate(facts)

    # Show consistency status prominently
    if result.overall_consistent():
        st.success("✅ All verifiable claims are consistent with measured data")
    else:
        failed = result.failed_checks()
        st.warning(f"⚠️ {len(failed)} claim(s) flagged — see Consistency tab")

    report_tab, patterns_tab, facts_tab, consistency_tab = st.tabs([
        "📄 Report", "🔍 Patterns", "📊 Facts", "✓ Consistency"
    ])

    with report_tab:
        if result.overview:
            st.markdown("### Overview")
            st.markdown(result.overview)
        if result.training_dynamics:
            st.markdown("### Training Dynamics")
            st.markdown(result.training_dynamics)
        if result.generalisation:
            st.markdown("### Generalisation")
            st.markdown(result.generalisation)
        if result.recommendations:
            st.markdown("### Recommendations")
            st.markdown(result.recommendations)
        if result.what_next:
            st.markdown("### What to Try Next")
            st.markdown(result.what_next)

        # Charts
        with tempfile.TemporaryDirectory() as tmpdir:
            from src.report.charts import plot_loss_curves, plot_learning_rate
            lc_path = plot_loss_curves(run, tmpdir)
            st.image(lc_path, use_column_width=True)
            lr_path = plot_learning_rate(run, tmpdir)
            if lr_path:
                st.image(lr_path, use_column_width=True)

        # Download button
        with tempfile.TemporaryDirectory() as tmpdir:
            from src.report.builder import build_report
            report_path = build_report(result, tmpdir)
            with open(report_path) as f:
                report_content = f.read()
        st.download_button(
            "📥 Download Report (Markdown)",
            data=report_content,
            file_name=f"{run.run_name}_report.md",
            mime="text/markdown"
        )

    with patterns_tab:
        if not patterns:
            st.info("No significant patterns detected.")
        for p in patterns:
            icon = {"critical": "🔴", "warning": "🟡", "info": "🟢"}.get(p.severity, "•")
            with st.expander(f"{icon} {p.title}", expanded=p.severity == "critical"):
                st.markdown(f"**Finding:** {p.description}")
                st.markdown(f"**Recommendation:** {p.recommendation}")

    with facts_tab:
        st.markdown("These are the verified facts extracted from your metrics.")
        for fact in facts.facts:
            st.markdown(f"- **{fact.key}**: {fact.description}")

    with consistency_tab:
        st.markdown(
            "ExpLens automatically checks whether the LLM's narrative "
            "is consistent with your measured metrics. "
            "This is the core technical contribution of this tool."
        )
        for section, report in result.consistency_checks.items():
            with st.expander(
                f"{section.replace('_', ' ').title()} "
                f"({'✅' if report.overall_consistent else '⚠️'})"
            ):
                st.markdown(report.summary())
