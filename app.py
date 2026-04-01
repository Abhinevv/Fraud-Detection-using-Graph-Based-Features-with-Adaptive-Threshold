"""Simple Streamlit frontend for the fraud detection mini project."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import streamlit as st

from fraud_detection import (
    generate_synthetic_data,
    get_user_profile,
    run_pipeline,
    run_prediction_pipeline,
)


st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="FD",
    layout="wide",
)

if "analysis_ready" not in st.session_state:
    st.session_state.analysis_ready = False
if "results" not in st.session_state:
    st.session_state.results = None
if "uploaded_bytes" not in st.session_state:
    st.session_state.uploaded_bytes = None
if "prediction_bytes" not in st.session_state:
    st.session_state.prediction_bytes = None
if "prediction_results" not in st.session_state:
    st.session_state.prediction_results = None

st.title("Fraud Investigation Intelligence Dashboard")
st.caption("Graph-based detection, adaptive thresholding, and user-level investigation workflows")

with st.sidebar:
    st.header("Controls")
    uploaded_file = st.file_uploader("Upload training CSV (with fraud_label)", type=["csv"])
    prediction_file = st.file_uploader("Upload prediction CSV (without fraud_label)", type=["csv"])
    synthetic_rows = st.slider("Synthetic transactions", 500, 3000, 1000, 100)
    test_size = st.slider("Test split", 0.1, 0.4, 0.2, 0.05)
    iterations = st.slider("Adaptive threshold iterations", 5, 15, 10, 1)
    initial_threshold = st.slider("Initial threshold", 0.1, 0.9, 0.5, 0.05)
    run_button = st.button("Run investigation system", use_container_width=True)

st.markdown(
    """
    Training CSV should contain
    `sender_id`, `receiver_id`, `amount`, `timestamp`, `fraud_label`,
    and prediction CSV should contain
    `sender_id`, `receiver_id`, `amount`, `timestamp`.
    If no training CSV is uploaded, the app trains on synthetic historical data.
    """
)

if uploaded_file is not None:
    st.session_state.uploaded_bytes = uploaded_file.getvalue()
else:
    st.session_state.uploaded_bytes = None

if prediction_file is not None:
    st.session_state.prediction_bytes = prediction_file.getvalue()
else:
    st.session_state.prediction_bytes = None

if not st.session_state.analysis_ready and not run_button:
    preview_df = generate_synthetic_data(n_transactions=10)
    st.subheader("Expected dataset format")
    st.dataframe(preview_df.head(), use_container_width=True)
    st.stop()

if st.session_state.uploaded_bytes is not None:
    temp_path = Path("uploaded_training_transactions.csv")
    temp_path.write_bytes(st.session_state.uploaded_bytes)
    data_path = str(temp_path)
else:
    data_path = None

if st.session_state.prediction_bytes is not None:
    prediction_temp_path = Path("uploaded_prediction_transactions.csv")
    prediction_temp_path.write_bytes(st.session_state.prediction_bytes)
    prediction_path = str(prediction_temp_path)
else:
    prediction_path = None

if run_button or not st.session_state.analysis_ready:
    with st.spinner("Running fraud detection pipeline..."):
        st.session_state.results = run_pipeline(
            filepath=data_path,
            n_transactions=synthetic_rows,
            test_size=test_size,
            init_threshold=initial_threshold,
            n_iterations=iterations,
        )
        if prediction_path:
            st.session_state.prediction_results = run_prediction_pipeline(
                st.session_state.results["model"],
                prediction_path,
                st.session_state.results["best_threshold"],
            )
        else:
            st.session_state.prediction_results = None
        st.session_state.analysis_ready = True

results = st.session_state.results
prediction_results = st.session_state.prediction_results

df = results["df"]
graph = results["graph"]
X = results["X"]
baseline = results["baseline"]
adaptive = results["adaptive"]
history_df = pd.DataFrame(results["rl_history"])
case_table = results["case_table"].copy()
scored_users = results["scored_users"].copy()


def with_user_id_column(frame: pd.DataFrame) -> pd.DataFrame:
    """Reset index and normalize the user identifier column name."""
    normalized = frame.reset_index()
    first_column = normalized.columns[0]
    if first_column != "user_id":
        normalized = normalized.rename(columns={first_column: "user_id"})
    return normalized

st.subheader("Executive Summary")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Transactions", f"{len(df):,}")
col2.metric("Users", f"{graph.number_of_nodes():,}")
col3.metric("Fraud Transactions", f"{int(df['fraud_label'].sum()):,}")
col4.metric("High-Risk Users", f"{int((case_table['risk_level'] == 'High').sum()):,}")

summary1, summary2, summary3, summary4 = st.columns(4)
summary1.metric("Baseline F1", f"{baseline['f1']:.3f}")
summary2.metric("Adaptive F1", f"{adaptive['f1']:.3f}", delta=f"{adaptive['f1'] - baseline['f1']:+.3f}")
summary3.metric("Final Threshold", f"{results['best_threshold']:.2f}")
summary4.metric("Graph Density", f"{nx.density(nx.DiGraph(graph)):0.4f}")

overview_tab, training_cases_tab, prediction_tab, user_tab, analytics_tab = st.tabs(
    ["Overview", "Training Queue", "Prediction Queue", "User Search", "Analytics"]
)

with overview_tab:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(25), use_container_width=True)

    c1, c2 = st.columns([1.2, 1.0])
    with c1:
        st.markdown("**Model Performance**")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "model_view": "Baseline",
                        "threshold": 0.50,
                        "accuracy": baseline["accuracy"],
                        "precision": baseline["precision"],
                        "recall": baseline["recall"],
                        "f1": baseline["f1"],
                    },
                    {
                        "model_view": "Adaptive",
                        "threshold": results["best_threshold"],
                        "accuracy": adaptive["accuracy"],
                        "precision": adaptive["precision"],
                        "recall": adaptive["recall"],
                        "f1": adaptive["f1"],
                    },
                ]
            ),
            use_container_width=True,
        )
    with c2:
        st.markdown("**Risk Distribution**")
        risk_counts = case_table["risk_level"].value_counts().reindex(["High", "Medium", "Low"]).fillna(0)
        st.bar_chart(risk_counts)

    st.markdown("**Saved Visualization**")
    st.image(results["visualization_path"], caption="System-wide fraud analysis output")

with training_cases_tab:
    st.subheader("Priority Investigation Queue")
    display_cases = with_user_id_column(case_table)
    queue_columns = [
        "user_id",
        "risk_level",
        "fraud_probability",
        "investigation_priority",
        "fraud_incidents",
        "recent_activity",
        "neighbor_fraud_ratio",
        "transaction_frequency",
        "max_transaction_amount",
    ]
    st.dataframe(
        display_cases[queue_columns].head(25),
        use_container_width=True,
    )
    st.markdown("This queue ranks users by model probability, suspicious neighbors, velocity, and repeated fraud involvement.")

with prediction_tab:
    st.subheader("Prediction on New Unlabeled Transactions")
    if prediction_results is None:
        st.info("Upload a prediction CSV without `fraud_label` to score new users after training.")
    else:
        prediction_case_table = prediction_results["case_table"]
        prediction_display = with_user_id_column(prediction_case_table)
        px1, px2, px3, px4 = st.columns(4)
        px1.metric("Prediction Transactions", f"{len(prediction_results['df']):,}")
        px2.metric("Prediction Users", f"{prediction_results['graph'].number_of_nodes():,}")
        px3.metric("High-Risk Predicted Users", f"{int((prediction_case_table['risk_level'] == 'High').sum()):,}")
        px4.metric("Threshold Used", f"{results['best_threshold']:.2f}")
        st.dataframe(
            prediction_display[
                [
                    "user_id",
                    "risk_level",
                    "fraud_probability",
                    "investigation_priority",
                    "recent_activity",
                    "neighbor_fraud_ratio",
                    "transaction_frequency",
                    "max_transaction_amount",
                ]
            ].head(25),
            use_container_width=True,
        )
        st.markdown(
            "These are predictions on new transactions with no ground-truth fraud labels. "
            "The model was trained on historical labeled data and is now scoring fresh users."
        )

with user_tab:
    st.subheader("Search and Investigate a User")
    data_source = st.radio("Investigation source", ["Training data", "Prediction data"], horizontal=True)
    selected_table = case_table
    selected_df = df
    selected_graph = graph
    prediction_available = prediction_results is not None
    if data_source == "Prediction data" and prediction_results is not None:
        selected_table = prediction_results["case_table"]
        selected_df = prediction_results["df"]
        selected_graph = prediction_results["graph"]
    if data_source == "Prediction data" and not prediction_available:
        st.info("Upload a prediction CSV to investigate new unlabeled users.")
    else:
        user_options = selected_table.index.tolist()
        search_user = st.selectbox("Select user ID", user_options, index=0)
        profile = get_user_profile(search_user, selected_df, selected_graph, selected_table)

        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Fraud Probability", f"{profile['fraud_probability']:.3f}")
        p2.metric("Risk Level", profile["risk_level"])
        p3.metric("Priority Score", f"{profile['investigation_priority']:.2f}")
        p4.metric("Fraud Incidents", profile["fraud_incidents"])

        st.markdown("**Why this user is flagged**")
        for reason in profile["top_reasons"]:
            st.write(f"- {reason}")

        p5, p6, p7 = st.columns(3)
        p5.metric("Recent Activity (3 days)", profile["recent_activity"])
        p6.metric("First Seen", profile["first_seen"].strftime("%Y-%m-%d %H:%M"))
        p7.metric("Last Seen", profile["last_seen"].strftime("%Y-%m-%d %H:%M"))

        subtab1, subtab2, subtab3, subtab4 = st.tabs(
            ["History", "Behavior", "Network Context", "Feature Snapshot"]
        )

        with subtab1:
            st.markdown("**Complete user transaction history**")
            history_cols = [
                "event_time",
                "direction",
                "counterparty",
                "amount",
                *(["fraud_label"] if "fraud_label" in profile["history"].columns else []),
                "is_high_value",
            ]
            st.dataframe(profile["history"][history_cols], use_container_width=True)

        with subtab2:
            left, right = st.columns(2)
            with left:
                st.markdown("**Daily activity summary**")
                st.dataframe(profile["daily_summary"], use_container_width=True)
            with right:
                st.markdown("**Top counterparties**")
                st.dataframe(profile["counterparties"], use_container_width=True)

        with subtab3:
            left, right = st.columns(2)
            with left:
                st.markdown("**Suspicious neighbor snapshot**")
                if profile["neighbor_snapshot"].empty:
                    st.info("No neighbor context available for this user.")
                else:
                    st.dataframe(profile["neighbor_snapshot"], use_container_width=True)
            with right:
                st.markdown("**Suspicious counterparties**")
                if profile["suspicious_contacts"].empty:
                    st.info("No suspicious counterparties found for this user.")
                else:
                    st.dataframe(profile["suspicious_contacts"], use_container_width=True)

            nearby_nodes = [search_user] + profile["neighbor_snapshot"]["user_id"].tolist()[:8]
            subgraph = selected_graph.subgraph(list(dict.fromkeys(nearby_nodes))).copy()
            if subgraph.number_of_nodes() > 1:
                figure, ax = plt.subplots(figsize=(8, 5))
                positions = nx.spring_layout(subgraph, seed=42)
                node_colors = ["#ff4b6e" if node == search_user else "#4b9bff" for node in subgraph.nodes()]
                nx.draw_networkx(
                    subgraph,
                    pos=positions,
                    ax=ax,
                    node_color=node_colors,
                    node_size=500,
                    font_size=8,
                    edge_color="#7c8a9a",
                    arrows=True,
                    arrowsize=12,
                )
                ax.set_title("Local network around selected user")
                ax.axis("off")
                st.pyplot(figure, use_container_width=True)
                plt.close(figure)

        with subtab4:
            snapshot = (
                pd.DataFrame([profile["feature_snapshot"]])
                .T.reset_index()
                .rename(columns={"index": "feature", 0: "value"})
            )
            st.dataframe(snapshot, use_container_width=True)

with analytics_tab:
    st.subheader("System Analytics")
    a1, a2 = st.columns(2)
    with a1:
        st.markdown("**Adaptive threshold iterations**")
        st.dataframe(history_df, use_container_width=True)
    with a2:
        st.markdown("**Top high-risk users**")
        st.dataframe(
            with_user_id_column(scored_users)
            [["user_id", "fraud_probability", "risk_level", "neighbor_fraud_ratio", "transaction_frequency"]]
            .head(15),
            use_container_width=True,
        )

    st.markdown("**Feature Matrix Preview**")
    st.dataframe(X.head(20), use_container_width=True)

st.success("Investigation system is ready. You can now search a user and inspect their full history.")
