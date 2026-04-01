"""
Fraud Detection using Graph-Based Features with Adaptive Threshold.

This module supports:
1. Loading a CSV dataset or generating synthetic transactions
2. Building a NetworkX transaction graph
3. Extracting graph-based user features
4. Training a RandomForest classifier
5. Evaluating baseline and adaptive-threshold performance
6. Saving a summary visualization figure
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
from typing import Any

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


BASE_COLUMNS = {"sender_id", "receiver_id", "amount", "timestamp"}
REQUIRED_COLUMNS = BASE_COLUMNS | {"fraud_label"}


def safe_print(*args: Any, **kwargs: Any) -> None:
    """Print without crashing environments that do not expose a normal stdout."""
    try:
        print(*args, **kwargs)
    except OSError:
        pass


def has_labels(df: pd.DataFrame) -> bool:
    """Return True when a dataset includes a usable fraud_label column."""
    return "fraud_label" in df.columns and df["fraud_label"].notna().any()


@dataclass
class TrainingArtifacts:
    model: RandomForestClassifier
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    y_proba: np.ndarray


def _to_datetime_series(values: pd.Series) -> pd.Series:
    """Convert unix timestamps to pandas datetime."""
    return pd.to_datetime(values, unit="s", errors="coerce")


def generate_synthetic_data(
    n_transactions: int = 1000,
    n_users: int = 140,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic transaction data with embedded fraud patterns.

    Fraud patterns:
    - High-value bursts from a small set of risky senders
    - Fan-out behavior where one sender hits many receivers quickly
    - Circular transactions inside a tight ring of users
    """
    rng = np.random.default_rng(seed)
    users = np.array([f"U{i:03d}" for i in range(n_users)])

    senders = rng.choice(users, size=n_transactions)
    receivers = rng.choice(users, size=n_transactions)

    same_mask = senders == receivers
    while same_mask.any():
        receivers[same_mask] = rng.choice(users, size=same_mask.sum())
        same_mask = senders == receivers

    amounts = rng.lognormal(mean=4.2, sigma=1.0, size=n_transactions).clip(5, 5000)
    base_time = pd.Timestamp("2025-01-01").timestamp()
    timestamps = (
        base_time + rng.integers(0, 30 * 24 * 60 * 60, size=n_transactions)
    ).astype(int)
    fraud_label = np.zeros(n_transactions, dtype=int)

    risky_senders = rng.choice(users[:25], size=10, replace=False)
    burst_mask = np.isin(senders, risky_senders)
    burst_count = int(burst_mask.sum())
    amounts[burst_mask] = rng.uniform(1800, 4200, size=burst_count)
    timestamps[burst_mask] = int(base_time) + rng.integers(0, 5400, size=burst_count)
    fraud_label[burst_mask] = 1

    fanout_sender = users[1]
    fanout_candidates = np.where(senders == fanout_sender)[0]
    fanout_size = min(25, len(fanout_candidates))
    if fanout_size > 0:
        fanout_idx = rng.choice(fanout_candidates, size=fanout_size, replace=False)
        amounts[fanout_idx] = rng.uniform(90, 650, size=fanout_size)
        timestamps[fanout_idx] = int(base_time) + rng.integers(0, 900, size=fanout_size)
        fraud_label[fanout_idx] = 1

    legit_sample_size = min(n_transactions, max(1, min(25, n_transactions // 25 if n_transactions >= 25 else n_transactions // 3 or 1)))
    legit_high_value_idx = rng.choice(np.arange(n_transactions), size=legit_sample_size, replace=False)
    legit_high_value_mask = fraud_label[legit_high_value_idx] == 0
    legit_high_value_idx = legit_high_value_idx[legit_high_value_mask]
    if len(legit_high_value_idx) > 0:
        amounts[legit_high_value_idx] = rng.uniform(1200, 2600, size=len(legit_high_value_idx))

    non_fraud_candidates = np.where(fraud_label == 0)[0]
    stealth_sample_size = min(
        len(non_fraud_candidates),
        max(1, min(18, n_transactions // 40 if n_transactions >= 40 else n_transactions // 4 or 1)),
    )
    if stealth_sample_size > 0:
        stealth_fraud_idx = rng.choice(non_fraud_candidates, size=stealth_sample_size, replace=False)
        amounts[stealth_fraud_idx] = rng.uniform(200, 900, size=len(stealth_fraud_idx))
        timestamps[stealth_fraud_idx] = int(base_time) + rng.integers(22 * 3600, 24 * 3600, size=len(stealth_fraud_idx))
        fraud_label[stealth_fraud_idx] = 1

    ring_users = users[20:26]
    extra_rows: list[dict[str, Any]] = []
    for _ in range(5):
        for index, sender in enumerate(ring_users):
            receiver = ring_users[(index + 1) % len(ring_users)]
            extra_rows.append(
                {
                    "sender_id": sender,
                    "receiver_id": receiver,
                    "amount": round(float(rng.uniform(600, 1800)), 2),
                    "timestamp": int(base_time) + int(rng.integers(0, 1800)),
                    "fraud_label": 1,
                }
            )

    df = pd.DataFrame(
        {
            "sender_id": senders,
            "receiver_id": receivers,
            "amount": np.round(amounts, 2),
            "timestamp": timestamps,
            "fraud_label": fraud_label,
        }
    )
    if extra_rows:
        df = pd.concat([df, pd.DataFrame(extra_rows)], ignore_index=True)

    safe_print(
        f"[DATA] Generated {len(df):,} transactions with "
        f"{int(df['fraud_label'].sum()):,} fraud transactions "
        f"({df['fraud_label'].mean() * 100:.2f}%)."
    )
    return df


def load_data(
    filepath: str | None = None,
    n_transactions: int = 1000,
    require_labels: bool = True,
) -> pd.DataFrame:
    """Load a CSV file if provided, otherwise generate synthetic data."""
    if filepath:
        df = pd.read_csv(filepath)
        required = REQUIRED_COLUMNS if require_labels else BASE_COLUMNS
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
        safe_print(f"[DATA] Loaded {len(df):,} transactions from {filepath}")
    else:
        if require_labels:
            safe_print("[DATA] No CSV provided. Generating synthetic data.")
            df = generate_synthetic_data(n_transactions=n_transactions)
        else:
            raise ValueError("Prediction mode requires a CSV with sender_id, receiver_id, amount, timestamp.")

    df = df.copy()
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    if "fraud_label" in df.columns:
        df["fraud_label"] = pd.to_numeric(df["fraud_label"], errors="coerce")
    elif require_labels:
        df["fraud_label"] = 0
    df = df.dropna(subset=["sender_id", "receiver_id", "amount", "timestamp"])
    if require_labels:
        df["fraud_label"] = df["fraud_label"].fillna(0).astype(int)
    df["event_time"] = _to_datetime_series(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def build_graph(df: pd.DataFrame) -> nx.MultiDiGraph:
    """Build a directed multi-graph where each edge represents one transaction."""
    graph = nx.MultiDiGraph()
    for row in df.itertuples(index=False):
        fraud_value = int(row.fraud_label) if hasattr(row, "fraud_label") and pd.notna(row.fraud_label) else 0
        graph.add_edge(
            row.sender_id,
            row.receiver_id,
            amount=float(row.amount),
            timestamp=float(row.timestamp),
            fraud=fraud_value,
        )

    safe_print(
        f"[GRAPH] Nodes={graph.number_of_nodes():,}, "
        f"Edges={graph.number_of_edges():,}"
    )
    return graph


def _compute_node_fraud_flags(df: pd.DataFrame) -> dict[str, int]:
    """
    Derive node labels from repeated fraud involvement instead of a single event.

    A node is marked suspicious when:
    - it is involved in at least 2 fraud transactions, or
    - at least 25% of its incident transactions are fraudulent
    """
    total_counts: defaultdict[str, int] = defaultdict(int)
    fraud_counts: defaultdict[str, int] = defaultdict(int)

    for row in df.itertuples(index=False):
        total_counts[row.sender_id] += 1
        total_counts[row.receiver_id] += 1
        if int(row.fraud_label) == 1:
            fraud_counts[row.sender_id] += 1
            fraud_counts[row.receiver_id] += 1

    labels: dict[str, int] = {}
    for node, total in total_counts.items():
        fraud_total = fraud_counts[node]
        fraud_ratio = fraud_total / total if total else 0.0
        labels[node] = int(fraud_total >= 2 or fraud_ratio >= 0.25)
    return labels


def _compute_proxy_risk_flags(df: pd.DataFrame) -> dict[str, int]:
    """Estimate suspicious nodes for unlabeled prediction datasets."""
    node_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"amounts": [], "times": [], "receivers": set(), "send_count": 0, "total_count": 0}
    )

    for row in df.itertuples(index=False):
        sender_stats = node_stats[row.sender_id]
        sender_stats["amounts"].append(float(row.amount))
        sender_stats["times"].append(float(row.timestamp))
        sender_stats["receivers"].add(row.receiver_id)
        sender_stats["send_count"] += 1
        sender_stats["total_count"] += 1

        receiver_stats = node_stats[row.receiver_id]
        receiver_stats["amounts"].append(float(row.amount))
        receiver_stats["times"].append(float(row.timestamp))
        receiver_stats["total_count"] += 1

    summary_rows: list[dict[str, Any]] = []
    for node, stats in node_stats.items():
        times = sorted(stats["times"])
        if len(times) > 1:
            span_hours = max((times[-1] - times[0]) / 3600.0, 1 / 60)
            frequency = len(times) / span_hours
        else:
            frequency = float(len(times))

        summary_rows.append(
            {
                "node": node,
                "max_amount": max(stats["amounts"]) if stats["amounts"] else 0.0,
                "fanout": len(stats["receivers"]),
                "frequency": frequency,
            }
        )

    summary = pd.DataFrame(summary_rows)
    if summary.empty:
        return {}

    amount_cutoff = summary["max_amount"].quantile(0.85)
    frequency_cutoff = summary["frequency"].quantile(0.85)
    fanout_cutoff = summary["fanout"].quantile(0.85)

    flags: dict[str, int] = {}
    for row in summary.itertuples(index=False):
        flags[row.node] = int(
            row.max_amount >= amount_cutoff
            or row.frequency >= frequency_cutoff
            or row.fanout >= fanout_cutoff
        )
    return flags


def _extract_features_internal(
    df: pd.DataFrame,
    graph: nx.MultiDiGraph,
    node_risk_flags: dict[str, int],
) -> tuple[pd.DataFrame, pd.Series]:
    """Extract node-level graph and transaction features."""
    simple_graph = nx.DiGraph()
    for sender, receiver, data in graph.edges(data=True):
        if simple_graph.has_edge(sender, receiver):
            simple_graph[sender][receiver]["weight"] += data["amount"]
        else:
            simple_graph.add_edge(sender, receiver, weight=data["amount"])

    degree_centrality = nx.degree_centrality(simple_graph)
    clustering = nx.clustering(simple_graph.to_undirected())
    rows: list[dict[str, Any]] = []
    for node in graph.nodes():
        outgoing = list(graph.out_edges(node, data=True))
        incoming = list(graph.in_edges(node, data=True))

        sent_amounts = [edge_data["amount"] for _, _, edge_data in outgoing]
        received_amounts = [edge_data["amount"] for _, _, edge_data in incoming]
        all_amounts = sent_amounts + received_amounts

        sent_times = sorted(edge_data["timestamp"] for _, _, edge_data in outgoing)
        if len(sent_times) > 1:
            span_hours = max((sent_times[-1] - sent_times[0]) / 3600.0, 1 / 60)
            transaction_frequency = len(sent_times) / span_hours
        else:
            transaction_frequency = float(len(sent_times))

        neighbors = set(graph.predecessors(node)).union(set(graph.successors(node)))
        neighbor_fraud_ratio = (
            sum(node_risk_flags.get(neighbor, 0) for neighbor in neighbors) / len(neighbors)
            if neighbors
            else 0.0
        )

        rows.append(
            {
                "node": node,
                "total_transactions": len(outgoing) + len(incoming),
                "total_amount_sent": float(sum(sent_amounts)),
                "total_amount_received": float(sum(received_amounts)),
                "avg_transaction_amount": float(np.mean(all_amounts)) if all_amounts else 0.0,
                "max_transaction_amount": float(max(all_amounts)) if all_amounts else 0.0,
                "unique_receivers": len({receiver for _, receiver, _ in outgoing}),
                "unique_senders": len({sender for sender, _, _ in incoming}),
                "degree_centrality": float(degree_centrality.get(node, 0.0)),
                "clustering_coefficient": float(clustering.get(node, 0.0)),
                "in_degree": int(graph.in_degree(node)),
                "out_degree": int(graph.out_degree(node)),
                "neighbor_fraud_ratio": float(neighbor_fraud_ratio),
                "transaction_frequency": float(transaction_frequency),
                "fraud_label": int(node_risk_flags.get(node, 0)),
            }
        )

    feature_df = pd.DataFrame(rows).set_index("node").sort_index()
    X = feature_df.drop(columns=["fraud_label"])
    y = feature_df["fraud_label"]

    safe_print(f"[FEATURES] X shape={X.shape}, fraud nodes={int(y.sum())}/{len(y)}")
    return X, y


def extract_features(df: pd.DataFrame, graph: nx.MultiDiGraph) -> tuple[pd.DataFrame, pd.Series]:
    """Extract node-level graph and transaction features for labeled data."""
    return _extract_features_internal(df, graph, _compute_node_fraud_flags(df))


def extract_features_for_inference(df: pd.DataFrame, graph: nx.MultiDiGraph) -> pd.DataFrame:
    """Extract node-level features for unlabeled prediction data."""
    X, _ = _extract_features_internal(df, graph, _compute_proxy_risk_flags(df))
    return X


def score_all_users(
    model: RandomForestClassifier,
    X: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """Predict fraud probability and risk level for every user in the feature matrix."""
    probabilities = model.predict_proba(X)[:, 1]
    scored = X.copy()
    scored["fraud_probability"] = probabilities
    scored["predicted_label"] = (probabilities >= threshold).astype(int)
    scored["risk_level"] = pd.cut(
        probabilities,
        bins=[-0.01, 0.35, 0.7, 1.0],
        labels=["Low", "Medium", "High"],
    ).astype(str)
    return scored.sort_values("fraud_probability", ascending=False)


def build_case_table(
    df: pd.DataFrame,
    scored_users: pd.DataFrame,
) -> pd.DataFrame:
    """Create a ranked investigation table for analysts."""
    case_table = scored_users.copy()
    fraud_counts = defaultdict(int)
    recent_cutoff = df["event_time"].max() - pd.Timedelta(days=3)

    for row in df.itertuples(index=False):
        if hasattr(row, "fraud_label") and pd.notna(row.fraud_label) and int(row.fraud_label) == 1:
            fraud_counts[row.sender_id] += 1
            fraud_counts[row.receiver_id] += 1

    recent_activity = defaultdict(int)
    for row in df[df["event_time"] >= recent_cutoff].itertuples(index=False):
        recent_activity[row.sender_id] += 1
        recent_activity[row.receiver_id] += 1

    case_table["fraud_incidents"] = [fraud_counts.get(node, 0) for node in case_table.index]
    case_table["recent_activity"] = [recent_activity.get(node, 0) for node in case_table.index]
    case_table["investigation_priority"] = (
        case_table["fraud_probability"] * 100
        + case_table["neighbor_fraud_ratio"] * 25
        + case_table["transaction_frequency"].clip(upper=100) * 0.2
        + case_table["fraud_incidents"] * 5
    ).round(2)
    return case_table.sort_values(
        ["investigation_priority", "fraud_probability", "fraud_incidents"],
        ascending=False,
    )


def get_user_transaction_history(df: pd.DataFrame, user_id: str) -> pd.DataFrame:
    """Return the complete transaction history for a selected user."""
    history = df[(df["sender_id"] == user_id) | (df["receiver_id"] == user_id)].copy()
    if history.empty:
        return history

    history["direction"] = np.where(history["sender_id"] == user_id, "Sent", "Received")
    history["counterparty"] = np.where(
        history["sender_id"] == user_id,
        history["receiver_id"],
        history["sender_id"],
    )
    history["is_high_value"] = history["amount"] >= history["amount"].quantile(0.9)
    history["hour"] = history["event_time"].dt.hour
    return history.sort_values("timestamp", ascending=False).reset_index(drop=True)


def get_user_profile(
    user_id: str,
    df: pd.DataFrame,
    graph: nx.MultiDiGraph,
    case_table: pd.DataFrame,
) -> dict[str, Any]:
    """Build a user investigation profile with history, patterns, and explanation."""
    history = get_user_transaction_history(df, user_id)
    if history.empty or user_id not in case_table.index:
        raise ValueError(f"User '{user_id}' not found in the current dataset.")

    row = case_table.loc[user_id]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]

    counterparties = history["counterparty"].value_counts().rename_axis("counterparty").reset_index(name="transactions")
    suspicious_contacts = counterparties[counterparties["counterparty"].isin(case_table.index[case_table["predicted_label"] == 1])]

    neighbor_ids = sorted(set(graph.predecessors(user_id)).union(set(graph.successors(user_id))))
    neighbor_snapshot = (
        case_table.loc[case_table.index.intersection(neighbor_ids), ["fraud_probability", "risk_level", "investigation_priority"]]
        .sort_values("fraud_probability", ascending=False)
        .head(10)
        .reset_index()
    )
    neighbor_first_column = neighbor_snapshot.columns[0]
    if neighbor_first_column != "user_id":
        neighbor_snapshot = neighbor_snapshot.rename(columns={neighbor_first_column: "user_id"})

    daily_summary = (
        history.assign(date=history["event_time"].dt.date)
        .groupby(["date", "direction"], as_index=False)
        .agg(transaction_count=("amount", "size"), total_amount=("amount", "sum"))
        .sort_values("date", ascending=False)
    )

    top_reasons: list[str] = []
    if row["neighbor_fraud_ratio"] >= 0.4:
        top_reasons.append("High proportion of suspicious neighbors")
    if row["transaction_frequency"] >= case_table["transaction_frequency"].quantile(0.8):
        top_reasons.append("Unusually high transaction frequency")
    if row["max_transaction_amount"] >= case_table["max_transaction_amount"].quantile(0.85):
        top_reasons.append("Large transaction spikes detected")
    if row["unique_receivers"] >= case_table["unique_receivers"].quantile(0.8):
        top_reasons.append("Fan-out behavior across many receivers")
    if row["fraud_incidents"] >= 2:
        top_reasons.append("Repeated involvement in fraudulent transactions")
    if not top_reasons:
        top_reasons.append("Moderate risk driven by combined graph and transaction features")

    summary = {
        "user_id": user_id,
        "fraud_probability": float(row["fraud_probability"]),
        "risk_level": str(row["risk_level"]),
        "predicted_label": int(row["predicted_label"]),
        "investigation_priority": float(row["investigation_priority"]),
        "fraud_incidents": int(row["fraud_incidents"]),
        "recent_activity": int(row["recent_activity"]),
        "history": history,
        "daily_summary": daily_summary,
        "counterparties": counterparties.head(10),
        "suspicious_contacts": suspicious_contacts.head(10),
        "neighbor_snapshot": neighbor_snapshot,
        "feature_snapshot": row.to_dict(),
        "top_reasons": top_reasons,
        "first_seen": history["event_time"].min(),
        "last_seen": history["event_time"].max(),
    }
    return summary


def run_prediction_pipeline(
    model: RandomForestClassifier,
    filepath: str,
    threshold: float,
) -> dict[str, Any]:
    """Score a new unlabeled transaction CSV using a trained model."""
    prediction_df = load_data(filepath=filepath, require_labels=False)
    prediction_graph = build_graph(prediction_df)
    prediction_X = extract_features_for_inference(prediction_df, prediction_graph)
    scored_users = score_all_users(model, prediction_X, threshold)
    case_table = build_case_table(prediction_df, scored_users)
    return {
        "df": prediction_df,
        "graph": prediction_graph,
        "X": prediction_X,
        "scored_users": scored_users,
        "case_table": case_table,
    }


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    seed: int = 42,
) -> TrainingArtifacts:
    """Train a RandomForest model and return training artifacts."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=6,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]

    safe_print(f"[MODEL] Training samples={len(X_train)}, test samples={len(X_test)}")
    return TrainingArtifacts(
        model=model,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        y_proba=y_proba,
    )


def evaluate_model(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
    label: str = "Evaluation",
) -> dict[str, Any]:
    """Evaluate classification results at a specified probability threshold."""
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "label": label,
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": cm,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=["Legit", "Fraud"],
            zero_division=0,
        ),
    }

    safe_print(f"\n{'=' * 62}")
    safe_print(f"{label} (threshold={threshold:.2f})")
    safe_print(f"{'=' * 62}")
    safe_print(f"Accuracy : {metrics['accuracy']:.4f}")
    safe_print(f"Precision: {metrics['precision']:.4f}")
    safe_print(f"Recall   : {metrics['recall']:.4f}")
    safe_print(f"F1-score : {metrics['f1']:.4f}")
    safe_print(f"TP={tp} FP={fp} FN={fn} TN={tn}")
    safe_print(metrics["classification_report"])

    return metrics


def adaptive_threshold(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    init_threshold: float = 0.5,
    n_iterations: int = 10,
    step: float = 0.05,
) -> tuple[float, list[dict[str, Any]]]:
    """
    RL-inspired adaptive threshold search.

    Reward:
    reward = (TP * 2) - (FP * 1) - (FN * 3)
    """
    threshold = float(np.clip(init_threshold, 0.1, 0.9))
    history: list[dict[str, Any]] = []
    best_threshold = threshold
    best_reward = -np.inf

    safe_print("\n" + "=" * 62)
    safe_print("Adaptive threshold tuning")
    safe_print("=" * 62)

    for iteration in range(1, n_iterations + 1):
        metrics = evaluate_model(y_true, y_proba, threshold=threshold, label=f"RL Iteration {iteration}")
        reward = (metrics["tp"] * 2) - metrics["fp"] - (metrics["fn"] * 3)

        if reward > best_reward:
            best_reward = reward
            best_threshold = threshold

        fp_ratio = metrics["fp"] / max(metrics["fp"] + metrics["tn"], 1)
        fn_ratio = metrics["fn"] / max(metrics["fn"] + metrics["tp"], 1)

        if metrics["fp"] > metrics["fn"] and fp_ratio >= fn_ratio:
            next_threshold = threshold + step
            direction = "increase"
        elif metrics["fn"] > metrics["fp"]:
            next_threshold = threshold - step
            direction = "decrease"
        else:
            if metrics["recall"] < metrics["precision"]:
                next_threshold = threshold - step
                direction = "decrease"
            else:
                next_threshold = threshold + step
                direction = "increase"

        next_threshold = float(np.clip(next_threshold, 0.1, 0.9))
        history.append(
            {
                "iteration": iteration,
                "threshold": threshold,
                "next_threshold": next_threshold,
                "direction": direction,
                "reward": reward,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "tp": metrics["tp"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "tn": metrics["tn"],
            }
        )

        safe_print(
            f"[RL] iter={iteration:02d} threshold={threshold:.2f} "
            f"reward={reward:+d} action={direction} next={next_threshold:.2f}"
        )
        threshold = next_threshold

    safe_print(f"[RL] Best threshold selected: {best_threshold:.2f} (reward={int(best_reward):+d})")
    return best_threshold, history


def visualize_results(
    baseline_metrics: dict[str, Any],
    adaptive_metrics: dict[str, Any],
    rl_history: list[dict[str, Any]],
    graph: nx.MultiDiGraph,
    X: pd.DataFrame,
    model: RandomForestClassifier,
    output_path: str = "fraud_detection_results.png",
) -> str:
    """Save a combined figure with confusion matrices, RL history, and graph view."""
    plt.style.use("dark_background")
    bg = "#0d1117"
    panel = "#161b22"
    accent = "#00f5a0"
    blue = "#4b9bff"
    red = "#ff4b6e"
    yellow = "#ffd700"

    figure = plt.figure(figsize=(18, 12), facecolor=bg)
    figure.suptitle(
        "Fraud Detection - Graph Features + Adaptive Threshold",
        fontsize=18,
        fontweight="bold",
        color=accent,
        y=0.98,
    )
    grid = gridspec.GridSpec(2, 3, figure=figure, hspace=0.4, wspace=0.35)

    def draw_cm(ax: Any, cm: np.ndarray, title: str, color: str) -> None:
        ax.set_facecolor(panel)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="YlOrRd",
            cbar=False,
            linewidths=1,
            linecolor=bg,
            xticklabels=["Legit", "Fraud"],
            yticklabels=["Legit", "Fraud"],
            ax=ax,
            annot_kws={"size": 14, "weight": "bold"},
        )
        ax.set_title(title, color=color, fontsize=11)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    ax1 = figure.add_subplot(grid[0, 0])
    draw_cm(
        ax1,
        baseline_metrics["confusion_matrix"],
        f"Baseline CM (thr={baseline_metrics['threshold']:.2f})\n"
        f"Acc={baseline_metrics['accuracy']:.3f} F1={baseline_metrics['f1']:.3f}",
        blue,
    )

    ax2 = figure.add_subplot(grid[0, 1])
    draw_cm(
        ax2,
        adaptive_metrics["confusion_matrix"],
        f"Adaptive CM (thr={adaptive_metrics['threshold']:.2f})\n"
        f"Acc={adaptive_metrics['accuracy']:.3f} F1={adaptive_metrics['f1']:.3f}",
        accent,
    )

    ax3 = figure.add_subplot(grid[0, 2])
    ax3.set_facecolor(panel)
    iterations = [row["iteration"] for row in rl_history]
    thresholds = [row["threshold"] for row in rl_history]
    accuracies = [row["accuracy"] for row in rl_history]
    ax3.plot(iterations, thresholds, "o-", color=yellow, label="Threshold")
    ax3.set_xlabel("RL Iteration")
    ax3.set_ylabel("Threshold", color=yellow)
    ax3.tick_params(axis="y", colors=yellow)
    twin = ax3.twinx()
    twin.plot(iterations, accuracies, "s--", color=accent, label="Accuracy")
    twin.set_ylabel("Accuracy", color=accent)
    twin.tick_params(axis="y", colors=accent)
    ax3.set_title("Threshold and Accuracy")
    ax3.grid(alpha=0.2)

    ax4 = figure.add_subplot(grid[1, 0])
    ax4.set_facecolor(panel)
    rewards = [row["reward"] for row in rl_history]
    reward_colors = [accent if reward >= 0 else red for reward in rewards]
    ax4.bar(iterations, rewards, color=reward_colors)
    ax4.set_title("RL Reward per Iteration")
    ax4.set_xlabel("RL Iteration")
    ax4.set_ylabel("Reward")
    ax4.grid(alpha=0.2, axis="y")

    ax5 = figure.add_subplot(grid[1, 1])
    ax5.set_facecolor(panel)
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values()
    colors = [accent if value >= importances.quantile(0.7) else blue for value in importances]
    ax5.barh(importances.index, importances.values, color=colors)
    ax5.set_title("Feature Importances")
    ax5.set_xlabel("Importance")
    ax5.grid(alpha=0.2, axis="x")

    ax6 = figure.add_subplot(grid[1, 2])
    ax6.set_facecolor(panel)
    top_nodes = [node for node, _ in sorted(graph.degree(), key=lambda item: item[1], reverse=True)[:20]]
    subgraph = graph.subgraph(top_nodes).copy()
    positions = nx.spring_layout(subgraph, seed=42)
    fraud_nodes = _compute_node_fraud_flags(
        pd.DataFrame(
            [
                {
                    "sender_id": sender,
                    "receiver_id": receiver,
                    "fraud_label": data.get("fraud", 0),
                }
                for sender, receiver, data in graph.edges(data=True)
            ]
        )
    )
    node_colors = [red if fraud_nodes.get(node, 0) else blue for node in subgraph.nodes()]
    nx.draw_networkx(
        subgraph,
        pos=positions,
        ax=ax6,
        node_color=node_colors,
        node_size=240,
        with_labels=True,
        font_size=7,
        edge_color="#5c6773",
        arrows=True,
        arrowsize=10,
        width=0.8,
    )
    ax6.set_title("Top 20 Transaction Nodes")
    ax6.axis("off")

    output = Path(output_path)
    figure.savefig(output, dpi=150, bbox_inches="tight", facecolor=bg)
    plt.close(figure)
    safe_print(f"[VIZ] Saved results figure to {output.resolve()}")
    return str(output.resolve())


def print_final_summary(
    baseline_metrics: dict[str, Any],
    adaptive_metrics: dict[str, Any],
    final_threshold: float,
) -> None:
    """Print a compact before/after comparison."""
    safe_print("\n" + "=" * 62)
    safe_print("Final comparison summary")
    safe_print("=" * 62)
    for key in ["accuracy", "precision", "recall", "f1"]:
        delta = adaptive_metrics[key] - baseline_metrics[key]
        safe_print(
            f"{key.capitalize():<10} baseline={baseline_metrics[key]:.4f} "
            f"adaptive={adaptive_metrics[key]:.4f} delta={delta:+.4f}"
        )
    safe_print(f"Final adaptive threshold: {final_threshold:.2f}")


def run_pipeline(
    filepath: str | None = None,
    n_transactions: int = 1000,
    test_size: float = 0.2,
    init_threshold: float = 0.5,
    n_iterations: int = 10,
    output_path: str = "fraud_detection_results.png",
) -> dict[str, Any]:
    """Run the complete project pipeline and return all major outputs."""
    safe_print("\n" + "=" * 62)
    safe_print("Fraud Detection using Graph-Based Features with Adaptive Threshold")
    safe_print("=" * 62)

    df = load_data(filepath=filepath, n_transactions=n_transactions)
    graph = build_graph(df)
    X, y = extract_features(df, graph)
    artifacts = train_model(X, y, test_size=test_size)

    baseline = evaluate_model(
        artifacts.y_test,
        artifacts.y_proba,
        threshold=0.5,
        label="Baseline Evaluation",
    )
    best_threshold, rl_history = adaptive_threshold(
        artifacts.y_test,
        artifacts.y_proba,
        init_threshold=init_threshold,
        n_iterations=n_iterations,
    )
    adaptive = evaluate_model(
        artifacts.y_test,
        artifacts.y_proba,
        threshold=best_threshold,
        label="Adaptive Threshold Evaluation",
    )
    scored_users = score_all_users(artifacts.model, X, best_threshold)
    case_table = build_case_table(df, scored_users)

    visualization_path = visualize_results(
        baseline,
        adaptive,
        rl_history,
        graph,
        X,
        artifacts.model,
        output_path=output_path,
    )
    print_final_summary(baseline, adaptive, best_threshold)

    return {
        "df": df,
        "graph": graph,
        "X": X,
        "y": y,
        "model": artifacts.model,
        "X_train": artifacts.X_train,
        "X_test": artifacts.X_test,
        "y_train": artifacts.y_train,
        "y_test": artifacts.y_test,
        "y_proba": artifacts.y_proba,
        "baseline": baseline,
        "adaptive": adaptive,
        "best_threshold": best_threshold,
        "rl_history": rl_history,
        "scored_users": scored_users,
        "case_table": case_table,
        "visualization_path": visualization_path,
    }


if __name__ == "__main__":
    run_pipeline()
