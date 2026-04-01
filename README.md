# 🚀 Fraud Detection using Graph-Based Features & Adaptive Threshold

## 📌 Overview

This project is a **graph-based fraud detection and investigation system** built using Python.

Unlike traditional systems that analyze transactions individually, this system models **relationships between users** to detect suspicious patterns.

It combines:

* 🧠 Graph-based feature engineering (NetworkX)
* 🤖 Machine Learning (Random Forest)
* 🔁 RL-inspired adaptive decision-making
* 📊 Interactive dashboard (Streamlit)

---

## 🎯 Problem Statement

Traditional fraud detection relies on:

* high transaction amount
* frequency thresholds
* rule-based systems

⚠️ Limitation: Fraud is **pattern-based**, not event-based.

Examples:

* Rapid transfers to multiple accounts (fan-out)
* Circular money movement
* Suspicious network clusters

✅ This project solves it using **graph-based behavioral analysis**.

---

## 💡 Core Idea

* **Nodes** → Users
* **Edges** → Transactions

We extract behavioral and structural features from this graph to identify fraud patterns.

---

## ⚙️ Project Workflow

### 1️⃣ Data Input

Supports:

#### ✅ Training Data (with labels)

```csv
sender_id,receiver_id,amount,timestamp,fraud_label
```

#### ✅ Prediction Data (without labels)

```csv
sender_id,receiver_id,amount,timestamp
```

---

### 2️⃣ Graph Construction

Using `NetworkX`:

* Directed graph
* Nodes = Users
* Edges = Transactions

Each edge stores:

* amount
* timestamp
* fraud label (if available)

---

### 3️⃣ Feature Engineering (🔥 Core Part)

#### Basic Features

* total_transactions
* total_amount_sent
* total_amount_received
* avg_transaction_amount
* max_transaction_amount
* unique_receivers
* unique_senders

#### Graph Features

* degree_centrality
* clustering_coefficient
* in_degree
* out_degree

#### Advanced Features

* neighbor_fraud_ratio
* transaction_frequency

---

### 4️⃣ Model Training

* Model: `RandomForestClassifier`
* Train/Test Split: 80/20
* Output: **Fraud Probability**

---

### 5️⃣ Baseline Evaluation

* Threshold = `0.5`

Metrics:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

---

### 6️⃣ 🔁 Adaptive Threshold (RL-Inspired)

Instead of fixed decision-making:

#### Reward Function:

```text
reward = (TP * 2) - (FP * 1) - (FN * 3)
```

#### Logic:

* High FP → Increase threshold
* High FN → Decrease threshold

✔ Makes system adaptive
✔ Reduces false positives
✔ Improves real-world usability

---

### 7️⃣ Final Evaluation

Compare:

* Before Adaptive Threshold
* After Adaptive Threshold

---

## 📊 Visualizations

* Confusion Matrix (Before vs After)
* Threshold vs Accuracy
* Feature Importance
* Transaction Graph Visualization

---

## 🖥️ Streamlit Dashboard

### Features:

#### 📍 Overview

* Dataset preview
* Graph stats
* Model performance

#### 📍 Investigation Queue

* Ranked suspicious users

#### 📍 User Analysis

* Fraud probability
* Transaction history
* Network connections
* Risk explanation

#### 📍 Analytics

* Threshold tuning graph
* High-risk users

---

## 🧪 Synthetic Data Support

If no dataset is provided:

* Generates ~1000 transactions
* Includes fraud patterns:

  * burst activity
  * fan-out behavior
  * circular transactions

---

## 📁 Project Structure

```bash
├── fraud_detection.py
├── app.py
├── requirements.txt
├── prediction_demo.csv
├── fraud_detection_results.png
```

---

## ▶️ How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Backend Pipeline

```bash
python fraud_detection.py
```

### 3. Run Streamlit App

```bash
streamlit run app.py
```

Open:

```
http://localhost:8501
```

---

## 📌 Real-World Use Case

Applicable in:

* Banking systems
* FinTech apps
* Digital wallets
* Transaction monitoring platforms

---

## ✅ Strengths

* Graph-based intelligence (not just rules)
* Explainable ML model
* Adaptive decision-making
* Investigation-ready system
* Real-world applicability

---

## ⚠️ Limitations

* No full GNN implementation
* No full RL agent
* Depends on dataset quality
* Not production-scale

---

## 🔮 Future Improvements

* Graph Neural Networks (GNN)
* Deep Reinforcement Learning (DQN)
* Real-time fraud detection
* API backend (FastAPI)
* SHAP explainability
* Database integration

---

## 🧠 Key Functions

```python
load_data()
build_graph()
extract_features()
train_model()
evaluate_model()
adaptive_threshold()
run_pipeline()
```

---

## 🏁 Conclusion

This project demonstrates how:

✔ Graph relationships
✔ Machine learning
✔ Adaptive logic

can be combined to build a **practical and intelligent fraud detection system**.

It goes beyond prediction and enables:

* investigation
* explanation
* prioritization of risk

---

## 👨‍💻 Authors

* Abhinav Nigade
* Team Members

---

⭐ If you like this project, consider starring the repository!
