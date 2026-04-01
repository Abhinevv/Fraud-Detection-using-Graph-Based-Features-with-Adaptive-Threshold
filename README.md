# 🚀 Fraud Detection using Graph Intelligence, GNN & Adaptive Threshold

## 📌 Overview

This project is a **graph-based fraud detection and investigation system** built in Python.

It combines:

* 🧠 Graph-based feature engineering (`NetworkX`)
* 🌲 Machine Learning (`RandomForestClassifier`)
* 🔗 Lightweight Graph Neural Network (NumPy-based GCN)
* 🔁 Reinforcement Learning-inspired adaptive threshold
* 📊 Interactive investigation dashboard (`Streamlit`)

The system supports both:

* Training on labeled historical data
* Predicting fraud risk on new unlabeled transactions

## 🧠 AI Concepts Used

This project incorporates key concepts from Artificial Intelligence and Expert Systems:

* **Machine Learning (Supervised Learning):**
  Uses Random Forest to learn fraud patterns from historical data.

* **Graph-Based Learning:**
  Models transactions as a network to capture relationships between users.

* **Graph Neural Network (GNN):**
  Applies GCN-style learning to understand neighborhood influence and detect fraud clusters.

* **Reinforcement Learning (Concept):**
  Uses a reward-based adaptive threshold to improve decision-making over time.

* **Expert System Behavior:**
  Provides risk scoring, pattern analysis, and decision support similar to a fraud analyst.

These components make the system an **AI-driven fraud detection solution**, not just a traditional rule-based model.

---

## 🎯 Problem Statement

Fraud is rarely visible from a single transaction.

Instead, fraud emerges from **patterns in relationships**, such as:

* fan-out transfers (one → many)
* circular money movement
* suspicious clusters
* repeated abnormal interactions

⚠️ Traditional row-based models fail to capture this.

✅ This project solves it using **graph-based relational learning + GNN logic**.

---

## 💡 Core Idea

* **Nodes** → Users
* **Edges** → Transactions

Fraud detection is performed using:

* behavioral features
* graph structure
* neighborhood risk patterns

---

## ⚙️ System Workflow

### 1️⃣ Training Workflow

Input:

```csv
sender_id,receiver_id,amount,timestamp,fraud_label
```

Steps:

1. Build transaction graph
2. Extract graph-based features
3. Train:

   * RandomForest model
   * GNN (GCN-style) model
4. Evaluate performance
5. Apply adaptive threshold

---

### 2️⃣ Prediction Workflow

Input:

```csv
sender_id,receiver_id,amount,timestamp
```

Steps:

1. Build graph
2. Extract features
3. Predict fraud probability
4. Generate investigation queue

---

## 🔄 PaySim Dataset Support

Automatically maps:

* `nameOrig → sender_id`
* `nameDest → receiver_id`
* `step → timestamp`
* `isFraud → fraud_label`

✔ No manual preprocessing required

---

## 🤖 Models Used

### 🌲 Random Forest

* Works on engineered graph features
* Fast and interpretable
* Provides feature importance

### 🔗 Graph Neural Network (GCN - NumPy)

* Uses graph structure directly
* Learns from neighbors
* Detects fraud rings and communities

> Note: Lightweight academic implementation (not PyTorch)

---

## 🧪 Feature Engineering (Core Strength)

### Basic Features

* total_transactions
* total_amount_sent
* total_amount_received
* avg_transaction_amount
* max_transaction_amount
* unique_receivers
* unique_senders

### Graph Features

* degree_centrality
* clustering_coefficient
* in_degree
* out_degree

### Advanced Fraud Features

* neighbor_fraud_ratio
* transaction_frequency

### Financial Behavior Features (PaySim)

* transfer_ratio
* cashout_ratio
* payment_ratio
* balance_delta features
* zero-balance anomalies

### Optional Real-World Features

* merchant_id
* device_id
* location
* account age

---

## 🔁 Adaptive Threshold (RL-Inspired)

Instead of fixed threshold:

### Reward Function

```text
reward = (TP * 2) - (FP * 1) - (FN * 3)
```

### Adjustment Logic

* High FP → Increase threshold
* High FN → Decrease threshold

✔ Improves decision quality
✔ Balances fraud detection vs false alerts

---

## 📊 Dashboard (Streamlit)

### Features

#### 📍 Overview

* dataset preview
* graph statistics
* model performance

#### 📍 Investigation Queue

* ranked suspicious users
* RF + GNN comparison

#### 📍 User Analysis

* fraud probability
* transaction history
* suspicious neighbors
* local graph view
* risk explanation

#### 📍 Analytics

* threshold tuning history
* high-risk users
* feature matrix

---

## 📁 Project Structure

```bash
├── fraud_detection.py
├── app.py
├── requirements.txt
├── prediction_demo.csv
├── gnn_synthetic_dataset.csv
├── fraud_detection_results.png
```

---

## ▶️ How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Backend

```bash
python fraud_detection.py
```

### 3. Run Dashboard

```bash
streamlit run app.py
```

Open:

```
http://localhost:8501
```

---

## 🧪 Synthetic Data

Included datasets:

* `prediction_demo.csv`
* `gnn_synthetic_dataset.csv`

Contains:

* fraud rings
* fan-out patterns
* bridge accounts
* realistic anomalies

---

## 🌍 Real-World Applications

* Banking fraud detection
* FinTech risk scoring
* Digital wallet monitoring
* Money mule detection
* Network-based fraud investigation

---

## ✅ Strengths

* Graph-based intelligence (beyond flat ML)
* Combines ML + GNN + adaptive logic
* Works with real & synthetic data
* Investigation-ready system
* Strong academic + practical value

---

## ⚠️ Limitations

* GNN is lightweight (NumPy-based)
* No deep RL agent
* Not production-scale
* Performance depends on data quality

---

## 🔮 Future Improvements

* PyTorch Geometric GNN
* Deep Reinforcement Learning (DQN)
* Real-time streaming detection
* FastAPI backend
* SHAP explainability
* Database integration

---

## 🧠 Key Functions

```python
load_data()
build_graph()
extract_features()
train_model()
train_gnn_model()
evaluate_model()
adaptive_threshold()
run_pipeline()
run_prediction_pipeline()
```

---

## 🏁 Conclusion

This project demonstrates how:

✔ Graph analytics
✔ Machine learning
✔ GNN concepts
✔ Adaptive decision-making

can be combined to build a **practical fraud detection system**.

It goes beyond prediction and supports:

* investigation
* explanation
* risk prioritization

---

## 👨‍💻 Authors

* Abhinav Nigade
* Team Members

---

⭐ Star this repo if you found it useful!
