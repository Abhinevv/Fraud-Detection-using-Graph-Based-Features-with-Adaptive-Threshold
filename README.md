# Fraud Detection using Graph-Based Features with Adaptive Threshold

## Project Overview

This project is a graph-based fraud detection and investigation system built in Python.

Instead of only checking transactions one by one, the system studies how users behave inside a transaction network:

- who sends money to whom
- how often transactions happen
- how much money is moved
- whether a user is connected to risky users
- whether a user shows unusual patterns like bursts, fan-out, or repeated suspicious behavior

The project combines:

- `NetworkX` for transaction graph construction
- graph-based feature engineering
- `RandomForestClassifier` for fraud prediction
- an RL-inspired adaptive threshold for better decision making
- a `Streamlit` dashboard for investigation and demo

This makes the project more than a prototype. It supports both:

1. `Training on historical labeled data`
2. `Predicting risk on new unlabeled transaction data`

---

## Problem Statement

Traditional fraud detection often relies only on basic transaction rules such as:

- large amount
- too many transactions
- suspicious single event

But fraud usually happens as a pattern, not just as a single isolated transaction.

For example:

- one sender may transfer money to many users quickly
- a small set of users may repeatedly send money in a loop
- a risky user may be strongly connected to other suspicious accounts

This project solves that by representing the transaction system as a graph and learning fraud behavior from graph-aware features.

---

## Goal of the Project

The main goal is to build a fraud detection system that:

- uses graph-based intelligence
- learns from historical fraud labels
- predicts fraud probability using machine learning
- improves decisions with an adaptive threshold
- helps investigation through user history and risk analysis

---

## Main Idea

Each transaction is treated as a connection in a network:

- `Nodes` = users
- `Edges` = transactions

Once the graph is built, the system computes user-level features such as:

- how active a user is
- how much they send and receive
- how many unique users they interact with
- how central they are in the network
- how many suspicious neighbors they have
- how quickly their transactions occur

These features are then used by a machine learning model to predict user risk.

---

## Project Workflow

### 1. Data Input

The system supports two types of CSV files.

#### Training CSV

Used for learning from historical transactions.

Required columns:

- `sender_id`
- `receiver_id`
- `amount`
- `timestamp`
- `fraud_label`

`fraud_label` is the historical ground truth.

- `0` means legitimate transaction
- `1` means fraudulent transaction

This labeled data is used to train and evaluate the model.

#### Prediction CSV

Used for scoring new unseen transactions.

Required columns:

- `sender_id`
- `receiver_id`
- `amount`
- `timestamp`

This file does not need `fraud_label`.

The idea is:

- train on old known fraud data
- predict risk on new unknown data

If no training CSV is provided, the system automatically generates synthetic labeled data for demo purposes.

---

### 2. Graph Construction

The system creates a directed transaction graph using `NetworkX`.

- each user becomes a node
- each transaction becomes a directed edge from sender to receiver

Each edge stores:

- transaction amount
- timestamp
- fraud label if available

This graph structure helps capture relationships between users, not just individual records.

---

### 3. Feature Engineering

This is one of the most important parts of the project.

The model does not directly learn from raw transaction rows. Instead, it learns from user-level graph and behavioral features.

#### Basic Features

- `total_transactions`
  Total number of sent + received transactions for a user.

- `total_amount_sent`
  Total money sent by the user.

- `total_amount_received`
  Total money received by the user.

- `avg_transaction_amount`
  Average amount across all transactions linked to the user.

- `max_transaction_amount`
  Maximum transaction amount linked to the user.

- `unique_receivers`
  Number of distinct users this user has sent money to.

- `unique_senders`
  Number of distinct users who have sent money to this user.

#### Graph Features

- `degree_centrality`
  Measures how connected the user is in the graph.

- `clustering_coefficient`
  Measures whether the user sits inside a tightly connected group.

- `in_degree`
  Number of incoming transactions.

- `out_degree`
  Number of outgoing transactions.

#### Advanced Features

- `neighbor_fraud_ratio`
  Fraction of connected users who are suspicious or fraud-linked.

- `transaction_frequency`
  Measures how quickly transactions happen over time.

These features are important because fraud is often visible through behavior patterns, not just raw values.

---

## Why `fraud_label` is Still Needed

This is a very important point.

If the dataset already contains `fraud_label`, we are not simply re-reading the answer. We are using historical labeled data to train a machine learning model.

What the project does is:

1. use old labeled transactions
2. convert them into graph-based user behavior features
3. train a model to learn hidden fraud patterns
4. apply that learned logic to new data

So `fraud_label` is used during training, not during future prediction.

Real-world meaning:

- historical data has labels
- future incoming transactions usually do not
- our model learns from the past to predict the future

---

## Model Training

The project uses:

- `RandomForestClassifier`

Why Random Forest:

- easy to explain in viva
- good for tabular features
- handles nonlinear patterns
- can show feature importance
- works well without deep learning complexity

The training flow is:

1. split data into train and test sets
2. train the model on user features
3. predict fraud probabilities on the test set

The model predicts probability, not just class labels, because the final decision threshold is adjusted later.

---

## Baseline Evaluation

At first, the model uses the normal threshold:

- `threshold = 0.5`

If probability is greater than or equal to `0.5`, the user is marked risky.

The project computes:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

This gives the baseline performance.

---

## RL-Inspired Adaptive Threshold

This project does not build a full reinforcement learning agent.

Instead, it uses an RL-inspired feedback idea:

- check model performance
- calculate reward
- adjust threshold
- repeat for multiple iterations

### Reward Function

```text
reward = (TP * 2) - (FP * 1) - (FN * 3)
```

Meaning:

- catching fraud is rewarded
- false positives are penalized
- false negatives are penalized more heavily

This makes sense because missing fraud is usually more costly than raising an extra alert.

### Threshold Adjustment Logic

- if false positives are high, threshold increases
- if false negatives are high, threshold decreases
- threshold is kept between `0.1` and `0.9`

This improves practical decision making and makes the project more intelligent than a fixed-threshold classifier.

---

## Investigation Layer

This project is not only a model evaluation script.

It also includes an investigation dashboard where you can inspect users in detail.

### Investigation Features

- ranked investigation queue
- searchable user analysis
- complete transaction history
- top counterparties
- suspicious neighbors
- local graph view
- feature snapshot for each user
- explanation of why a user is risky

This makes the project more aligned with real fraud investigation workflows.

---

## Streamlit Dashboard Features

The dashboard includes these main sections:

### Overview

- dataset preview
- model comparison
- graph statistics
- risk distribution
- saved visualization output

### Training Queue

- ranked suspicious users from labeled or synthetic training data

### Prediction Queue

- ranked suspicious users from new unlabeled prediction CSV

### User Search

For any selected user, the dashboard shows:

- fraud probability
- risk level
- priority score
- fraud incidents
- recent activity
- first seen / last seen
- complete transaction history
- behavior summary
- suspicious contacts
- network context
- local graph visualization

### Analytics

- RL threshold history
- top high-risk users
- feature matrix preview

---

## Synthetic Data

If no training CSV is uploaded, the project generates synthetic historical data with fraud patterns for demo.

The synthetic generator includes patterns such as:

- high-value bursts
- fan-out behavior
- circular transfer structures
- stealth fraud
- normal high-value legitimate behavior

This helps show that the system is learning patterns, not only transaction size.

There is also a demo unlabeled prediction file:

- [prediction_demo.csv](C:\Users\Abhinav Nigade\OneDrive\Attachments\Desktop\AIES MINI PROJECT\prediction_demo.csv)

You can upload it in the prediction section of the app.

---

## Project Files

- `fraud_detection.py`
  Main backend logic, graph building, feature extraction, training, evaluation, inference, investigation helpers

- `app.py`
  Streamlit dashboard

- `requirements.txt`
  Required Python libraries

- `prediction_demo.csv`
  Demo file for prediction mode without labels

- `fraud_detection_results.png`
  Saved visualization from pipeline run

---

## Console Output and Visualizations

When the pipeline runs, it produces:

- console evaluation metrics
- threshold tuning history
- before vs after comparison
- a saved visualization image

The saved plot includes:

- baseline confusion matrix
- adaptive confusion matrix
- threshold vs accuracy graph
- RL reward graph
- feature importance chart
- graph visualization

---

## How to Run the Project

## 1. Install Dependencies

```powershell
& 'C:\Users\Abhinav Nigade\AppData\Local\Programs\Python\Python313\python.exe' -m pip install -r requirements.txt
```

## 2. Run the Python Pipeline

```powershell
& 'C:\Users\Abhinav Nigade\AppData\Local\Programs\Python\Python313\python.exe' fraud_detection.py
```

This runs:

- data loading or synthetic generation
- graph construction
- feature extraction
- model training
- baseline evaluation
- adaptive threshold tuning
- final comparison
- result plot generation

## 3. Run the Streamlit Dashboard

```powershell
& 'C:\Users\Abhinav Nigade\AppData\Local\Programs\Python\Python313\python.exe' -m streamlit run app.py --server.port 8502
```

Open:

- [http://127.0.0.1:8502](http://127.0.0.1:8502)

---

## Expected CSV Examples

### Training CSV Example

```csv
sender_id,receiver_id,amount,timestamp,fraud_label
U001,U045,1200,1735689600,0
U003,U019,4800,1735690200,1
U001,U078,300,1735690800,0
```

### Prediction CSV Example

```csv
sender_id,receiver_id,amount,timestamp
U201,U045,1250,1738368000
U201,U078,1320,1738368120
U201,U099,1285,1738368240
```

---

## Real-World Use Case

This project can be presented as a fraud monitoring platform for:

- digital payments
- banking transactions
- fintech wallets
- money transfer systems

Typical workflow:

1. collect historical transaction data with known fraud outcomes
2. train the graph-based fraud model
3. receive new transaction batch without labels
4. score users by risk
5. investigate top suspicious users using dashboard history and graph context

---

## Strengths of the Project

- goes beyond simple rule-based fraud detection
- uses graph relationships between users
- includes machine learning with explainable tabular features
- includes adaptive decision making
- supports both training and prediction workflows
- includes analyst-style investigation features
- suitable for mini project or academic demo

---

## Limitations

- this is not a full production fraud engine
- it does not use deep learning or GNNs
- adaptive threshold is RL-inspired, not a full RL agent
- model quality depends on dataset quality
- graph features are user-level, not full transaction-sequence modeling

These are acceptable limitations for a mini project and keep the design understandable.

---

## Future Improvements

Possible upgrades:

- transaction-level prediction, not only user-level risk
- API backend with Flask or FastAPI
- database storage for case history
- analyst case notes and review actions
- login system
- live streaming transaction monitoring
- SHAP explanations for feature-level model explainability
- anomaly rules for money mule detection and ring detection

---

## Main Functions

- `load_data()`
- `build_graph()`
- `extract_features()`
- `extract_features_for_inference()`
- `train_model()`
- `evaluate_model()`
- `adaptive_threshold()`
- `score_all_users()`
- `build_case_table()`
- `get_user_profile()`
- `run_pipeline()`
- `run_prediction_pipeline()`

---

## Conclusion

This project demonstrates how graph analytics, machine learning, and adaptive decision logic can be combined to build a fraud detection system that is both technically meaningful and practically useful.

It is not just predicting fraud scores. It also helps answer investigation questions such as:

- which users are most suspicious
- why they are risky
- what their history looks like
- who they are connected to
- how to prioritize investigation

That is what makes the project stronger than a basic classification prototype.
#   F r a u d - D e t e c t i o n - u s i n g - G r a p h - B a s e d - F e a t u r e s - w i t h - A d a p t i v e - T h r e s h o l d  
 