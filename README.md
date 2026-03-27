
# 🧠 NeuroAge: Explainable Deep Learning for EEG-Based Brain Age Prediction

> **A clinical-grade, research-oriented platform for predicting functional brain age from Resting-State EEG spectral power features, powered by 17 ML/DL models, dual XAI (SHAP + LIME), causal AI, and a reinforcement learning recommendation agent.**

---

## 🏗️ Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                     Flask Web Server (app.py)                 │
├──────────────────────────────────────────────────────────────┤
│  model.py          │ 17 ML/DL models + SHAP + LIME + Monte  │
│                    │ Carlo Dropout uncertainty estimation      │
├────────────────────┼─────────────────────────────────────────┤
│  advanced_models.py│ EEG Transformer, GNN (GCN/GAT),        │
│                    │ Self-Supervised (SimCLR), Masked EEG     │
├────────────────────┼─────────────────────────────────────────┤
│  causal_engine.py  │ Structural Causal Models, Do-Calculus,  │
│                    │ 10-Year Prognostic Trajectories          │
├────────────────────┼─────────────────────────────────────────┤
│  recommendation    │ Q-Learning RL Agent for lifestyle        │
│  _agent.py         │ optimization (Sleep, Stress, Nutrition)  │
├────────────────────┼─────────────────────────────────────────┤
│  federated_        │ FedAvg simulation, DANN domain           │
│  fairness.py       │ adaptation, Fairness Auditing            │
├────────────────────┼─────────────────────────────────────────┤
│  task_queue.py     │ Async background task manager for        │
│                    │ high-latency XAI computations             │
├────────────────────┼─────────────────────────────────────────┤
│  health_           │ Personalized clinical recommendations    │
│  recommendations.py│ based on feature contributions            │
├──────────────────────────────────────────────────────────────┤
│  Frontend: HTML + CSS + JS + Chart.js + Three.js (3D Brain)  │
└──────────────────────────────────────────────────────────────┘
```

---

## 🔬 Key Features

### 1. Multi-Model Ensemble Prediction
- **17 trained models**: Random Forest, Gradient Boosting, SVR, Lasso, Ridge, ElasticNet, KNN, Neural Network, Bayesian Ridge, Decision Tree, Extra Trees, AdaBoost, Linear Regression, Huber Regressor, Deep Neural Net (PyTorch), EEG Transformer, Graph Neural Net (GNN)
- **Ensemble mode** with weighted averaging across top performers
- **Model comparison table** with R², CV-R², and MAE

### 2. Dual Explainability (SHAP + LIME)
- **SHAP** (SHapley Additive exPlanations): Global feature importance
- **LIME** (Local Interpretable Model-agnostic Explanations): Local per-prediction explanations
- **Combined visualization**: Side-by-side SHAP vs. LIME bar chart for 25 EEG features

### 3. Causal AI & Do-Calculus
- **Structural Causal Models** (SCM) for brain aging pathways
- **Intervention simulation**: `do(Sleep = 8.5h)` → causal effect on brain age
- **10-year prognostic trajectories** under different lifestyle conditions

### 4. Reinforcement Learning Recommendation Agent
- **Q-Learning** policy for lifestyle optimization
- Reward-optimized suggestions: Sleep, Stress, Nutrition, Meditation, Exercise
- State-based recommendations adapting to Brain Age Gap severity

### 5. Advanced Neural Architectures
- **EEG Transformer**: Spectral-spatial attention for long-range EEG dependencies
- **Graph Neural Network (GNN)**: GCN/GAT for electrode-level connectivity
- **Self-Supervised Learning**: Contrastive (SimCLR) and Masked EEG Modeling (BERT-style)
- **Cross-Modal Fusion**: Transformer-based EEG + Cognitive + Lifestyle fusion

### 6. Ethical & Distributed AI
- **Federated Learning**: Simulated FedAvg across hospital nodes
- **Domain Adaptation**: DANN with gradient reversal for cross-site generalization
- **Fairness Auditing**: Bias analysis across gender and age groups

### 7. Clinical Intelligence Dashboard
- **Brain Age Gauge**: Visual dial showing predicted vs. chronological age
- **Spatial Brain Heatmap**: Region-level SHAP aggregation (Frontal, Temporal, Parietal, Central, Occipital)
- **3D Brain Mapping**: Interactive Three.js neuro-spatial visualization
- **Longitudinal Patient History**: Local tracking with aging trend detection
- **Export Clinical Report (PDF)**: One-click professional report generation

### 8. Asynchronous Analysis Workbench
- **Background task queue** for heavy XAI computations
- **Real-time progress overlay**: 4-step visual pipeline (Neural Prediction → XAI → Causal Simulation → RL Agent)

---

## 📋 Prerequisites
- **Python 3.11+**
- **pip** (Python package manager)

## 🚀 Getting Started

### 1. (Optional) Create a Virtual Environment
```powershell
python -m eegenv venv
.\eegenv\Scripts\activate
```

### 2. Install Required Packages
```powershell
pip install -r requirements.txt
```

### 3. Start the Flask Server
```powershell
py -3.11 app.py
```

### 4. Access the Platform
Open your browser and navigate to: **[http://localhost:5000](http://localhost:5000)**

---

## ⚡ How to Use

1. **Quick Load**: Click any **Sample Subject** (e.g., `sub-004`) to populate all 25 EEG features
2. **Select Model**: Choose a prediction algorithm (e.g., Random Forest, Lasso, or Ensemble)
3. **Predict**: Click **"Predict Brain Age"** — the **Analysis Workbench** overlay shows real-time progress
4. **Review Results**:
   - **Brain Age Gap** and **Confidence Intervals** (Monte Carlo Dropout)
   - **Risk Classification**: Normal / Accelerated / Decelerated aging
   - **SHAP vs. LIME** dual explainability chart
   - **Personalized Recommendations** with RL-optimized lifestyle actions
5. **Research Suite**: Switch to the Research tab for 3D Brain Mapping, Causal Intervention, and 10-Year Prognosis

---

## 📂 Project Structure

```
maj_pro/
├── app.py                    # Flask server with API endpoints
├── model.py                  # 17 ML/DL models + XAI + uncertainty
├── advanced_models.py        # Transformer, GNN, SSL architectures
├── causal_engine.py          # SCM + Do-Calculus + trajectory prediction
├── recommendation_agent.py   # RL Q-Learning recommendation agent
├── federated_fairness.py     # FedAvg + DANN + Fairness Auditing
├── health_recommendations.py # Clinical recommendation generator
├── task_queue.py             # Async background task manager
├── requirements.txt          # Python dependencies
├── synthetic_dataset.csv     # EEG dataset (DS003775)
├── trained_models.joblib     # Pre-trained model cache
├── trained_scaler.joblib     # Feature scaler
├── templates/
│   └── index.html            # Main frontend template
└── static/
    ├── script.js             # Frontend logic (1500+ lines)
    └── style.css             # Dark-mode clinical UI
```

---

## 🧪 Dataset

Based on the **OpenNeuro DS003775** dataset: Resting-state EEG recordings.

**25 spectral power features** across 5 brain regions (Frontal, Central, Temporal, Parietal, Occipital) × 5 frequency bands (Delta, Theta, Alpha, Beta, Gamma).

---

## 📖 Technical References

- **SHAP**: Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions" (NeurIPS 2017)
- **LIME**: Ribeiro et al., "Why Should I Trust You?" (KDD 2016)
- **FedAvg**: McMahan et al., "Communication-Efficient Learning of Deep Networks" (AISTATS 2017)
- **DANN**: Ganin et al., "Domain-Adversarial Training of Neural Networks" (JMLR 2016)
- **Do-Calculus**: Pearl, "Causality" (Cambridge University Press, 2009)
