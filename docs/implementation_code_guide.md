# 💻 NeuroAge: Implementation Code Guide

> **A comprehensive technical deep-dive into the core implementation logic, class structures, and algorithmic code blocks of the NeuroAge project.**

---

## 1. Backend: Core Intelligence & Inference

### 🟢 `model.py`: The Predictive Engine
This file contains the `BrainAgeModel` class, which manages the lifecycle of 17 distinct machine learning algorithms.

*   **Ensemble Logic**: 
    ```python
    # Computes a weighted average based on model performance (MAE)
    ensemble_pred = sum(p * w for p, w in zip(preds, weights)) / sum(weights)
    ```
    The design utilizes a **weighted meta-learner** approach. Instead of a simple average, models with lower Cross-Validated MAE (Mean Absolute Error) are given higher "voting power."
*   **Uncertainty Estimation**: 
    Implements **Monte Carlo (MC) Dropout**. By keeping dropout active during inference and running 50+ iterations, the code calculates the variance of predictions to provide a `+/-` confidence interval.
*   **Explainability Scaling**:
    ```python
    scale_factor = np.mean(np.abs(shap_values)) / (np.mean(np.abs(lime_coefs)) + 1e-9)
    aligned_lime = lime_coefs * scale_factor
    ```
    This critical block ensures that LIME (local) and SHAP (global) values are algebraically comparable for the frontend charts.

### 🟣 `advanced_models.py`: Deep Neural Architectures
*   **Graph Neural Network (GNN)**:
    Uses `EEGGraphConv` to propagate spectral features across a spatial graph. The adjacency matrix (`adj`) defines which brain regions are physically connected (e.g., Frontal to Parietal).
*   **EEG Transformer**:
    Implements `nn.TransformerEncoderLayer` with multi-head attention to capture "Cross-Band" dependencies, such as how Alpha power in the Occipital lobe relates to Gamma spikes in the Frontal lobe.
*   **Self-Supervised Learning (SSL)**:
    Includes a `ContrastiveEncoder` (SimCLR style) that learns to map high-dimensional EEG features into a latent space where healthy and aging brain signatures are naturally clustered without needing explicit labels.

---

## 2. Backend: Causal & Agentic Engines

### 🔵 `causal_engine.py`: Structural Causal Modeling
This module moves beyond correlation to **Causation**.
*   **SCM Simulation**: 
    Defines a Directed Acyclic Graph (DAG) using linear coefficients (e.g., `S_to_A = 2.5`). 
*   **Do-Calculus Implementation**:
    ```python
    def perform_do_calculus(self, target_variable, target_value):
        # Forces the variable (e.g., Sleep=8.5) and recalculates the DAG
        results = [self.scm.simulate(interventions={target_variable: target_value}) for _ in range(100)]
    ```
    It compares 100 simulations of the "intervention" vs the "baseline" to extract the **Average Treatment Effect (ATE)**.

### 🟠 `recommendation_agent.py`: Reinforcement Learning
*   **Q-Learning Engine**: 
    Uses a standard **Q-Table** where **States** are {Younger, Normal, Accelerated} and **Actions** are {Sleep+, Stress-, Nutrition+, etc.}.
*   **Policy Selection**:
    ```python
    top_indices = np.argsort(q_values)[::-1][:3] # Selects highest-reward actions
    ```
    The code dynamically adjusts Q-values based on a heuristic "Stress Index" derived from real-time EEG Theta/Alpha ratios.

---

## 3. Infrastructure: Asynchrony

### 🔘 `task_queue.py`: Async Management
To prevent the Flask server from timing out during heavy SHAP/DeepExplainer computations, we implemented a custom asynchronous manager.
*   **Mechanism**: Uses `threading.Thread(daemon=True)` to detach long-running analysis jobs from the main request/response cycle.
*   **State Management**: Tracks `pending`, `running`, `completed`, and `failed` statuses using a `uuid` system, allowing the frontend to poll for real-time progress.

---

## 4. Frontend: Visual Intelligence

### 🟡 `static/script.js`: State & Visualization
With over 1500 lines of code, this is the most complex frontend component.
*   **Three.js Brain Rendering**: 
    Initializes a WebGL scene, loads a 3D cerebral mesh, and dynamically updates vertex colors based on the **SHAP Importance Map**.
    ```javascript
    // Conceptual vertex coloring logic
    const intensity = Math.abs(shapValue) * scalingFactor;
    color.setHSL(0, 1, 0.5); // Color mapping from Green to Red
    ```
*   **Chart.js Lifecycles**: 
    Manages the creation, destruction, and updating of 6+ distinct chart types (Radars, Bar, Trajectory, Gauges).
*   **Fast-Prediction Polling**:
    Implements a debounced `triggerFastPrediction()` function. Every time a slider moves, it hits `/api/predict_fast` to provide immediate visual feedback on the Brain Age Gauge.

---

## 5. Summary of Implementation Principles

1.  **Modularity**: Every intelligence layer (Inference, Causal, RL, Ethics) is isolated into its own Python module, making the system easy to extend.
2.  **Safety**: Extensive try-catch blocks in `app.py` ensure that if a complex module fails (e.g., GNN gradient error), the rest of the clinical dashboard remains functional.
3.  **Transparency**: The codebase is not just designed to predict; it is designed to **Explain**, with half of the implementation dedicated purely to extraction and visualization of inner-model logic.

---
*Documentation of the NeuroAge codebase version 2.0.*
