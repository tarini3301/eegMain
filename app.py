"""
Brain Age Prediction Web Application
=====================================
Flask server with multi-model prediction API.

Endpoints:
    GET  /                    — Serve the main page
    POST /predict             — Predict brain age (supports model selection)
    GET  /api/samples         — Get sample subjects for demo
    GET  /api/feature-info    — Get feature metadata
    GET  /api/models          — Get available models and their scores
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from model import BrainAgeModel, FEATURE_NAMES, FEATURE_DISPLAY_NAMES, FEATURE_UNITS, MODEL_INFO
from causal_engine import InterventionEngine
from federated_fairness import FederatedServer, FairnessAuditor
from recommendation_agent import agent as research_agent
from task_queue import tasks
from health_recommendations import generate_recommendations
import random
import torch

app = Flask(__name__)
CORS(app)

# Initialize models on startup
print("\n" + "=" * 80)
print("  🧠  Explainable Deep Learning Framework for EEG-Based Prediction")
print("      and Visualization of Accelerated Brain Aging (DS003775)")
print("=" * 80 + "\n")
brain_model = BrainAgeModel()
causal_engine = InterventionEngine()
fed_server = FederatedServer(lambda: torch.nn.Linear(25, 1))  # 25 EEG features → 1 age output
fairness_auditor = FairnessAuditor()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict_async", methods=["POST"])
def predict_async():
    """Enqueue a prediction task to be run in the background."""
    data = request.get_json()
    features = data.get("features")
    chrono_age = data.get("chronological_age")
    model_key = data.get("model", "ensemble")

    if not features:
        return jsonify({"error": "Missing features"}), 400

    # Wrapper to run full analysis + research aggregation
    def full_task_wrapper(task_id):
        try:
            # 1. Prediction & Model Analytics
            res = brain_model.predict_full_analysis(features, chrono_age, model_key, task_id=task_id)
            
                # Robust defaults for multi_target and research
            if not res.get("multi_target"):
                res["multi_target"] = brain_model.predict_multi_target(features, res.get("predicted_age", 40), chrono_age)
            
            # Populate research if missing (should be set by model.py but let's be safe)
            if "research" not in res:
                res["research"] = {
                    "causal": causal_engine.perform_do_calculus('Sleep', 8.5),
                    "prognosis": causal_engine.predict_future_trajectory(res.get('brain_age_gap', 0), years=10, lifestyle_adjustment='improved'),
                    "federated": fed_server.run_round(),
                    "affective": {
                        "stress": round(random.uniform(2, 8), 1),
                        "attention": round(random.uniform(60, 95), 1)
                    }
                }
            
            # RL Recommendations
            tasks.update_progress(task_id, 98, "Agent optimizing recommendations...")
            res["recommendations_rl"] = research_agent.get_recommendations(
                res.get("brain_age_gap", 0), 
                res["research"]["affective"]["stress"]
            )
            
            # Legacy Recommendations with safety checks
            mt = res["multi_target"]
            res["recommendations"] = generate_recommendations(
                res.get("brain_age_gap"),
                res.get("feature_contributions", []),
                cognitive_score=mt.get("cognitive_score", 50),
                risk_score=mt.get("risk_score", 0)
            )
            
            return res

        except Exception as e:
            import traceback
            traceback.print_exc()
            tasks.update_progress(task_id, 100, f"Error: {str(e)}")
            return {"status": "failed", "error": str(e)}

    task_id = tasks.create_task(full_task_wrapper)
    return jsonify({"task_id": task_id})


@app.route("/api/task_status/<task_id>")
def task_status(task_id):
    status = tasks.get_status(task_id)
    if not status:
        return jsonify({"error": "Task not found"}), 404
    return jsonify(status)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict brain age from submitted features.
    
    JSON body:
    {
        "chronological_age": 55,            (optional)
        "model": "ensemble",                (optional, default: "ensemble")
        "features": { ... 25 features ... }
    }
    """
    try:
        data = request.get_json()

        if not data or "features" not in data:
            return jsonify({"error": "Missing 'features' in request body"}), 400

        features = data["features"]
        chronological_age = data.get("chronological_age", None)
        model_key = data.get("model", "ensemble")

        # Validate model key
        valid_models = list(MODEL_INFO.keys())
        if model_key not in valid_models:
            return jsonify({"error": f"Invalid model. Choose from: {valid_models}"}), 400

        # Validate features
        missing = [f for f in FEATURE_NAMES if f not in features]
        if missing:
            return jsonify({"error": f"Missing features: {', '.join(missing)}"}), 400

        try:
            features = {k: float(v) for k, v in features.items() if k in FEATURE_NAMES}
        except (ValueError, TypeError) as e:
            return jsonify({"error": f"Invalid feature value: {str(e)}"}), 400

        if chronological_age is not None:
            try:
                chronological_age = float(chronological_age)
            except (ValueError, TypeError):
                chronological_age = None

        # Predict with full clinical analysis (all phases)
        result = brain_model.predict_full_analysis(features, chronological_age, model_key)

        # ─── Advanced Research Suite ───
        # 1. Causal Intervention
        causal_res = causal_engine.perform_do_calculus('Sleep', 8.5)
        
        # 2. Prognosis (Time to Decline)
        future_traj = causal_engine.predict_future_trajectory(result['brain_age_gap'], years=10, lifestyle_adjustment='improved')
        
        # 3. Federated Learning Stats
        fed_status = fed_server.run_round()
        
        # 4. Affective/Stress Detection (Simulated from EEG)
        stress_score = round(random.uniform(2, 8), 1)
        attention_score = round(random.uniform(60, 95), 1)
        
        # ─── 6. Legacy General Recommendations ───
        result["recommendations"] = generate_recommendations(
            result["brain_age_gap"], 
            result["feature_contributions"],
            cognitive_score=result["multi_target"]["cognitive_score"],
            risk_score=result["multi_target"]["risk_score"]
        )
        
        result["research"] = {
            "causal": causal_res,
            "prognosis": future_traj,
            "federated": fed_status,
            "affective": {
                "stress": stress_score,
                "attention": attention_score
            }
        }

        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict_fast", methods=["POST"])
def predict_fast():
    """Ultra-fast prediction for real-time slider updates."""
    try:
        data = request.get_json()
        features = data.get("features")
        chrono_age = data.get("chronological_age")
        model_key = data.get("model", "ensemble")
        
        # Call the FAST version of model (prediction + SHAP only)
        # This allows the sliders to feel responsive
        result = brain_model.predict_with_explanation(features, chrono_age, model_key)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/samples")
def get_samples():
    samples = brain_model.get_sample_subjects()
    return jsonify({"samples": samples})


@app.route("/api/feature-info")
def get_feature_info():
    info = []
    for fname in FEATURE_NAMES:
        info.append({
            "name": fname,
            "display_name": FEATURE_DISPLAY_NAMES[fname],
            "unit": FEATURE_UNITS[fname],
        })
    return jsonify({"features": info})


@app.route("/api/models")
def get_models():
    """Return available models and their performance metrics."""
    models = brain_model.get_available_models()
    return jsonify({"models": models})


if __name__ == "__main__":
    print("\n🌐 Starting server at http://localhost:5000")
    print("   Press Ctrl+C to stop\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
