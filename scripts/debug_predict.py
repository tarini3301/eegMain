import traceback
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import BrainAgeModel

def debug():
    try:
        model = BrainAgeModel()
        samples = model.get_sample_subjects()
        if not samples:
            print("No samples found")
            return
        
        s = samples[0]
        print(f"Testing subject {s['subject_id']} (Age {s['chronological_age']})")
        
        result = model.predict_full_analysis(s["features"], s["chronological_age"], "ensemble")
        
        print("\nResult Structure Check:")
        for key in ["predicted_age", "multi_target", "disease_risk", "counterfactuals", "tcav_scores"]:
            val = result.get(key)
            print(f"  - {key}: {type(val)} {'(None!)' if val is None else ''}")
            if isinstance(val, dict):
                print(f"    Keys: {list(val.keys())}")
        
        # Test the line that likely failed in app.py
        print("\nMocking app.py access:")
        cog = result["multi_target"]["cognitive_score"]
        risk = result["multi_target"]["risk_score"]
        print(f"  Cognitive: {cog}, Risk: {risk}")
        print("SUCCESS: Structure is correct.")

    except Exception as e:
        print("\nERROR DETECTED:")
        traceback.print_exc()

if __name__ == "__main__":
    debug()
