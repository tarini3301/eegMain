import time
import traceback
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import BrainAgeModel, generate_synthetic_dataset

def main():
    try:
        print("Initializing model...")
        brain_model = BrainAgeModel()
        
        # synthetic data
        df = generate_synthetic_dataset(n_subjects=1)
        features = df.iloc[0].drop(["subject_id", "gender", "chronological_age"]).to_dict()
        chrono_age = float(df.iloc[0]["chronological_age"])
        
        print("Running prediction...")
        start = time.time()
        res = brain_model.predict_full_analysis(features, chrono_age, model_key="ensemble")
        end = time.time()
        print(f"Success! Time taken: {end - start:.2f} seconds")
    except Exception as e:
        print("ERROR ENCOUNTERED:")
        traceback.print_exc()

if __name__ == "__main__":
    main()
