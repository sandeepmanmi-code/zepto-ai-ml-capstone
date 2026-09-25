from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "best_pipeline.joblib"
)

model = joblib.load(MODEL_PATH)

# Raw-style input. No manual imputation, encoding or scaling.
raw_passenger = pd.DataFrame(
    [
        {
            "pclass": 1,
            "sex": "female",
            "age": 29,
            "sibsp": 0,
            "parch": 0,
            "fare": 100.0,
            "embarked": "S",
        }
    ]
)

prediction = model.predict(raw_passenger)
probability = model.predict_proba(raw_passenger)[:, 1]

print("Raw input:")
print(raw_passenger)

print("\nPrediction:")
print(prediction)

print("\nProbability of survival:")
print(probability)
