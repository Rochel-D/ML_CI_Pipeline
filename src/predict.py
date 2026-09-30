import sys
from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_FILE = BASE_DIR / "model" / "model.joblib"

FEATURES = [
    "variance",
    "skewness",
    "curtosis",
    "entropy",
]


def predict(sample):
    missing = [
        feature
        for feature in FEATURES
        if feature not in sample
    ]

    if missing:
        raise ValueError(
            f"Missing required feature(s): {missing}"
        )

    model = joblib.load(MODEL_FILE)

    X = pd.DataFrame(
        [[sample[feature] for feature in FEATURES]],
        columns=FEATURES,
    )

    prediction = model.predict(X)

    if len(prediction) != 1:
        raise ValueError(
            "Prediction must contain exactly one output."
        )

    return int(prediction[0])


if __name__ == "__main__":

    sample = {
        "variance": 3.6216,
        "skewness": 8.6661,
        "curtosis": -2.8073,
        "entropy": -0.44699,
    }

    try:
        result = predict(sample)
        print(f"Prediction: {result}")

    except Exception as error:
        print(f"Prediction failed: {error}")
        sys.exit(1)