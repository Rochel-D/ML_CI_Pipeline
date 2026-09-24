from pathlib import Path

import pandas as pd
import joblib

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier


BASE_DIR = Path(__file__).resolve().parents[1]

DATA_FILE = BASE_DIR / "data" / "data.csv"
MODEL_FILE = BASE_DIR / "model" / "model.joblib"

FEATURES = [
    "variance",
    "skewness",
    "curtosis",
    "entropy",
]

TARGET = "class"

RANDOM_STATE = 42
TEST_SIZE = 0.20


def main():

    df = pd.read_csv(DATA_FILE)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    baseline = DummyClassifier(
        strategy="most_frequent"
    )

    baseline.fit(X_train, y_train)

    baseline_score = accuracy_score(
        y_test,
        baseline.predict(X_test)
    )

    model = joblib.load(MODEL_FILE)

    model_score = accuracy_score(
        y_test,
        model.predict(X_test)
    )

    print(f"Baseline accuracy: {baseline_score:.4f}")
    print(f"Model accuracy: {model_score:.4f}")


if __name__ == "__main__":
    main()