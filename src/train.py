import os
import sys
import zipfile
import urllib.request
from pathlib import Path

import joblib
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_URL = (
    "https://archive.ics.uci.edu/static/public/267/"
    "banknote+authentication.zip"
)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"

DATA_FILE = DATA_DIR / "data.csv"
MODEL_FILE = MODEL_DIR / "model.joblib"
METRICS_FILE = MODEL_DIR / "metrics.txt"

TARGET = "class"

FEATURES = [
    "variance",
    "skewness",
    "curtosis",
    "entropy",
]

RANDOM_STATE = 42
TEST_SIZE = 0.20

# Minimum improvement required over the baseline.
MARGIN = 0.10


def download_dataset():
    DATA_DIR.mkdir(exist_ok=True)

    if DATA_FILE.exists():
        return

    zip_path = DATA_DIR / "banknote.zip"

    print("Downloading dataset...")

    urllib.request.urlretrieve(DATA_URL, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)

    extracted_file = DATA_DIR / "data_banknote_authentication.txt"

    if not extracted_file.exists():
        raise FileNotFoundError(
            "Expected dataset file was not found after extraction."
        )

    df = pd.read_csv(
        extracted_file,
        header=None,
        names=FEATURES + [TARGET],
    )

    df.to_csv(DATA_FILE, index=False)

    zip_path.unlink()
    extracted_file.unlink()


def validate_dataset(df):
    required_columns = FEATURES + [TARGET]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        print(f"VALIDATION FAILED: Missing columns: {missing}")
        sys.exit(1)

    if df.empty:
        print("VALIDATION FAILED: Dataset is empty.")
        sys.exit(1)

    if df[TARGET].isna().any():
        print("VALIDATION FAILED: Target contains missing values.")
        sys.exit(1)

    if df[FEATURES].isna().any().any():
        print("VALIDATION FAILED: Features contain missing values.")
        sys.exit(1)

    print("Dataset validation passed.")


def main():
    download_dataset()

    df = pd.read_csv(DATA_FILE)

    validate_dataset(df)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # ---------------------------------------------------------
    # Failure demonstration support
    # ---------------------------------------------------------
    # Set FAILURE_MODE=quality when demonstrating Failure A.
    failure_mode = os.getenv("FAILURE_MODE", "").lower()

    if failure_mode == "quality":
        print("FAILURE MODE: Training on a deliberately tiny subset.")
        X_train = X_train.iloc[:10]
        y_train = y_train.iloc[:10]

    # ---------------------------------------------------------
    # Baseline
    # ---------------------------------------------------------
    baseline = DummyClassifier(
        strategy="most_frequent"
    )

    baseline.fit(X_train, y_train)

    baseline_predictions = baseline.predict(X_test)

    baseline_score = accuracy_score(
        y_test,
        baseline_predictions
    )

    # ---------------------------------------------------------
    # Candidate model
    # ---------------------------------------------------------
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    model_score = accuracy_score(
        y_test,
        predictions
    )

    required_score = baseline_score + MARGIN

    gate_passed = model_score >= required_score

    print()
    print("===== MODEL EVALUATION =====")
    print(f"Baseline accuracy : {baseline_score:.4f}")
    print(f"Model accuracy    : {model_score:.4f}")
    print(f"Required accuracy : {required_score:.4f}")
    print(f"Margin            : {MARGIN:.2f}")
    print(f"Quality gate      : {'PASSED' if gate_passed else 'FAILED'}")
    print("============================")
    print()

    MODEL_DIR.mkdir(exist_ok=True)

    with open(METRICS_FILE, "w") as file:
        file.write(
            f"Dataset: UCI Banknote Authentication\n"
            f"Metric: Accuracy\n"
            f"Baseline score: {baseline_score:.4f}\n"
            f"Model score: {model_score:.4f}\n"
            f"Improvement margin: {MARGIN:.2f}\n"
            f"Required score: {required_score:.4f}\n"
            f"Gate result: {'PASSED' if gate_passed else 'FAILED'}\n"
        )

    if not gate_passed:
        print(
            "QUALITY GATE FAILED. Model package will not be published."
        )
        sys.exit(1)

    joblib.dump(model, MODEL_FILE)

    print(f"Model saved to: {MODEL_FILE}")


if __name__ == "__main__":
    main()