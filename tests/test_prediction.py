import sys
from pathlib import Path

import pytest
import joblib


BASE_DIR = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(BASE_DIR / "src")
)

from predict import predict


MODEL_FILE = BASE_DIR / "model" / "model.joblib"


def test_model_can_be_loaded():

    model = joblib.load(MODEL_FILE)

    assert model is not None


def test_valid_sample_prediction():

    sample = {
        "variance": 3.6216,
        "skewness": 8.6661,
        "curtosis": -2.8073,
        "entropy": -0.44699,
    }

    result = predict(sample)

    assert isinstance(result, int)


def test_missing_feature_is_rejected():

    sample = {
        "variance": 3.6216,
        "skewness": 8.6661,
        "curtosis": -2.8073,
    }

    with pytest.raises(
        ValueError,
        match="Missing required feature"
    ):
        predict(sample)