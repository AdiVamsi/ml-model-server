"""Model loading and prediction helpers for the API.

The FastAPI layer should not know how the model artifact is stored. It should
only ask this module to load the artifact during startup and to make predictions
once requests arrive.
"""

from __future__ import annotations

from pathlib import Path

import joblib

MODEL_PATH = Path("artifacts") / "model.pkl"
CLASS_NAMES = ("setosa", "versicolor", "virginica")

_model = None


def load_model(model_path: Path = MODEL_PATH) -> None:
    """Load the trained model artifact into module-level memory once."""
    global _model

    if _model is None:
        _model = joblib.load(model_path)


def is_loaded() -> bool:
    """Return whether the startup lifespan has successfully loaded the model."""
    return _model is not None


def predict(features: list[float]) -> dict[str, object]:
    """Predict the Iris class and class probabilities for one feature vector."""
    if _model is None:
        raise RuntimeError("Model is not loaded")

    # scikit-learn estimators expect a 2D array shaped like
    # [sample_count, feature_count]. The API receives exactly one Iris sample.
    sample = [features]
    predicted_class_id = int(_model.predict(sample)[0])
    probabilities = _model.predict_proba(sample)[0]

    return {
        "predicted_class": CLASS_NAMES[predicted_class_id],
        "probabilities": {
            class_name: float(probability)
            for class_name, probability in zip(CLASS_NAMES, probabilities)
        },
    }
