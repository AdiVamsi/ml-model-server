"""
Validate the locally saved Iris classifier.

This script is intentionally small and boring: it loads the pickle that the
training script writes to artifacts/model.pkl, evaluates it on the same
deterministic held-out Iris split used during training, and exits with a
process status that CI/CD can understand.

Exit codes:
    0 = model accuracy is high enough
    1 = model accuracy is below the configured threshold, or validation cannot
        run because the model artifact is missing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# The default quality gate for this repo. Keeping this as a named constant makes
# the policy obvious at the top of the file instead of burying a magic number in
# the validation logic.
MIN_ACCURACY = 0.90

# The training script writes this plain pickle for serving. The validator checks
# that exact serving artifact, not the MLflow model directory, because this is
# the file the API will actually load.
MODEL_PATH = Path("artifacts") / "model.pkl"

# Match train.py so validation is repeatable and measures the same held-out
# examples every time. This avoids a noisy gate where a random split could pass
# on one run and fail on the next for reasons unrelated to the saved model.
TEST_SIZE = 0.2
RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    """Parse an optional threshold override for smoke-testing pass/fail paths."""
    parser = argparse.ArgumentParser(
        description="Validate artifacts/model.pkl against a held-out Iris split."
    )
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=MIN_ACCURACY,
        help=f"Minimum acceptable accuracy. Defaults to {MIN_ACCURACY:.2f}.",
    )
    return parser.parse_args()


def load_held_out_iris_split() -> tuple[object, object]:
    """Return the deterministic held-out feature matrix and labels."""
    X, y = load_iris(return_X_y=True)

    # We only need X_test/y_test for validation, but train_test_split returns the
    # training side as well. The underscore names make that intentional: the
    # saved model is already trained, so no fitting happens in this script.
    _X_train, X_test, _y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    return X_test, y_test


def main() -> int:
    args = parse_args()
    min_accuracy = args.min_accuracy

    # A missing model is a validation failure, not an unexpected traceback. This
    # keeps CI logs direct: run train.py first, then validate the artifact.
    if not MODEL_PATH.exists():
        print(f"Validation failed: missing model artifact at {MODEL_PATH}")
        return 1

    model = joblib.load(MODEL_PATH)
    X_test, y_test = load_held_out_iris_split()

    # The model only needs to implement predict(), which is the serving-time
    # contract we care about for this artifact.
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    if accuracy >= min_accuracy:
        print(
            f"Validation passed: accuracy={accuracy:.3f} "
            f">= min_accuracy={min_accuracy:.3f}"
        )
        return 0

    print(
        f"Validation failed: accuracy={accuracy:.3f} "
        f"< min_accuracy={min_accuracy:.3f}"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
