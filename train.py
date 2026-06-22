"""
train.py — Phase 0 of ml-model-server.

Trains a RandomForest on Iris, tracks the run in MLflow (params, metrics,
model), and also dumps a plain joblib pickle the FastAPI server will load.

This file is the "the model already exists" stand-in for Scott's data
scientists. Everything downstream — serving, Docker, CI/CD, deploy — is
the actual DevOps/ML job.
"""

from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# --- Config -----------------------------------------------------------------
ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
EXPERIMENT_NAME = "iris-rf"
REGISTERED_MODEL_NAME = "iris-classifier"
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
N_ESTIMATORS = 100
MAX_DEPTH = 3
RANDOM_STATE = 42


def main() -> None:
    # 1. Data
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # 2. Use the local MLflow store and point MLflow at an experiment
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # 3. One run = one tracked training attempt
    with mlflow.start_run() as run:
        model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            random_state=RANDOM_STATE,
        )
        model.fit(X_train, y_train)

        # 4. Evaluate
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")

        # 5. Log inputs + outputs to MLflow (the tracking story)
        mlflow.log_param("n_estimators", N_ESTIMATORS)
        mlflow.log_param("max_depth", MAX_DEPTH)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_macro", f1)
        model_info = mlflow.sklearn.log_model(model, name="model")
        registered_model = mlflow.register_model(
            model_uri=model_info.model_uri,
            name=REGISTERED_MODEL_NAME,
        )

        # 6. Also save a plain pickle for the API to load at serve time
        ARTIFACTS_DIR.mkdir(exist_ok=True)
        joblib.dump(model, MODEL_PATH)

        print(f"accuracy={acc:.3f}  f1_macro={f1:.3f}")
        print(f"mlflow_run_id={run.info.run_id}")
        print(
            f"registered_model={REGISTERED_MODEL_NAME} "
            f"version={registered_model.version}"
        )
        print(f"saved model -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
