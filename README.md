---
title: ML Model Server
emoji: "🌸"
colorFrom: green
colorTo: blue
sdk: docker
app_port: 8080
pinned: false
---

# ml-model-server

A production-shaped ML serving project around a deliberately simple Iris
classifier. The model is small on purpose; the value is the serving spine around
it: training, local MLflow tracking and registration, artifact validation,
FastAPI serving, metrics, tests, and a Docker image that can run the API.

## What It Demonstrates

Train -> track -> register -> validate -> serve -> containerize -> test ->
deploy scaffold.

Live demo: https://AdiVamsiSai-ml-model-server.hf.space/docs

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Liveness check. Returns `{"status":"ok"}` if the app process is up. |
| `GET` | `/ready` | Readiness check. Returns `200` only when `app/model.py` has loaded the model. |
| `POST` | `/predict` | Accepts four named float Iris features and returns a predicted class plus class probabilities. |
| `GET` | `/metrics` | Prometheus text counters for HTTP requests and prediction latency totals/counts. |

Verified prediction request:

```bash
curl -X POST http://127.0.0.1:8080/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "sepal_length": 6.3,
    "sepal_width": 3.3,
    "petal_length": 6.0,
    "petal_width": 2.5
  }'
```

Verified response from the current model artifact, rounded for readability:

```json
{
  "predicted_class": "virginica",
  "probabilities": {
    "setosa": 0.0,
    "versicolor": 0.0031,
    "virginica": 0.9969
  }
}
```

## Project Layout

```text
.
├── app/
│   ├── main.py              # FastAPI app, schemas, lifespan, endpoints, metrics
│   └── model.py             # loads artifacts/model.pkl and runs predictions
├── artifacts/model.pkl      # local serving artifact produced by train.py
├── scripts/validate_model.py# deterministic quality gate for model.pkl
├── tests/unit/test_api.py   # TestClient scaffold for API behavior
├── train.py                 # train, log, register, and save the Iris classifier
├── Dockerfile               # multi-stage Python 3.14 slim image build
├── docker-compose.localstack.yml
├── iam/                     # starter IAM policies for the LocalStack/AWS scaffold
├── pyproject.toml           # uv project metadata and dependencies
└── .github/workflows/       # CI, train, and CD scaffold workflows
```

## Architecture

```mermaid
flowchart TB
    subgraph TRAIN["Training"]
        D[Iris dataset] --> T[train.py<br/>RandomForestClassifier]
        T --> L[MLflow local tracking<br/>sqlite:///mlflow.db]
        T --> R[MLflow Model Registry<br/>iris-classifier]
        T --> A[artifacts/model.pkl]
    end

    subgraph GATE["Validation"]
        A --> V[scripts/validate_model.py<br/>accuracy >= 0.90]
        V -->|pass| S[servable artifact]
        V -->|fail| B[exit 1]
    end

    subgraph SERVE["Serving"]
        S --> M[app/model.py<br/>load once at startup]
        M --> API[app/main.py<br/>FastAPI]
        API --> E[/health /ready /predict /metrics]
    end

    subgraph PACKAGE["Packaging"]
        API --> IMG[Dockerfile<br/>python:3.14-slim<br/>train model during build<br/>non-root runtime user]
    end
```

## Local Development

Install dependencies:

```bash
uv sync
```

Runtime dependencies in `pyproject.toml` are FastAPI, joblib, MLflow, Pydantic,
scikit-learn, and uvicorn. Dev dependencies are httpx2, pytest, and ruff.

Train and save the serving artifact:

```bash
uv run python train.py
```

That command:

- trains a `RandomForestClassifier` on Iris,
- logs params, metrics, and the model to MLflow,
- uses the local MLflow store at `sqlite:///mlflow.db`,
- registers the logged model as `iris-classifier`,
- prints the created model registry version,
- writes `artifacts/model.pkl` for serving.

Validate the saved model:

```bash
uv run python scripts/validate_model.py
```

The validator loads `artifacts/model.pkl`, evaluates a deterministic held-out
Iris split, and exits `0` only if accuracy is at least `MIN_ACCURACY = 0.90`.
It exits `1` if the artifact is missing or the measured accuracy is below the
threshold.

Run the API locally:

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8080
```

Smoke check:

```bash
curl -i http://127.0.0.1:8080/health
curl -i http://127.0.0.1:8080/ready
curl -i http://127.0.0.1:8080/metrics
```

Run tests:

```bash
uv run pytest -q
```

## Docker

Build the image:

```bash
docker build -t ml-model-server:local .
```

Run it:

```bash
docker run --rm -p 8080:8080 ml-model-server:local
```

Check it from the host:

```bash
curl -i http://127.0.0.1:8080/health
```

The Dockerfile is multi-stage and both stages use `python:3.14-slim`. The
`builder` stage copies `uv` from `ghcr.io/astral-sh/uv:latest` and creates a
production virtualenv with `uv sync --frozen --no-dev --no-install-project`.
The `runtime` stage copies the virtualenv and app code, copies `train.py`, runs
`.venv/bin/python train.py`, and therefore trains/bakes `artifacts/model.pkl`
during the image build instead of copying the local artifact. It then creates
and runs as the non-root `appuser`, exposes `8080`, and starts uvicorn.

## CI/CD

Current workflow files:

- `.github/workflows/ci.yml` runs on pull requests and pushes to `main`. It
  checks out the repo, installs `uv`, installs Python 3.12, runs
  `uv sync --all-extras --dev`, runs `ruff check .`, runs `pytest -q`, and has a
  stub dependency-scan step.
- `.github/workflows/train.yml` runs on manual dispatch and on a weekly Monday
  cron. It installs `uv`, installs Python 3.12, installs dependencies, runs
  `train.py`, runs `scripts/validate_model.py`, and ends with a stub
  registration step.
- `.github/workflows/cd.yml` is an AWS CD scaffold validated against LocalStack,
  not a live production AWS deployment. It configures AWS credentials through
  GitHub OIDC, logs in to ECR, builds and pushes an image tagged with the commit
  SHA, has a stub image-scan step, then calls staging and production
  deploy/smoke-test scripts. The repository includes LocalStack and IAM starter
  files for validating that AWS-shaped path locally.

## Design Choices

- The model loads once in FastAPI lifespan startup, not per request. This keeps
  joblib deserialization out of the prediction hot path.
- `app/model.py` owns loading and prediction while `app/main.py` owns HTTP. This
  keeps artifact mechanics separate from request/response handling.
- Pydantic validates at the edge. Bad input returns `422` before the model sees
  it.
- `/health` and `/ready` are separate. Liveness says the process responds;
  readiness says the model is loaded and the app can serve traffic.
- The validation gate is deterministic and runs on the real serving artifact.
  That makes the deploy decision about `artifacts/model.pkl`, not an unrelated
  training object.
- The container runs as a non-root user. The API does not need root privileges
  to serve HTTP on port `8080`.

## Out of Scope

Kubernetes/EKS, Terraform, and Kubeflow are deliberate omissions, not hidden
requirements. They are reasonable choices to discuss for larger systems, but
this repository keeps the focus on a clean serving path around one model.
