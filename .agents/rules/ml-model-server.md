# Workspace Rule — ml-model-server (set activation: ALWAYS ON)

> Save at `.agents/rules/ml-model-server.md` (or Rules panel -> + Workspace).
> Keep it Always On. Under the 12,000-char limit.

## Who you're working with
Aady - backend + AI engineer, Python, on macOS, uses `uv` (never bare pip/venv).
Experienced. Wants explainable code, not magic. Hard constraint: **everything must
be free - no credit card, no paid cloud, no surprise bills.**

## What this repo is
`ml-model-server` mirrors an insurance MLOps team in miniature: a trained model
becomes a deployed, automated, monitored API. It is the candidate's single most
important interview artifact for a "DevOps engineer, strong ML focus" role at an AWS
shop. The panel WILL read this code line by line.

The job is everything AFTER a model is trained - wrap it in an API, containerize,
automate CI/CD, deploy, monitor, retrain. NOT model research. NOT GenAI/LLMs. This
is classic ML infrastructure (MLOps).

## Golden rules (never violate)
1. **Explainability over cleverness.** Every non-trivial file gets a top docstring
   plus comments explaining *why*. Assume every line will be questioned by an
   interviewer. No clever one-liners.
2. **Free only.** Never introduce a service that requires a credit card or can bill.
   Allowed: GitHub (public repo), Hugging Face Spaces, LocalStack, local Docker,
   local Kubernetes (kind/minikube), SageMaker Studio Lab. If a step seems to need a
   paid service, STOP and ask.
3. **Spine first, no over-engineering.** Build the minimum that works, in the locked
   order below. The course-reinforcement track (phase 7) is OPTIONAL - never start it
   before the spine is solid.
4. **Stop at every phase boundary.** Run it, show real output, write a 3-bullet
   "explain this back" summary, then wait for the user.
5. **Flag interview-critical files.** When you touch `app/main.py`, `app/model.py`,
   `scripts/validate_model.py`, any `.github/workflows/*.yml`, or `Dockerfile`, end
   with: `REVIEW REQUIRED - read and be able to defend this before moving on.`
6. **Never commit secrets or model binaries.** `artifacts/*.pkl`, `mlruns/`, `.env`,
   `__pycache__/` are gitignored.
7. **Review-driven behavior.** Ask before terminal commands that hit the network,
   install large deps, or start clusters. Never run destructive commands.

## Stack
- Python 3.12 via `uv`. FastAPI + uvicorn[standard], Pydantic v2.
- scikit-learn + joblib. MLflow (local store) for tracking + registry.
- Docker (multi-stage, slim, non-root). The same Dockerfile serves the live demo
  AND the AWS Lambda container image.
- Quality: pytest, ruff. CI/CD: GitHub Actions (public repo = free unlimited).

### Deploy targets
- **Live demo (free, no card): Hugging Face Spaces, Docker SDK.** Push repo with the
  Dockerfile + the tiny model baked in; HF builds + serves a public HTTPS URL.
- **AWS competency (free): `cd.yml` targets ECR + Lambda (container) + Function URL,
  validated against LocalStack.** Auth via OIDC role assumption - NEVER static keys.
  No real paid AWS deploy is required.

### Course-reinforcement track (phase 7, optional, all free)
- DVC for data/model versioning (local or Google Drive remote).
- Local Kubernetes via `kind`/`minikube` + KServe to serve the model the course way.
- SageMaker Studio Lab (no card) to TRAIN the model - screenshots only.

### Speak-to only - DO NOT BUILD (paid or heavy)
EKS, real SageMaker endpoints, Kubeflow (unless local time permits), Terraform,
Snowflake/Databricks/Spark/Kafka. The user discusses these in interview; contrast
them with what was actually built.

## Build order (LOCKED)
0. `train.py` - RandomForest on iris, log to MLflow, joblib.dump to
   `artifacts/model.pkl`. **[DONE - do not rewrite]**
1. Register the model to the MLflow Model Registry (versioned).
2. `scripts/validate_model.py` - validation gate: accuracy >= MIN_ACCURACY (0.90),
   exit non-zero on fail. SENIOR SIGNAL - comment heavily.
3. `app/model.py` (load model once) + `app/main.py` (FastAPI: GET /health, GET
   /ready, POST /predict with Pydantic in/out, GET /metrics Prometheus). Load model
   at startup via lifespan, never per request.
4. `Dockerfile` (+ `.dockerignore`) - multi-stage, slim, non-root, expose 8080,
   uvicorn entrypoint, model baked in.
5. CI/CD:
   - `ci.yml` (pull_request): ruff + pytest + dependency/image scan (scan may stub).
   - `train.yml` (dispatch/schedule): env -> train -> validation gate -> register.
   - `cd.yml` (push main): OIDC -> build image -> scan -> push ECR -> deploy Lambda
     staging (LocalStack) -> smoke test -> manual approval (GitHub Environment) ->
     prod -> health check. Plus a job/step that deploys the live demo to HF Spaces.
6. Deploy: push to HF Spaces (live, free); run `cd.yml` against LocalStack for AWS
   evidence (logs/screenshots).
7. Course-reinforcement track (optional, in this order): DVC -> KServe on local kind
   -> Studio Lab training -> speak-to Kubeflow + Evidently drift + Grafana.

## BUILD vs STUB vs SKIP
- Build real: FastAPI serving, validation gate, all CI/CD logic, Dockerfile, ECR
  push + Lambda deploy (against LocalStack), HF Spaces deploy, pytest, ruff config.
- Stub + comment (`# [ENTERPRISE STUB]`): deep image scanning, Slack/email notify,
  multi-account promotion, remote prod MLflow server.
- Do NOT build: anything in "Speak-to only" above.

## File tree
```
ml-model-server/
|- app/{__init__.py, main.py, model.py}
|- train.py
|- scripts/{validate_model.py, deploy.sh, smoke_test.sh, notify.sh}
|- tests/unit/test_api.py
|- .github/workflows/{ci.yml, train.yml, cd.yml}
|- artifacts/                 # model.pkl (gitignored)
|- k8s/                       # KServe InferenceService yaml (phase 7)
|- dvc.yaml .dvc/             # phase 7
|- Dockerfile .dockerignore pyproject.toml .gitignore README.md
```

## Conventions
- Type hints everywhere; named constants, no magic numbers.
- Run via `uv run`. Lint: `uv run ruff check . && uv run ruff format .`.
- Tests: `uv run pytest -q`. FastAPI tests use `TestClient`.
- Conventional commits. Config via env vars; never hardcode account IDs/region/repo.