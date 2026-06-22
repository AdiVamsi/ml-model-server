"""FastAPI entrypoint for serving the Iris classifier."""

from __future__ import annotations

import time
from collections import Counter
from contextlib import asynccontextmanager
from threading import Lock

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field

from app import model


class IrisFeatures(BaseModel):
    """Four numeric measurements used by the classic Iris classifier."""

    sepal_length: float = Field(..., description="Sepal length in centimeters")
    sepal_width: float = Field(..., description="Sepal width in centimeters")
    petal_length: float = Field(..., description="Petal length in centimeters")
    petal_width: float = Field(..., description="Petal width in centimeters")


class PredictionResponse(BaseModel):
    """Structured prediction response returned by POST /predict."""

    predicted_class: str
    probabilities: dict[str, float]


_metrics_lock = Lock()
_request_counts: Counter[tuple[str, str, int]] = Counter()
_predict_latency_seconds_total = 0.0
_predict_latency_seconds_count = 0


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load the model once during application startup."""
    model.load_model()
    yield


app = FastAPI(title="Iris Model Server", lifespan=lifespan)


@app.middleware("http")
async def count_requests(request: Request, call_next):
    """Count completed HTTP requests with method, path, and status labels."""
    response = await call_next(request)
    route = request.scope.get("route")
    path = getattr(route, "path", request.url.path)

    with _metrics_lock:
        _request_counts[(request.method, path, response.status_code)] += 1

    return response


@app.get("/health")
def health() -> dict[str, str]:
    """Return a basic liveness response regardless of model readiness."""
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, str]:
    """Return success only after the model has been loaded."""
    if not model.is_loaded():
        raise HTTPException(status_code=503, detail="model not loaded")

    return {"status": "ready"}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: IrisFeatures) -> PredictionResponse:
    """Return the predicted class and probabilities for one Iris sample."""
    global _predict_latency_seconds_count, _predict_latency_seconds_total

    features = [
        payload.sepal_length,
        payload.sepal_width,
        payload.petal_length,
        payload.petal_width,
    ]

    start = time.perf_counter()
    prediction = model.predict(features)
    elapsed = time.perf_counter() - start

    with _metrics_lock:
        _predict_latency_seconds_total += elapsed
        _predict_latency_seconds_count += 1

    return PredictionResponse.model_validate(prediction)


@app.get("/metrics")
def metrics() -> Response:
    """Return a small Prometheus text exposition for local scraping."""
    lines = [
        "# HELP app_requests_total Total HTTP requests served.",
        "# TYPE app_requests_total counter",
    ]

    with _metrics_lock:
        request_counts = _request_counts.copy()
        predict_latency_total = _predict_latency_seconds_total
        predict_latency_count = _predict_latency_seconds_count

    for (method, path, status_code), count in sorted(request_counts.items()):
        lines.append(
            'app_requests_total{'
            f'method="{method}",path="{path}",status_code="{status_code}"'
            f"}} {count}"
        )

    lines.extend(
        [
            "# HELP app_predict_latency_seconds_total Total predict latency.",
            "# TYPE app_predict_latency_seconds_total counter",
            f"app_predict_latency_seconds_total {predict_latency_total:.9f}",
            "# HELP app_predict_latency_seconds_count Predict request count.",
            "# TYPE app_predict_latency_seconds_count counter",
            f"app_predict_latency_seconds_count {predict_latency_count}",
            "",
        ]
    )

    return Response(content="\n".join(lines), media_type="text/plain")
