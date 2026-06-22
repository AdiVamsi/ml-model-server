from fastapi.testclient import TestClient

from app.main import app


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")

    # TODO: Fill in assertions.


def test_predict_valid_payload():
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

    # TODO: Fill in assertions.


def test_predict_malformed_payload():
    payload = {
        "sepal_length": "wide",
        "sepal_width": 3.5,
        "petal_length": 1.4,
    }

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

    # TODO: Fill in assertions. Expected status code: 422.
