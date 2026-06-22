from fastapi.testclient import TestClient

from app.main import app


def test_health():
    """Liveness probe must return 200 with an ok status."""
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_valid_payload():
    """A well-formed request returns 200 and a predicted class in the body."""
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert "predicted_class" in body
    assert isinstance(body["predicted_class"], str)

def test_predict_malformed_payload():
    """A non-numeric feature must be rejected by Pydantic validation (422)."""
    payload = {
        "sepal_length": "wide",   # string where a float is required
        "sepal_width": 3.5,
        "petal_length": 1.4,
        # petal_width also missing — both reasons trigger 422
    }

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 422