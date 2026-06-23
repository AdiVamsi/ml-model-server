"""AWS Lambda handler: loads the model from S3, returns an iris prediction.

This mirrors the FastAPI /predict endpoint but in Lambda's handler shape.
The model is pulled from S3 at cold start (not baked in) — the decoupled
artifact pattern.
"""
import json
import os
import pickle

import boto3

# Loaded once per warm container (cold start only), reused across invokes.
_model = None


def _load_model():
    global _model
    if _model is None:
        bucket = os.environ["MODEL_BUCKET"]
        key = os.environ["MODEL_KEY"]
        s3 = boto3.client("s3", endpoint_url=os.environ.get("S3_ENDPOINT"))
        obj = s3.get_object(Bucket=bucket, Key=key)
        _model = pickle.loads(obj["Body"].read())
    return _model


def handler(event, context):
    model = _load_model()

    # event carries the four features; default to a setosa-like sample.
    body = event if isinstance(event, dict) else json.loads(event)
    features = [
        body.get("sepal_length", 0.0),
        body.get("sepal_width", 0.0),
        body.get("petal_length", 0.0),
        body.get("petal_width", 0.0),
    ]

    pred = model.predict([features])[0]
    return {"predicted_class": str(pred)}
