import os
import time

import pandas as pd
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

from churn_project.api.schemas import BatchInput, PredictionResponse, UserInput
from churn_project.inference.prediction_service import PredictionService

# Load environment variables to run locally
load_dotenv()

app = FastAPI()

# Initialize Prometheus Instrumentator
instrumentator = Instrumentator().instrument(app)
instrumentator.expose(app)  # Expose metrics at /metrics endpoint

# Define custom metrics
PREDICTION_COUNT = Counter(
    "prediction_requests_total", "Total number of prediction requests"
)
PREDICTION_LATENCY = Histogram(
    "prediction_request_latency_seconds", "Latency of prediction requests in seconds"
)
OUTPUT_DISTRIBUTION = Counter(
    "prediction_output_distribution", "Distribution of prediction outputs", ["label"]
)

# Initialize PredictionService singleton
predictor = PredictionService(
    prod_s3_uri=os.getenv("PROD_S3_URI", "s3://churn-production/champion-model/")
)


# Dependency to ensure model is loaded
def model_available():
    if predictor.model is None:
        raise HTTPException(
            status_code=503, detail="Model is not loaded. Please try again later."
        )
    return True


# Root Endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Churn Prediction API"}


# Health Check Endpoint
@app.get("/health")
def health_check():
    return {
        "status": "OK",
        "model_version": predictor.model_version,
        "model_loaded": predictor.model is not None,
    }


# Prediction Endpoint
@app.post("/predict", response_model=PredictionResponse)
def predict(input_data: UserInput, _: bool = Depends(model_available)):
    start = time.time()

    # Convert input data to DataFrame and make prediction
    df = pd.DataFrame([input_data.model_dump()])
    output = predictor.predict(df)

    # Record metrics
    latency = time.time() - start
    PREDICTION_LATENCY.observe(latency)

    PREDICTION_COUNT.inc()

    for pred in output["predictions"]:
        OUTPUT_DISTRIBUTION.labels(label=str(pred)).inc()

    return PredictionResponse(**output)


# Prediction Endpoint for batch input
@app.post("/predict_batch", response_model=PredictionResponse)
def predict_batch(batch_input: BatchInput, _: bool = Depends(model_available)):
    start = time.time()

    # Convert input data to DataFrame and make prediction
    df = pd.DataFrame([item.model_dump() for item in batch_input.records])
    output = predictor.predict(df)

    # Record metrics
    latency = time.time() - start
    PREDICTION_LATENCY.observe(latency)

    PREDICTION_COUNT.inc(len(output["predictions"]))

    for pred in output["predictions"]:
        OUTPUT_DISTRIBUTION.labels(label=str(pred)).inc()

    return PredictionResponse(**output)
