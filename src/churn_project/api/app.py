import pandas as pd
from fastapi import Depends, FastAPI, HTTPException

from churn_project.api.schemas import BatchInput, PredictionResponse, UserInput
from churn_project.entity.config_entity import MlflowConfig
from churn_project.inference.prediction_service import PredictionService

app = FastAPI()

# Initialize PredictionService singleton
mlflow_config = MlflowConfig(
    tracking_uri="http://localhost:5000",
    prod_registry_name="prod.churn_model",
    registry_name="churn_model",
    experiment_name="churn_prediction",
)

predictor = PredictionService(mlflow_config)


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

    df = pd.DataFrame([input_data.model_dump()])

    output = predictor.predict(df)

    return PredictionResponse(**output)


# Prediction Endpoint for batch input
@app.post("/predict_batch", response_model=PredictionResponse)
def predict_batch(batch_input: BatchInput, _: bool = Depends(model_available)):

    df = pd.DataFrame([item.model_dump() for item in batch_input.records])

    output = predictor.predict(df)

    return PredictionResponse(**output)
