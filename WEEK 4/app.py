
import joblib
import pandas as pd
import math

from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from pydantic import BaseModel, Field, ConfigDict, model_validator


# Project directory
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model.pkl"

templates = Jinja2Templates(
    directory=str(BASE_DIR / "templates")
)


# Load model during startup
@asynccontextmanager
async def lifespan(app: FastAPI):

    if not MODEL_PATH.exists():
        raise RuntimeError(
            "Model not found. Run python train.py first."
        )

    app.state.artifact = joblib.load(MODEL_PATH)

    yield

    app.state.artifact = None


# Create FastAPI application
app = FastAPI(
    title="Breast Cancer Prediction API",
    description="Week 4 Machine Learning Capstone",
    version="1.0.0",
    lifespan=lifespan
)


# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"]
)


# Pydantic Input Validation
class PredictionRequest(BaseModel):

    model_config = ConfigDict(extra="forbid")

    features: dict[str, float] = Field(
        ...,
        description="30 numerical model features"
    )

    @model_validator(mode="after")
    def validate_features(self):

        for name, value in self.features.items():

            if not math.isfinite(value):
                raise ValueError(
                    f"{name} must be a finite number"
                )

        return self


# Home Page
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


# Health Check
@app.get("/health")
async def health():

    return {
        "status": "Running",
        "model_loaded": app.state.artifact is not None
    }


# Get Features
@app.get("/features")
async def get_features():

    artifact = app.state.artifact

    return {
        "feature_names": artifact["feature_names"],
        "defaults": artifact["defaults"],
        "target_names": artifact["target_names"]
    }


# Prediction Endpoint
@app.post("/predict")
async def predict(data: PredictionRequest):

    artifact = app.state.artifact

    required_features = artifact["feature_names"]

    supplied = set(data.features.keys())
    expected = set(required_features)

    missing = sorted(expected - supplied)
    unexpected = sorted(supplied - expected)

    if missing or unexpected:

        raise HTTPException(
            status_code=422,
            detail={
                "message": "Invalid feature names",
                "missing_features": missing,
                "unexpected_features": unexpected
            }
        )

    try:

        # Prepare input
        input_data = pd.DataFrame(
            [[data.features[name] for name in required_features]],
            columns=required_features
        )

        model = artifact["pipeline"]

        # Generate prediction
        prediction = int(model.predict(input_data)[0])

        # Generate probability
        probabilities = model.predict_proba(input_data)[0]

        confidence = float(probabilities[prediction])

        result = artifact["target_names"][prediction]

        return {
            "prediction": result,
            "confidence": round(confidence, 4),
            "confidence_percent": round(
                confidence * 100, 2
            ),
            "class_probabilities": {
                artifact["target_names"][i]: round(
                    float(probabilities[i]), 4
                )
                for i in range(len(probabilities))
            },
            "message": "Educational demonstration only. Not a medical diagnosis."
        }

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
