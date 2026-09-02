import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.data_preprocess.process_text import extract_json, dict_to_rows
from src.xgbmodel.models_predict import load_model, predict_with_shap
from src.manual_llm_agent.huggingface_connect import (
    instruct_and_extract_schema,
    format_for_llm,
    instruct_and_interpret_report,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model-api")

MODEL_PATH = os.environ.get("MODEL_PATH", "PATH/TO/YOUR/XGBOOST/MODEL.json")
TOP_FEATURES = 5

ml_artifacts = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model from %s", MODEL_PATH)
    try:
        ml_artifacts["model"] = load_model(MODEL_PATH)
        logger.info("Model loaded.")
    except Exception:
        ml_artifacts["model"] = None
        logger.exception("Could not load model from %s -- set MODEL_PATH", MODEL_PATH)
    yield
    ml_artifacts.clear()


app = FastAPI(title="Loan Risk Model API", version="1.0.0", lifespan=lifespan)


class TextRequest(BaseModel):
    text: str


class ExplanationResponse(BaseModel):
    probability: float
    explanation: str


@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": ml_artifacts.get("model") is not None}


@app.post("/predict", response_model=ExplanationResponse)
def predict(request: TextRequest):
    try:
        # 1. Free text -> structured applicant record (LLM extraction)
        applicant = instruct_and_extract_schema(text_from_function=request.text)

        # 2. Structured record -> typed feature row matching training dtypes
        record, feature_dtypes, feature_order = extract_json(applicant.model_dump_json())
        inference_row = dict_to_rows(record, feature_order, feature_dtypes)

        # 3. XGBoost prediction + SHAP feature attribution
        (probability, drivers), feature_impacts = predict_with_shap(
            inference_df=inference_row,
            model_path=MODEL_PATH,
            top_features=TOP_FEATURES,
        )

        # 4. Prediction + drivers -> plain-language explanation (LLM interpretation)
        report_lines = format_for_llm(probability, feature_impacts)
        explanation = instruct_and_interpret_report(report_lines)

        return ExplanationResponse(probability=probability, explanation=explanation)

    except Exception as e:
        logger.exception("Prediction pipeline failed")
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")
