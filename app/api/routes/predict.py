from fastapi import APIRouter, HTTPException
from app.models.schemas import PredictRequest, PredictResponse, PredictBatchRequest, PredictBatchResponse
from app.services.bias_service import analyze_text, analyze_batch

router = APIRouter()

@router.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict_bias(request: PredictRequest):
    """
    Detect narrative bias in a single piece of text.
    Returns likelihood scores for 8 bias categories.
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
        
    return analyze_text(request.text, request.threshold)

@router.post("/predict/batch", response_model=PredictBatchResponse, tags=["Inference"])
async def predict_bias_batch(request: PredictBatchRequest):
    """
    Detect narrative bias in multiple pieces of text simultaneously.
    """
    if not request.texts or not all(t.strip() for t in request.texts):
        raise HTTPException(status_code=400, detail="Texts list cannot be empty and items cannot be blank.")
        
    return analyze_batch(request.texts, request.threshold)
