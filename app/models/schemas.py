from typing import List, Optional
from pydantic import BaseModel, Field

class BiasLabel(BaseModel):
    label: str = Field(..., description="The narrative bias category")
    score: float = Field(..., description="Confidence score between 0 and 1")
    flagged: bool = Field(..., description="True if score >= threshold")

class PredictRequest(BaseModel):
    text: str = Field(..., description="The text to analyze", max_length=5000)
    threshold: Optional[float] = Field(None, description="Custom threshold to override default (0.0 to 1.0)", ge=0.0, le=1.0)

class PredictBatchRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to analyze (max 20)", max_length=20)
    threshold: Optional[float] = Field(None, description="Custom threshold to override default (0.0 to 1.0)", ge=0.0, le=1.0)

class PredictResponse(BaseModel):
    text: str = Field(..., description="The analyzed text")
    labels: List[BiasLabel] = Field(..., description="Scores for each bias category")
    dominant_narrative: Optional[str] = Field(None, description="The label with the highest score, if flagged")
    is_biased: bool = Field(..., description="True if any label is flagged")
    processing_time_ms: int = Field(..., description="Inference time in milliseconds")

class PredictBatchResponse(BaseModel):
    results: List[PredictResponse]
    total_processing_time_ms: int = Field(..., description="Total inference time for the batch in milliseconds")

class HealthResponse(BaseModel):
    status: str = Field(..., description="API health status")
    model: str = Field(..., description="The loaded model name")
    model_loaded: bool = Field(..., description="True if model is loaded into memory")
