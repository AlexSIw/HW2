from fastapi import APIRouter
from app.models.schemas import HealthResponse
from app.models.classifier import classifier_instance
from app.core.config import get_settings

router = APIRouter()
settings = get_settings()

@router.get("/health", response_model=HealthResponse, tags=["Diagnostics"])
async def health_check():
    """
    Check API health and model loader status.
    """
    return HealthResponse(
        status="ok",
        model=settings.MODEL_NAME,
        model_loaded=classifier_instance.is_loaded()
    )
