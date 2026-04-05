from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import os

from app.api.routes import router as api_router
from app.core.config import get_settings
from app.core.logging import logger
from app.models.classifier import classifier_instance

settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup resources (e.g. load model into memory on startup)
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    try:
        classifier_instance.load_model()
        logger.info("ML Model pre-loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load ML model on startup: {e}")
        # Not terminating here in case we want to retry or load on demand
        
    yield
    
    # Cleanup resources
    logger.info("Shutting down API...")

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="MLOps Narrative Bias Detection API using Zero-Shot Classification",
    lifespan=lifespan
)

# Set up CORS logic to allow calling from any web frontend for flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)

# Mount static files folder
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_ui():
    with open(os.path.join(STATIC_DIR, "index.html"), "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.API_HOST, port=settings.API_PORT, reload=True)
