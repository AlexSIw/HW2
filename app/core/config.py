from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    # App configuration
    APP_NAME: str = "Narrative Bias API"
    APP_VERSION: str = "0.1.0"
    
    # Model configuration
    MODEL_NAME: str = "facebook/bart-large-mnli"
    THRESHOLD: float = 0.3
    MAX_TEXT_LENGTH: int = 1024
    
    # API configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 1
    
    # Logging configuration
    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

@lru_cache()
def get_settings() -> Settings:
    return Settings()
