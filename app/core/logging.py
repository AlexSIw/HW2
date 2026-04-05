import logging
from rich.logging import RichHandler
from app.core.config import get_settings

def setup_logging():
    settings = get_settings()
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, markup=True)]
    )
    
    # Set loggers for noisy libraries to WARNING
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    
    return logging.getLogger("app")

logger = setup_logging()
