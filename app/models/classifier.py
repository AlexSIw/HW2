from transformers import pipeline
import torch
from app.core.config import get_settings
from app.core.logging import logger

class BiasClassifier:
    _instance = None
    
    # The 8 bias categories, phrased for the NLI model hypothesis:
    # "This text is {label}."
    LABELS = [
        "homophobic",
        "feminist",
        "racist",
        "xenophobic",
        "islamophobic",
        "ageist",
        "sizeist",
        "ableist"
    ]

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.model_pipeline = None
            cls._instance.settings = get_settings()
        return cls._instance

    def load_model(self):
        """Loads the model onto GPU if available, else CPU."""
        if self.model_pipeline is not None:
            return

        device = 0 if torch.cuda.is_available() else -1
        model_name = self.settings.MODEL_NAME

        logger.info(f"Loading zero-shot classification model: {model_name} on device: {'GPU' if device == 0 else 'CPU'}")
        
        # Load HuggingFace pipeline
        self.model_pipeline = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=device
        )
        logger.info("Model loaded successfully.")

    def predict(self, text: str) -> dict:
        """Runs zero-shot inference on text against the predefined labels."""
        if self.model_pipeline is None:
            self.load_model()
            
        # Truncate text if necessary to prevent out-of-memory
        text = text[:self.settings.MAX_TEXT_LENGTH]
            
        result = self.model_pipeline(text, candidate_labels=self.LABELS)
        return result
        
    def is_loaded(self) -> bool:
        return self.model_pipeline is not None

classifier_instance = BiasClassifier()
