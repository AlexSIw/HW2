import time
from typing import List, Optional
from app.models.classifier import classifier_instance
from app.models.schemas import BiasLabel, PredictResponse, PredictBatchResponse
from app.core.config import get_settings

settings = get_settings()

def analyze_text(text: str, custom_threshold: Optional[float] = None) -> PredictResponse:
    start_time = time.time()
    
    threshold = custom_threshold if custom_threshold is not None else settings.THRESHOLD
    
    # Run the model
    raw_result = classifier_instance.predict(text)
    
    labels: List[BiasLabel] = []
    dominant_narrative = None
    is_biased = False
    
    # Process the result (HuggingFace returns labels and scores in descending order)
    for label, score in zip(raw_result["labels"], raw_result["scores"]):
        flagged = score >= threshold
        if flagged:
            is_biased = True
            if dominant_narrative is None:
                dominant_narrative = label  # The first flagged label is the highest scoring one
                
        labels.append(BiasLabel(label=label, score=score, flagged=flagged))
        
    processing_time_ms = int((time.time() - start_time) * 1000)
    
    return PredictResponse(
        text=text,
        labels=labels,
        dominant_narrative=dominant_narrative,
        is_biased=is_biased,
        processing_time_ms=processing_time_ms
    )

def analyze_batch(texts: List[str], custom_threshold: Optional[float] = None) -> PredictBatchResponse:
    start_time = time.time()
    results = [analyze_text(text, custom_threshold) for text in texts]
    total_time_ms = int((time.time() - start_time) * 1000)
    
    return PredictBatchResponse(
        results=results,
        total_processing_time_ms=total_time_ms
    )
