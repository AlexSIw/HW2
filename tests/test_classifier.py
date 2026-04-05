from app.models.classifier import BiasClassifier
from app.core.config import get_settings

def test_classifier_initialization():
    classifier = BiasClassifier()
    assert len(classifier.LABELS) == 8
    assert "homophobic" in classifier.LABELS
    assert "feminist" in classifier.LABELS

def test_classifier_mocked_prediction(mocker):
    classifier = BiasClassifier()
    # Mocking the pipeline object
    mock_pipeline = mocker.MagicMock()
    mock_pipeline.return_value = {
        "sequence": "I hate you.",
        "labels": ["racist", "ageist"],
        "scores": [0.9, 0.05]
    }
    
    classifier.model_pipeline = mock_pipeline
    
    result = classifier.predict("I hate you.")
    
    assert "labels" in result
    assert "scores" in result
    assert result["labels"][0] == "racist"
    assert result["scores"][0] == 0.9
