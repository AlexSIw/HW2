import pytest
import httpx
import logging
from unittest.mock import patch
from app.main import app

def test_health_endpoint():
    # Because we're not running the asynchronous test client heavily here
    # A simple way to test fastapi synchronously is TestClient
    from fastapi.testclient import TestClient
    client = TestClient(app)
    
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model" in data

@pytest.mark.asyncio
async def test_predict_endpoint_empty():
    from fastapi.testclient import TestClient
    client = TestClient(app)
    
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 400

@pytest.mark.asyncio
async def test_predict_endpoint_valid():
    from fastapi.testclient import TestClient
    client = TestClient(app)
    
    # Needs the model to be loaded; if not loaded, it might take a while on first run
    # For a real CI pipeline, we'd mock the classifier_instance.predict
    with patch("app.models.classifier.BiasClassifier.predict") as mock_predict:
        mock_predict.return_value = {
            "labels": ["homophobic", "racist"],
            "scores": [0.8, 0.1]
        }
        
        response = client.post("/predict", json={"text": "This is a mock test."})
        assert response.status_code == 200
        data = response.json()
        assert data["is_biased"] == True
        assert data["dominant_narrative"] == "homophobic"
        assert len(data["labels"]) == 2
