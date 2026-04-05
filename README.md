# MLOps Narrative Bias Detection API

A production-ready FastAPI endpoint for zero-shot text classification, specifically detecting biased narratives and sways using the `facebook/bart-large-mnli` model.

## Features
- **Zero-Shot Classification**: Detects 8 predefined bias categories without requiring any labeled training data.
- **FastAPI**: Fully asynchronous web server with automatic Swagger UI (`/docs`).
- **MLOps Structure**: Organized code base separating web routes, machine learning code, business logic, and testing.
- **Batch Processing**: Supports analyzing multiple text strings simultaneously.

## Bias Categories Detected
1. `homophobic`
2. `feminist`
3. `racist`
4. `xenophobic`
5. `islamophobic`
6. `ageist`
7. `sizeist`
8. `ableist`

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
# On Windows
.\\venv\\Scripts\\activate
# On Mac/Linux
source venv/bin/activate
```

2. Install the requirements:
```bash
pip install -r requirements.txt
```

3. Setup environment variables:
```bash
cp .env.example .env
```
*(Optionally tweak the settings inside `.env`)*

## Running the Server

Start the API on port 8000:
```bash
uvicorn app.main:app --reload
```
*(Or use `sh run.sh`)*

The API docs will be available at [http://localhost:8000/docs](http://localhost:8000/docs).

## Usage Examples

### 1. Single Text Prediction

```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "text": "Those immigrants are taking all of our jobs.",
  "threshold": 0.3
}'
```

### 2. Batch Text Prediction

```bash
curl -X 'POST' \
  'http://localhost:8000/predict/batch' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "texts": [
    "Those immigrants are taking all of our jobs.",
    "I love a sunny day."
  ],
  "threshold": 0.3
}'
```

## Running Tests

Integration and unit tests can be executed via pytest:
```bash
pytest tests/ -v
```
