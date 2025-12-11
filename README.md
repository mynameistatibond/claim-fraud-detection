---
title: Fraud Detector
emoji: 🕵️‍♀️
colorFrom: blue
colorTo: purple
sdk: docker
sdk_version: "1.0"
app_file: app.py
pinned: false
---


# Fraud Detection API

Production-ready inference API for insurance fraud detection using pre-trained ML models.

## 🚀 Quick Start

### Local Development

1. **Copy your trained models:**
   ```bash
   cp ../models/best_tree_models_calibrated.joblib models/
   cp ../models/best_tree_models_uncalibrated.joblib models/
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the server:**
   ```bash
   uvicorn app:app --reload --port 7860
   ```

4. **Open the UI:**
   Visit `http://localhost:7860` in your browser

### Example API Request

```bash
curl -X POST "http://localhost:7860/predict?model=xgb&scenario=dashboard" \
  -H "Content-Type: application/json" \
  -d '{
    "policy_annual_premium": 1200.0,
    "total_claim_amount": 15000.0,
    "vehicle_age": 5,
    "days_since_bind": 300,
    "months_as_customer": 24,
    "capital-gains": 0,
    "capital-loss": 0,
    "injury_share": 0.4,
    "property_share": 0.6,
    "umbrella_limit": 0,
    "incident_hour_of_the_day": 14
  }'
```

### Example Response

```json
{
  "model": "XGBoost",
  "calibrated": true,
  "probability": 0.73,
  "threshold_flag": null,
  "scenario": "dashboard"
}
```

## 📋 API Reference

### Endpoints

#### `POST /predict`

Make a fraud prediction for an insurance claim.

**Query Parameters:**
- `model` (string): Model type - `rf` (RandomForest), `et` (ExtraTrees), or `xgb` (XGBoost)
- `scenario` (string): `dashboard` (calibrated) or `auto_flagger` (uncalibrated + threshold)
- `calibrated` (boolean): Override calibration (optional, scenario takes precedence)

**Request Body:**
```json
{
  "policy_annual_premium": float,
  "total_claim_amount": float,
  "vehicle_age": int,
  "days_since_bind": int,
  "months_as_customer": int,
  "capital-gains": float,
  "capital-loss": float,
  "injury_share": float,
  "property_share": float,
  "umbrella_limit": int,
  "incident_hour_of_the_day": int (0-23)
}
```

#### `GET /health`

Health check endpoint returning model status.

## 🎯 Deployment Scenarios

### Scenario A: Auto-Flagger
**Use Case:** Automated claim flagging system

- Uses **uncalibrated** models for maximum recall
- Returns decision flag: `AUTO_FLAG` or `AUTO_APPROVE`
- Threshold: 0.53 (adjust based on your F2 optimization)

```bash
curl -X POST "http://localhost:7860/predict?model=xgb&scenario=auto_flagger" \
  -H "Content-Type: application/json" \
  -d @claim_data.json
```

### Scenario B: Investigator Dashboard
**Use Case:** Human-in-the-loop prioritization

- Uses **calibrated** models for accurate probabilities
- Returns probability score for ranking claims
- No hard threshold decision

```bash
curl -X POST "http://localhost:7860/predict?model=xgb&scenario=dashboard" \
  -H "Content-Type: application/json" \
  -d @claim_data.json
```

## 🐳 Docker Deployment

### Build and Run Locally

```bash
docker build -t fraud-api .
docker run -p 7860:7860 fraud-api
```

### Deploy to HuggingFace Spaces

1. Create a new Space on HuggingFace
2. Select **Docker** as SDK
3. Push this folder to your Space repository:

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
git push -u origin main
```

4. HuggingFace will automatically build and deploy your Docker container
5. Your API will be available at: `https://YOUR_USERNAME-YOUR_SPACE_NAME.hf.space`

## 📁 Project Structure

```
fraud_api/
├── app.py              # FastAPI backend
├── index.html          # Web UI
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container configuration
├── README.md           # This file
└── models/             # Model files (add your .joblib files here)
    ├── best_tree_models_calibrated.joblib
    └── best_tree_models_uncalibrated.joblib
```

## ⚙️ Configuration

### Adjust Auto-Flag Threshold

Edit `app.py` line 19:
```python
THRESHOLD_AUTO_FLAG = 0.53  # Adjust based on your requirements
```

### Model Loading

Models are loaded on startup from `models/` directory. Expected format:
```python
{
  'Trees': {
    'RandomForest': <model_pipeline>,
    'ExtraTrees': <model_pipeline>,
    'XGBoost': <model_pipeline>
  }
}
```

## 🛠️ Testing

Test the API with sample data:

```bash
# High-risk claim
curl -X POST "http://localhost:7860/predict?model=xgb&scenario=auto_flagger" \
  -H "Content-Type: application/json" \
  -d '{
    "policy_annual_premium": 500,
    "total_claim_amount": 50000,
    "vehicle_age": 1,
    "days_since_bind": 10,
    "months_as_customer": 2,
    "capital-gains": 10000,
    "capital-loss": 0,
    "injury_share": 0.8,
    "property_share": 0.2,
    "umbrella_limit": 0,
    "incident_hour_of_the_day": 3
  }'
```

## 📊 Model Information

This API serves predictions from models trained on insurance claim data with F2-score optimization for fraud detection. The models were calibrated using Platt scaling to ensure probability quality.

**Available Models:**
- **RandomForest**: Ensemble of decision trees
- **ExtraTrees**: Extra randomized trees
- **XGBoost**: Gradient boosted decision trees

**Calibration:**
- Uncalibrated: Optimized for maximum recall (catching fraud)
- Calibrated: Optimized for probability accuracy (ranking)

## 🔒 Security Notes

- This is a minimal inference API for demonstration
- For production deployment, add:
  - Authentication (API keys, OAuth)
  - Rate limiting
  - Input sanitization
  - HTTPS/TLS
  - Monitoring and logging
  - Model versioning

## 📝 License

MIT License - See project root for details
