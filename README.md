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

🏠 **Source Code:** [GitHub Repository](https://github.com/mynameistatibond/claim-fraud-detection)

## 🚀 Quick Start

### Local Development

1. **Verify models exist:**
   Ensure `best_tree_models_calibrated.joblib` and `best_tree_models_uncalibrated.joblib` are in the `models/` directory.
   (They should have been moved there during setup)

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the server:**
   ```bash
   uvicorn app:app --reload --port 8000
   ```

4. **Open the UI:**
   Visit `http://localhost:8000` in your browser

### Example API Request

```bash
curl -X POST "http://localhost:8000/predict?model=xgb&scenario=dashboard" \
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
    "incident_hour_of_the_day": 14,
    "incident_severity": "Major Damage",
    "collision_type": "Rear Collision",
    "police_report_available": "YES",
    "authorities_contacted": "Police",
    "number_of_vehicles_involved": 2,
    "bodily_injuries": 1
  }'
```

The API now accepts both numeric and categorical features, automatically handling preprocessing and missing values.

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

#### `POST /submit_batch`

Async endpoint for processing CSV files with Agentic Triage.

**Query Parameters:**
- `team_size` (int): Number of available investigators (Default: 5)
- `review_time` (int): Minutes per review (Default: 20)
- `operating_mode` (str): `daily_ops`, `incident_sweep`, or `regulatory_audit`

**Request Body:**
- `file`: CSV file upload (multipart/form-data)

**Example:**
```bash
curl -X POST "http://localhost:8000/submit_batch?team_size=5&operating_mode=daily_ops" \
  -F "file=@/path/to/claims.csv"
```

## 🎯 Deployment Scenarios

### Scenario A: Auto-Flagger
**Use Case:** Automated claim flagging system

- Uses **uncalibrated** models for maximum recall
- Returns decision flag: `AUTO_FLAG` or `AUTO_APPROVE`
- Threshold: 0.53 (adjust based on your F2 optimization)

```bash
curl -X POST "http://localhost:8000/predict?model=xgb&scenario=auto_flagger" \
  -H "Content-Type: application/json" \
  -d @claim_data.json
```

### Scenario B: Investigator Dashboard
**Use Case:** Human-in-the-loop prioritization

- Uses **calibrated** models for accurate probabilities
- Returns probability score for ranking claims
- No hard threshold decision

```bash
curl -X POST "http://localhost:8000/predict?model=xgb&scenario=dashboard" \
  -H "Content-Type: application/json" \
  -d @claim_data.json
```

## ✨ Agentic System

This project goes beyond simple prediction by employing a multi-agent system to manage Fraud Operations:

### 1. Risk Appetite Agent (`risk_appetite_agent.py`)
- **Role**: Operational Commander
- **Input**: Team size, backlog size, batch statistics (P95 scores), and operating mode.
- **Output**: Sets the daily **Risk Appetite** (Conservative, Balanced, Aggressive).
- **Logic**: If the team is overloaded or the batch looks anomalous, it tightens thresholds to protect the team from alert fatigue.

### 2. Triage Agent (`triage_agent.py`)
- **Role**: Resource Allocator
- **Input**: Risk Appetite + Model Probabilities.
- **Output**: Assigns each claim to a priority queue (`P0` Must Review, `P1` Should Review, `P2` Maybe, `P3` Ignore).
- **Goal**: Maximizes fraud capture within the calculated `Daily Capacity`.

### 3. Explanation Agent (`explanation_agent.py`)
- **Role**: Communicator
- **Input**: SHAP values (feature contributions) + Context.
- **Output**: Generates a human-readable narrative explaining *why* a claim was flagged, specifically tailored to the investigator's time constraints.

---

## 🐳 Docker Deployment

### Build and Run Locally

```bash
docker build -t fraud-api .
docker run -p 8000:8000 fraud-api
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
fraud-detector/
├── app.py                  # FastAPI Entry Point
├── batch_ingest.py         # Async Batch Processor
├── mappings.py             # Feature Name Definitions (Shared)
├── preprocessing.py        # sklearn/pandas pipelines
├── llm_explainer.py        # LLM Integration Layer
├── data/                   # Local storage for CSVs
│
├── agents/                 # (Logical Grouping)
│   ├── triage_agent.py          # Priority Queue Logic
│   ├── risk_appetite_agent.py   # Operational Posture Logic
│   └── explanation_agent.py     # Narrative Generation
│
├── models/                 # Model Artifacts (.joblib, .npy)
├── scripts/                # Maintenance Utilities
├── Dockerfile              # Container Config
├── requirements.txt        # Dependencies
└── README.md               # Documentation
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
}
}
```

### LLM Configuration

The application uses an LLM to generate plain-language explanations of the risk scores. It supports multiple providers with the following priority:

1. **Groq** (Recommended - Fast & Free tier coverage)
   - Env Var: `GROQ_API_KEY`
   - Model: `llama-3.3-70b-versatile`

2. **Together AI**
   - Env Var: `TOGETHER_API_KEY`
   - Model: `mistralai/Mistral-7B-Instruct-v0.1`

3. **OpenRouter**
   - Env Var: `OPENROUTER_API_KEY`
   - Model: `mistralai/mistral-7b-instruct:free`

4. **HuggingFace** (Fallback)
   - Env Var: `HF_TOKEN`
   - Model: `HuggingFaceH4/zephyr-7b-beta`

Create a `.env` file in the root directory to set these keys:
```bash
GROQ_API_KEY=gsk_...
```

## 🛠️ Testing

Test the API with sample data:

```bash
# High-risk claim
curl -X POST "http://localhost:8000/predict?model=xgb&scenario=auto_flagger" \
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

## � Troubleshooting

### **Common Issues**

1. **Python Version Compatibility**
   - This project requires **Python 3.10+**.
   - If you see `TypeError: unsupported operand type(s) for |`, you are likely running Python 3.9 or older. Please upgrade.

2. **Port 8000 Already in Use**
   - If `uvicorn` fails to start, check if another process is using port 8000:
   ```bash
   lsof -i :8000
   kill -9 <PID>
   ```

## 🧹 Maintenance

The `scripts/` directory contains utility tools:

- **`verify_models.py`**: Checks if model files are present and loadable.
- **`test_api.sh`**: Runs a suite of curl commands to verify endpoints.
- **`build_trend_registry.py`**: Rebuilds the trend registry from background data.

```bash
# Example: Verify models
python scripts/verify_models.py
```

## �📝 License

MIT License - See project root for details
