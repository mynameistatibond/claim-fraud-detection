# Fraud Detector App Specification

## 1. System Overview
The **Fraud Detector** (`fraud-detector/`) is a lightweight, containerized API built with **FastAPI**. It serves predictions from the project's trained machine learning models (XGBoost, RandomForest, ExtraTrees) and includes a standalone preprocessing pipeline to ensure production data matches training data.

## 2. API Architecture

### Endpoints
*   `GET /`: Returns the simple `index.html` UI.
*   `GET /health`: Returns system status and count of loaded models.
*   `POST /predict`: The main inference endpoint.

### Model Loading Strategy
The app scans its local `models/` directory on startup for:
*   `best_tree_models_calibrated.joblib`
*   `best_tree_models_uncalibrated.joblib`

It loads specific models ("RandomForest", "ExtraTrees", "XGBoost", "VotingEnsemble") from these dictionaries into memory.
> **Note**: Currently, the `fraud-detector/models/` directory is empty. You must copy the .joblib files from the root `models/` directory into `fraud-detector/models/` for the app to function locally.

## 3. Input Data Specification (`/predict`)

The API accepts a JSON payload matching the `ClaimInput` schema. Critical fields include:

| Field | Type | Description |
| :--- | :--- | :--- |
| `policy_annual_premium` | Float | Annual cost of policy |
| `total_claim_amount` | Float | Total value of the claim |
| `vehicle_age` | Int | Age of car in years |
| `days_since_bind` | Int | Policy tenure in days |
| `age` | Int | Insured Age (Years) |
| `incident_severity` | String | e.g., "Major Damage", "Total Loss" |
| `incident_hour_of_the_day` | Int | 0-23 |
| `collision_type` | String | "Front Collision", "Side Collision", etc. |
| `authorities_contacted` | String | "Police", "None", etc. |

**Parameters:**
*   `model`: `rf` (RandomForest), `et` (ExtraTrees), `xgb` (XGBoost), or `voting` (VotingEnsemble).
*   `calibrated`: `true` (standard) or `false`.
*   `scenario`:
    *   `dashboard`: Uses **Calibrated** probabilities (Good for human review).
    *   `auto_flagger`: Uses **Uncalibrated** models (Good for high recall automation).

## 4. Preprocessing Logic
The app uses `preprocessing.py` to transform the raw JSON input into the exact 42-column format required by the models.

1.  **Defaults**: It starts with a default dictionary (loaded from `preprocessing_metadata.json`) containing median/mode values for all 42 columns (e.g., `age=38`, `policy_state="OH"`).
2.  **Merge**: User input overrides these defaults.
3.  **Feature Engineering**:
    *   **Time**: `incident_hour_of_the_day` is converted into `hour_sin` and `hour_cos` (Cyclical encoding) and `hour_bin_4` (Morning/Evening bins).
    *   **Missing Flags**: If fields like `collision_type` are missing or "?", a specific flag column (e.g., `collision_type_missing`) is set to 1.
4.  **Output**: A Pandas DataFrame with strictly ordered columns matching the training set.

## 5. Response Format
The API returns a JSON object:
```json
{
  "model": "RandomForest",
  "calibrated": true,
  "probability": 0.76,
  "threshold_flag": "AUTO_FLAG",  // Only if scenario="auto_flagger"
  "scenario": "auto_flagger"
}
```
*   **Threshold**: If `scenario="auto_flagger"`, it applies a threshold of **0.53**. If probability >= 0.53, it returns `AUTO_FLAG`.

## 6. How to Run
1.  **Copy Models**:
    ```bash
    cp models/best_tree_models_*.joblib fraud-detector/models/
    ```
2.  **Build & Run Docker**:
    ```bash
    cd fraud-detector
    docker build -t fraud-api .
    docker run -p 5000:7860 fraud-api
    ```
3.  **Test**: Open `http://localhost:5000` in your browser.
