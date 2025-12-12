# Insurance Claim Fraud Prediction

This project builds a machine learning pipeline to detect fraudulent insurance claims. It progresses from data analysis and feature engineering to training, calibrating, and evaluating tree-based models (XGBoost, Random Forest, etc.).

## 📂 Project Organization

```
├── README.md          <- Project documentation
├── requirements.txt   <- Python dependencies
│
├── data               <- Data source
│   ├── raw            <- Immutable original data (insurance_claims.csv)
│   └── processed      <- Cleaned and engineered datasets
│
├── models             <- Trained model artifacts (.joblib)
│   ├── best_tree_models_uncalibrated.joblib
│   └── [individual model files...]
│
├── notebooks          <- Analysis workflow (Run in order)
│   ├── 00_data_analysis_&_vizualisations.ipynb   <- EDA & Data understanding
│   ├── 01_preprocessing.ipynb                    <- Cleaning & Imputation
│   ├── 02_baseline_comparison.ipynb              <- Simple baseline models
│   ├── 03_feature_engineering_linear.ipynb       <- Advanced feature creation
│   ├── 04_logreg_tuning.ipynb                    <- Logistic Regression baseline
│   ├── 05_tree_training.ipynb                    <- Train Tree Models (XGB, RF, etc.)
│   ├── 06_tree_calibration.ipynb                 <- Probability Calibration
│   ├── 07_tree_shap.ipynb                        <- Model Interpretability (SHAP)
│   ├── 08_tree_evaluation.ipynb                  <- Final Metrics & Financial Analyis
│   └── 09_hobby_leakage_investigation.ipynb      <- Specific hypothesis testing
│
├── final_models       <- Production-ready models selected from experiments
│
├── fraud-detector     <- Deployable API (Flask) for serving predictions
│   ├── app.py
│   └── Dockerfile
│
└── src                <- Source code modules
    ├── features       <- Feature generation scripts
    ├── models         <- Model training scripts
    └── visualization  <- Plotting utilities
```

## 🚀 Getting Started

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Notebooks**:
    Start Jupyter Lab or Jupyter Notebook and execute the notebooks in numerical order (00 -> 08).
    *   **00-01**: Prepare the data.
    *   **02-04**: Establish baselines.
    *   **05**: Perform heavy training (Grid Search).
    *   **06-08**: Refine, explain, and evaluate the best models.

## 🧠 Key Findings
*   **Tree models** (XGBoost, Random Forest) significantly outperform linear baselines.
*   **Calibration** (Platt Scaling) is critical for accurate probability estimation.
*   **SHAP Analysis** reveals that `incident_severity`, `policy_state`, and `insured_hobbies` are top drivers of fraud risk.

## 🛠️ Fraud Detector API
The `fraud-detector/` folder contains a containerized Flask API to serve specific models.
To build and run:
```bash
cd fraud-detector
docker build -t fraud-api .
docker run -p 5000:5000 fraud-api
```
