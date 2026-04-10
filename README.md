# Loan Default Risk Prediction

An end-to-end machine learning pipeline for loan default prediction using the Lending Club dataset.

## What Is Implemented

- Raw data loading, loan-status filtering, and binary target creation
- Reusable preprocessing pipeline shared by training and inference
- Feature engineering for FICO averages, income ratios, DTI bands, and risk flags
- EDA figure generation and JSON summary output
- Stratified train/test split, SMOTE balancing, and feature scaling
- Multi-model training with Logistic Regression, Random Forest, and XGBoost
- Evaluation reports with metrics, confusion matrices, ROC comparison, and threshold optimization
- Optional XGBoost tuning with RandomizedSearchCV or Optuna
- SHAP explainability plots for the best tree-based model
- Deployment artifacts for prediction: model, scaler, preprocessor, feature list, and metadata

## Quick Start

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Run EDA:

```bash
python -m src.eda --data-path data/raw/accepted_2007_to_2018Q4.csv --reports-dir reports
```

Prepare the modeling dataset:

```bash
python -m src.prepare_data --data-path data/raw/accepted_2007_to_2018Q4.csv --output-path data/processed/prepared_dataset.csv
```

Run training:

```bash
python -m src.train --data-path data/raw/accepted_2007_to_2018Q4.csv --model-dir models --reports-dir reports --tune-method optuna
```

Run prediction from a JSON payload:

```bash
python -m src.predict --input-json sample_applicant.json
```

## Key Output Files

- `reports/figures/*.png`: EDA, confusion matrices, ROC, and SHAP plots
- `reports/eda_summary.json`: EDA summary statistics
- `reports/metrics.json`: model evaluation reports
- `models/xgb_loan_default_model.pkl`: saved best model artifact
- `models/scaler.pkl`: fitted scaler
- `models/preprocessor.pkl`: reusable preprocessing pipeline
- `models/feature_list.json`: expected model input columns
- `models/metadata.json`: best model name, threshold, and training metadata
- `data/processed/prepared_dataset.csv`: cleaned and engineered modeling table
