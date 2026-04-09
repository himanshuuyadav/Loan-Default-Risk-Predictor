from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd


def load_prediction_artifacts(
    model_path: str | Path,
    scaler_path: str | Path,
    feature_path: str | Path,
    preprocessor_path: str | Path,
    metadata_path: str | Path | None = None,
):
    """Load all artifacts required for deployment-style inference."""

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    preprocessor = joblib.load(preprocessor_path)

    with Path(feature_path).open("r", encoding="utf-8") as file:
        features = json.load(file)

    metadata = {}
    if metadata_path and Path(metadata_path).exists():
        with Path(metadata_path).open("r", encoding="utf-8") as file:
            metadata = json.load(file)

    return model, scaler, preprocessor, features, metadata


def predict_loan_default(
    applicant_data: Dict | pd.DataFrame,
    model,
    scaler,
    preprocessor,
    features,
    threshold: float = 0.5,
) -> Dict[str, float | str]:
    """Predict default risk using raw applicant data."""

    transformed = preprocessor.transform_applicant(applicant_data)
    transformed = transformed.reindex(columns=features, fill_value=0.0)
    scaled = scaler.transform(transformed)

    probability = float(model.predict_proba(scaled)[0][1])
    prediction = int(probability >= threshold)

    return {
        "default_probability": round(probability, 4),
        "prediction": prediction,
        "risk_label": "HIGH" if prediction == 1 else "LOW",
        "recommendation": "DECLINE" if prediction == 1 else "APPROVE",
        "decision_threshold": round(float(threshold), 4),
    }
