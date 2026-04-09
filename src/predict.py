import joblib
import pandas as pd
import json

def load_prediction_artifacts(model_path, scaler_path, feature_path):
    """
    Load model, scaler, and feature list for inference.
    """
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with open(feature_path, 'r') as f:
        features = json.load(f)
    return model, scaler, features

def predict_loan_default(applicant_data, model, scaler, features):
    """
    Predict default risk for a new loan applicant.
    
    applicant_data: dict or DataFrame with raw features.
    """
    if isinstance(applicant_data, dict):
        df_input = pd.DataFrame([applicant_data])
    else:
        df_input = applicant_data
        
    # Ensure all required features are present (pre-processing should happen here too)
    # This is a simplified version; in production, you'd apply the exact same pipeline.
    
    df_scaled = scaler.transform(df_input[features])
    prob = model.predict_proba(df_scaled)[0][1]
    
    return {
        'default_probability': round(float(prob), 4),
        'risk_label': 'HIGH' if prob > 0.5 else 'LOW',
        'recommendation': 'DECLINE' if prob > 0.5 else 'APPROVE'
    }
