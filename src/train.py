import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

def prepare_train_test_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def apply_smote(X_train, y_train, sampling_strategy=0.5, random_state=42):
    """
    Handle class imbalance using SMOTE.
    """
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    return smote.fit_resample(X_train, y_train)

def train_xgb_model(X_train, y_train, params=None):
    """
    Train an XGBoost model.
    """
    if params is None:
        params = {
            'n_estimators': 300,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    return model

def save_artifacts(model, scaler, feature_list, model_path, scaler_path, feature_path):
    """
    Save model, scaler, and feature list.
    """
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    import json
    with open(feature_path, 'w') as f:
        json.dump(feature_list, f)
