from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import joblib
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

from src.evaluate import (
    evaluate_model,
    find_optimal_threshold,
    plot_confusion_matrix,
    plot_roc_curve,
    print_evaluation_report,
    save_metrics_report,
)
from src.preprocessing import LoanDefaultPreprocessor, build_model_frame


def prepare_train_test_data(X, y, test_size: float = 0.2, random_state: int = 42):
    """Split the dataset while preserving class balance."""

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def apply_smote(
    X_train,
    y_train,
    sampling_strategy: float = 0.5,
    random_state: int = 42,
    k_neighbors: int = 5,
):
    """Address class imbalance using SMOTE."""

    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        random_state=random_state,
        k_neighbors=k_neighbors,
    )
    return smote.fit_resample(X_train, y_train)


def fit_scaler(X_train, X_test):
    """Fit a scaler on the training split and transform both splits."""

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return scaler, X_train_scaled, X_test_scaled


def train_models(X_train_scaled, y_train_resampled, scale_pos_weight: float) -> Dict[str, object]:
    """Train the baseline, tree, and boosting models from the plan."""

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="auc",
            random_state=42,
            n_jobs=-1,
        ),
    }

    for model in models.values():
        model.fit(X_train_scaled, y_train_resampled)

    return models


def tune_xgboost_random_search(X_train_scaled, y_train_resampled, scale_pos_weight: float):
    """Tune XGBoost with RandomizedSearchCV."""

    param_dist = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 5, 7, 9, 12],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5, 7],
    }

    estimator = xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
    )
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_dist,
        n_iter=20,
        cv=3,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1,
        random_state=42,
    )
    search.fit(X_train_scaled, y_train_resampled)
    return search.best_estimator_, {"best_score": float(search.best_score_), "best_params": search.best_params_}


def tune_xgboost_optuna(X_train_scaled, y_train_resampled, scale_pos_weight: float, n_trials: int):
    """Tune XGBoost with Optuna."""

    import optuna

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
            "scale_pos_weight": scale_pos_weight,
            "eval_metric": "auc",
            "random_state": 42,
            "n_jobs": -1,
        }

        model = xgb.XGBClassifier(**params)
        scores = cross_val_score(
            model,
            X_train_scaled,
            y_train_resampled,
            cv=3,
            scoring="roc_auc",
            n_jobs=-1,
        )
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_model = xgb.XGBClassifier(
        **study.best_params,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
    )
    best_model.fit(X_train_scaled, y_train_resampled)
    return best_model, {"best_score": float(study.best_value), "best_params": study.best_params}


def explain_with_shap(model, X_test_scaled, feature_names, output_dir: str | Path, sample_size: int = 1000) -> None:
    """Generate SHAP summary plots for the best tree-based model."""

    import matplotlib.pyplot as plt
    import shap

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_size = min(sample_size, len(X_test_scaled))
    sample = X_test_scaled[:sample_size]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    plt.figure()
    shap.summary_plot(
        shap_values,
        sample,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
    )
    plt.tight_layout()
    plt.savefig(output_dir / "shap_importance.png")
    plt.close()

    shap.summary_plot(
        shap_values,
        sample,
        feature_names=feature_names,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(output_dir / "shap_beeswarm.png")
    plt.close()


def save_artifacts(
    model,
    scaler,
    preprocessor,
    feature_list,
    metadata,
    model_dir: str | Path,
) -> None:
    """Save the deployment artifacts described in the plan."""

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_dir / "xgb_loan_default_model.pkl")
    joblib.dump(scaler, model_dir / "scaler.pkl")
    joblib.dump(preprocessor, model_dir / "preprocessor.pkl")

    with (model_dir / "feature_list.json").open("w", encoding="utf-8") as file:
        json.dump(feature_list, file, indent=2)

    with (model_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


def save_processed_dataset(processed_frame, output_path: str | Path) -> None:
    """Persist the prepared modeling dataset for reproducibility."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_frame.to_csv(output_path, index=False)


def run_training_pipeline(
    data_path: str | Path,
    model_dir: str | Path,
    reports_dir: str | Path,
    processed_data_path: str | Path = "data/processed/prepared_dataset.csv",
    nrows: int | None = None,
    tune_method: str = "none",
    optuna_trials: int = 20,
    enable_shap: bool = True,
):
    """Execute the end-to-end training workflow."""

    reports_dir = Path(reports_dir)
    figure_dir = reports_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    model_frame = build_model_frame(data_path, nrows=nrows)

    preprocessor = LoanDefaultPreprocessor()
    processed = preprocessor.fit_transform(model_frame)
    save_processed_dataset(processed, processed_data_path)

    X = processed.drop(columns=["default"])
    y = processed["default"]

    X_train, X_test, y_train, y_test = prepare_train_test_data(X, y)
    X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)

    scaler, X_train_scaled, X_test_scaled = fit_scaler(X_train_resampled, X_test)
    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

    models = train_models(X_train_scaled, y_train_resampled, scale_pos_weight=scale_pos_weight)

    tuning_result = None
    if tune_method == "random":
        tuned_model, tuning_result = tune_xgboost_random_search(
            X_train_scaled,
            y_train_resampled,
            scale_pos_weight,
        )
        models["XGBoost (Tuned)"] = tuned_model
    elif tune_method == "optuna":
        tuned_model, tuning_result = tune_xgboost_optuna(
            X_train_scaled,
            y_train_resampled,
            scale_pos_weight,
            n_trials=optuna_trials,
        )
        models["XGBoost (Tuned)"] = tuned_model

    results = {}
    for model_name, model in models.items():
        result = evaluate_model(model, X_test_scaled, y_test, model_name)
        results[model_name] = result
        print_evaluation_report(result)
        plot_confusion_matrix(
            y_test,
            result["y_pred"],
            model_name,
            figure_dir / f"confusion_matrix_{model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png",
        )

    plot_roc_curve(models, X_test_scaled, y_test, figure_dir / "roc_curve_comparison.png")
    save_metrics_report(results, reports_dir / "metrics.json")

    best_model_name = max(results, key=lambda name: results[name]["metrics"]["auc_roc"])
    best_model = models[best_model_name]
    threshold_metrics = find_optimal_threshold(y_test, results[best_model_name]["y_prob"])

    if enable_shap and isinstance(best_model, (xgb.XGBClassifier, RandomForestClassifier)):
        explain_with_shap(best_model, X_test_scaled, X.columns.tolist(), figure_dir)

    metadata = {
        "best_model_name": best_model_name,
        "optimal_threshold": threshold_metrics,
        "feature_count": len(X.columns),
        "train_shape": list(X_train.shape),
        "test_shape": list(X_test.shape),
        "class_balance": {
            "train_default_rate": float(y_train.mean()),
            "test_default_rate": float(y_test.mean()),
        },
        "tuning": tuning_result,
    }

    save_artifacts(
        best_model,
        scaler,
        preprocessor,
        X.columns.tolist(),
        metadata,
        model_dir=model_dir,
    )

    return {
        "best_model_name": best_model_name,
        "results": results,
        "threshold_metrics": threshold_metrics,
        "metadata": metadata,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train the loan default prediction models.")
    parser.add_argument("--data-path", required=True, help="Path to the raw Lending Club CSV file.")
    parser.add_argument("--model-dir", default="models", help="Directory for saved model artifacts.")
    parser.add_argument("--reports-dir", default="reports", help="Directory for evaluation reports and figures.")
    parser.add_argument(
        "--processed-data-path",
        default="data/processed/prepared_dataset.csv",
        help="Path for saving the cleaned and engineered modeling dataset.",
    )
    parser.add_argument("--nrows", type=int, default=None, help="Optional row limit for quick experiments.")
    parser.add_argument(
        "--tune-method",
        choices=["none", "random", "optuna"],
        default="none",
        help="Hyperparameter tuning strategy for XGBoost.",
    )
    parser.add_argument("--optuna-trials", type=int, default=20, help="Number of Optuna trials when enabled.")
    parser.add_argument(
        "--skip-shap",
        action="store_true",
        help="Skip SHAP explainability plots.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training_pipeline(
        data_path=args.data_path,
        model_dir=args.model_dir,
        reports_dir=args.reports_dir,
        processed_data_path=args.processed_data_path,
        nrows=args.nrows,
        tune_method=args.tune_method,
        optuna_trials=args.optuna_trials,
        enable_shap=not args.skip_shap,
    )
