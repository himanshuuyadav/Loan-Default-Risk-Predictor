"""
Loan Default Predictor - Complete 6-Phase ML Pipeline
Execute: python run_pipeline.py
"""

import os
import sys
import json
import time
from pathlib import Path

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from imblearn.over_sampling import SMOTE

# Import modules
from src.preprocessing import build_model_frame, LoanDefaultPreprocessor
from src.train import prepare_train_test_data, apply_smote
from src.evaluate import evaluate_model, plot_confusion_matrix, plot_roc_curve, find_optimal_threshold, save_metrics_report, print_evaluation_report
from src.eda import run_eda

CONFIG = {
    "data_path": "data/raw/accepted_2007_to_2018Q4.csv",
    "nrows": None,
    "model_dir": "models",
    "reports_dir": "reports",
    "processed_data_path": "data/processed/processed_data.csv",
}

def fit_scaler(X_train, X_test):
    """Fit scaler on training data, apply to both train and test."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return scaler, X_train_scaled, X_test_scaled

def train_models(X_train_scaled, y_train, scale_pos_weight):
    """Train 3 models."""
    models = {}
    
    print("  Training Logistic Regression...")
    models["Logistic Regression"] = LogisticRegression(
        max_iter=1000, class_weight='balanced', random_state=42
    )
    models["Logistic Regression"].fit(X_train_scaled, y_train)
    
    print("  Training Random Forest...")
    models["Random Forest"] = RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_leaf=10,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    models["Random Forest"].fit(X_train_scaled, y_train)
    
    print("  Training XGBoost...")
    models["XGBoost"] = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric='auc', use_label_encoder=False, random_state=42, n_jobs=-1
    )
    models["XGBoost"].fit(X_train_scaled, y_train)
    
    return models

def phase1_eda():
    """PHASE 1: Exploratory Data Analysis"""
    print("\n" + "="*70)
    print("PHASE 1: EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*70)
    
    start_time = time.time()
    try:
        run_eda(CONFIG["data_path"], CONFIG["reports_dir"], nrows=CONFIG["nrows"])
        print("✅ EDA completed successfully")
        figures_dir = Path(CONFIG["reports_dir"]) / "figures"
        figures = list(figures_dir.glob("*.png")) if figures_dir.exists() else []
        print(f"   Generated {len(figures)} PNG figures")
        print(f"⏱️  Completed in {time.time() - start_time:.2f}s")
        return True
    except Exception as e:
        print(f"❌ Phase 1 failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def phase2_preprocessing():
    """PHASE 2: Data Preprocessing & Feature Engineering"""
    print("\n" + "="*70)
    print("PHASE 2: DATA PREPROCESSING & FEATURE ENGINEERING")
    print("="*70)
    
    start_time = time.time()
    try:
        print("Loading raw data...")
        df_raw = build_model_frame(CONFIG["data_path"], nrows=CONFIG["nrows"])
        print(f"   Raw shape: {df_raw.shape}")
        print(f"   Default distribution: {df_raw['default'].value_counts().to_dict()}")
        
        print("Fitting preprocessor...")
        preprocessor = LoanDefaultPreprocessor()
        preprocessor.fit(df_raw)
        
        print("Transforming data...")
        df_processed = preprocessor.transform(df_raw)
        print(f"   Processed shape: {df_processed.shape}")
        print(f"   Feature count: {len(df_processed.columns) - 1}")
        print(f"   Nulls remaining: {df_processed.isnull().sum().sum()}")
        
        # Save processed data
        processed_path = Path(CONFIG["processed_data_path"])
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        df_processed.to_csv(processed_path, index=False)
        print(f"   Saved to {processed_path}")
        
        # Save feature list
        models_dir = Path(CONFIG["model_dir"])
        models_dir.mkdir(parents=True, exist_ok=True)
        feature_list = [col for col in df_processed.columns if col != 'default']
        with open(models_dir / "feature_list.json", "w") as f:
            json.dump(feature_list, f, indent=2)
        
        print(f"✅ Phase 2 completed in {time.time() - start_time:.2f}s")
        return {"preprocessor": preprocessor, "df_processed": df_processed, "feature_list": feature_list}
    except Exception as e:
        print(f"❌ Phase 2 failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def phase3_modeling(phase2_result):
    """PHASE 3: Model Training"""
    print("\n" + "="*70)
    print("PHASE 3: MODEL TRAINING")
    print("="*70)
    
    if not phase2_result:
        print("❌ Skipped - Phase 2 failed")
        return None
    
    start_time = time.time()
    try:
        df_processed = phase2_result["df_processed"]
        feature_list = phase2_result["feature_list"]
        preprocessor = phase2_result["preprocessor"]
        
        print("Preparing train/test split...")
        X = df_processed[feature_list]
        y = df_processed["default"]
        X_train, X_test, y_train, y_test = prepare_train_test_data(X, y)
        print(f"   Train: {X_train.shape}, Test: {X_test.shape}")
        print(f"   Train default rate: {y_train.mean():.4f}")
        
        print("Applying SMOTE...")
        X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)
        print(f"   After SMOTE: {X_train_resampled.shape}, default rate: {y_train_resampled.mean():.4f}")
        
        print("Fitting scaler...")
        scaler, X_train_scaled, X_test_scaled = fit_scaler(X_train_resampled, X_test)
        
        scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
        print(f"   XGBoost scale_pos_weight: {scale_pos_weight:.4f}")
        
        print("Training models...")
        models = train_models(X_train_scaled, y_train_resampled, scale_pos_weight)
        
        # Save models and artifacts
        models_dir = Path(CONFIG["model_dir"])
        for model_name, model in models.items():
            filename = model_name.lower().replace(" ", "_") + ".pkl"
            joblib.dump(model, models_dir / filename)
        joblib.dump(scaler, models_dir / "scaler.pkl")
        joblib.dump(preprocessor, models_dir / "preprocessor.pkl")
        
        print("Testing inference on 5 samples...")
        for model_name, model in models.items():
            probs = model.predict_proba(X_test_scaled[:5])[:, 1]
            print(f"   {model_name}: {probs.round(3)}")
        
        print(f"✅ Phase 3 completed in {time.time() - start_time:.2f}s")
        return {
            "models": models, "scaler": scaler, "preprocessor": preprocessor,
            "feature_list": feature_list, "X_train_scaled": X_train_scaled,
            "X_test_scaled": X_test_scaled, "y_train": y_train, "y_test": y_test,
            "scale_pos_weight": scale_pos_weight,
        }
    except Exception as e:
        print(f"❌ Phase 3 failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def phase4_evaluation(phase3_result):
    """PHASE 4: Model Evaluation"""
    print("\n" + "="*70)
    print("PHASE 4: MODEL EVALUATION")
    print("="*70)
    
    if not phase3_result:
        print("❌ Skipped - Phase 3 failed")
        return None
    
    start_time = time.time()
    try:
        models = phase3_result["models"]
        X_test_scaled = phase3_result["X_test_scaled"]
        y_test = phase3_result["y_test"]
        feature_list = phase3_result["feature_list"]
        
        figures_dir = Path(CONFIG["reports_dir"]) / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Evaluating {len(models)} models...")
        results = {}
        
        for model_name, model in models.items():
            print(f"\n  {model_name}:")
            result = evaluate_model(model, X_test_scaled, y_test, model_name)
            results[model_name] = result
            print_evaluation_report(result)
            
            filename = model_name.lower().replace(" ", "_") + ".png"
            plot_confusion_matrix(y_test, result["y_pred"], model_name, 
                                figures_dir / f"confusion_matrix_{filename}")
        
        print("\nGenerating ROC curve comparison...")
        plot_roc_curve(models, X_test_scaled, y_test, 
                      figures_dir / "roc_curve_comparison.png")
        
        # Identify champion
        best_model_name = max(results, key=lambda n: results[n]["metrics"]["auc_roc"])
        best_model = models[best_model_name]
        
        # Optimal threshold
        threshold_metrics = find_optimal_threshold(y_test, results[best_model_name]["y_prob"])
        
        # Save metrics report
        save_metrics_report(results, Path(CONFIG["reports_dir"]) / "metrics_report.json")
        
        # Print comparison table
        print("\n" + "="*80)
        print("MODEL PERFORMANCE COMPARISON")
        print("="*80)
        print(f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC-ROC':>10}")
        print("-"*80)
        for name in sorted(results.keys()):
            m = results[name]["metrics"]
            print(f"{name:<25} {m['accuracy']:>10.4f} {m['precision']:>10.4f} "
                  f"{m['recall']:>10.4f} {m['f1']:>10.4f} {m['auc_roc']:>10.4f}")
        print("-"*80)
        print(f"🏆 Champion: {best_model_name} (AUC-ROC: {results[best_model_name]['metrics']['auc_roc']:.4f})")
        
        print(f"\n✅ Phase 4 completed in {time.time() - start_time:.2f}s")
        return {
            "results": results, "best_model_name": best_model_name,
            "best_model": best_model, "threshold_metrics": threshold_metrics,
            "X_test_scaled": X_test_scaled, "y_test": y_test, "feature_list": feature_list,
        }
    except Exception as e:
        print(f"❌ Phase 4 failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def phase5_explainability(phase3_result, phase4_result):
    """PHASE 5: SHAP Explainability"""
    print("\n" + "="*70)
    print("PHASE 5: MODEL EXPLAINABILITY (SHAP)")
    print("="*70)
    
    if not phase3_result or not phase4_result:
        print("❌ Skipped - Phase 3 or 4 failed")
        return False
    
    start_time = time.time()
    try:
        import shap
        
        best_model = phase4_result["best_model"]
        X_test_scaled = phase4_result["X_test_scaled"]
        feature_list = phase4_result["feature_list"]
        
        print(f"Champion model: {phase4_result['best_model_name']}")
        
        # Sample for speed
        sample_size = min(1000, len(X_test_scaled))
        X_sample = X_test_scaled[:sample_size]
        
        print(f"Creating SHAP explainer for {sample_size} samples...")
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_sample)
        
        figures_dir = Path(CONFIG["reports_dir"]) / "figures"
        
        # Bar plot
        print("Generating SHAP importance bar plot...")
        plt.figure()
        shap.summary_plot(shap_values, X_sample, feature_names=feature_list, 
                         plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(figures_dir / "shap_importance.png", dpi=150, bbox_inches="tight")
        plt.close()
        
        # Beeswarm plot
        print("Generating SHAP beeswarm plot...")
        plt.figure()
        shap.summary_plot(shap_values, X_sample, feature_names=feature_list, show=False)
        plt.tight_layout()
        plt.savefig(figures_dir / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
        plt.close()
        
        # Top 5 features
        shap_array = shap_values if isinstance(shap_values, np.ndarray) else shap_values[1]
        mean_shap = np.abs(shap_array).mean(axis=0)
        top_5_idx = np.argsort(mean_shap)[-5:][::-1]
        
        print("\nTop 5 Most Impactful Features:")
        for i, idx in enumerate(top_5_idx, 1):
            print(f"  {i}. {feature_list[idx]}: {mean_shap[idx]:.4f}")
        
        # Sample waterfalls
        print("Generating waterfall plots...")
        for idx in range(min(3, len(X_sample))):
            plt.figure()
            shap.waterfall_plot(
                shap.Explanation(values=shap_array[idx], base_values=explainer.expected_value,
                               data=X_sample[idx], feature_names=feature_list),
                show=False
            )
            plt.tight_layout()
            plt.savefig(figures_dir / f"waterfall_sample_{idx}.png", dpi=150, bbox_inches="tight")
            plt.close()
        
        print(f"✅ Phase 5 completed in {time.time() - start_time:.2f}s")
        return True
    except Exception as e:
        print(f"❌ Phase 5 failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def phase6_validation(phase3_result, phase4_result):
    """PHASE 6: Final Validation"""
    print("\n" + "="*70)
    print("PHASE 6: FINAL VALIDATION & ARTIFACT VERIFICATION")
    print("="*70)
    
    if not phase3_result or not phase4_result:
        print("❌ Skipped")
        return False
    
    start_time = time.time()
    try:
        # Verify figures
        figures_dir = Path(CONFIG["reports_dir"]) / "figures"
        expected = ["target_distribution.png", "numeric_distributions.png", 
                   "default_by_grade.png", "correlation_heatmap.png",
                   "roc_curve_comparison.png", "shap_importance.png", "shap_beeswarm.png"]
        
        print("\nFigures Generated:")
        for fig in expected:
            exists = "✅" if (figures_dir / fig).exists() else "❌"
            print(f"  {exists} {fig}")
        
        # Verify models
        models_dir = Path(CONFIG["model_dir"])
        expected_models = ["logistic_regression.pkl", "random_forest.pkl", 
                          "xgboost.pkl", "scaler.pkl", "preprocessor.pkl", "feature_list.json"]
        
        print("\nModel Artifacts:")
        for model_file in expected_models:
            exists = "✅" if (models_dir / model_file).exists() else "❌"
            print(f"  {exists} {model_file}")
        
        # Verify processed data
        exists = "✅" if Path(CONFIG["processed_data_path"]).exists() else "❌"
        print(f"\nProcessed Data:")
        print(f"  {exists} {CONFIG['processed_data_path']}")
        
        # Test inference
        print("\nTesting end-to-end inference...")
        scaler = joblib.load(models_dir / "scaler.pkl")
        xgb_model = joblib.load(models_dir / "xgboost.pkl")
        X_test_scaled = phase3_result["X_test_scaled"]
        y_test = phase3_result["y_test"]
        
        random_idx = np.random.choice(len(X_test_scaled), min(5, len(X_test_scaled)), replace=False)
        for idx in random_idx:
            pred = xgb_model.predict([X_test_scaled[idx]])[0]
            prob = xgb_model.predict_proba([X_test_scaled[idx]])[0, 1]
            actual = y_test.iloc[idx]
            match = "✓" if (pred == actual) else "✗"
            print(f"  {match} Pred: {pred}, Prob: {prob:.4f}, Actual: {actual}")
        
        # Final metrics
        print("\n" + "="*80)
        print("FINAL PERFORMANCE SUMMARY")
        print("="*80)
        results = phase4_result["results"]
        for name in sorted(results.keys()):
            m = results[name]["metrics"]
            print(f"\n{name}:")
            print(f"  Accuracy:  {m['accuracy']:.4f}")
            print(f"  Precision: {m['precision']:.4f}")
            print(f"  Recall:    {m['recall']:.4f}")
            print(f"  F1:        {m['f1']:.4f}")
            print(f"  AUC-ROC:   {m['auc_roc']:.4f}")
        
        best = phase4_result["best_model_name"]
        print(f"\n🏆 Champion: {best} (AUC-ROC: {results[best]['metrics']['auc_roc']:.4f})")
        
        print(f"\n✅ Phase 6 completed in {time.time() - start_time:.2f}s")
        return True
    except Exception as e:
        print(f"❌ Phase 6 failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Execute all 6 phases"""
    print("\n" + "="*70)
    print("🚀 LOAN DEFAULT PREDICTOR - 6 PHASE PIPELINE EXECUTION")
    print("="*70)
    
    total_start = time.time()
    
    if not phase1_eda():
        return
    if not (p2 := phase2_preprocessing()):
        return
    if not (p3 := phase3_modeling(p2)):
        return
    if not (p4 := phase4_evaluation(p3)):
        return
    phase5_explainability(p3, p4)
    phase6_validation(p3, p4)
    
    total = (time.time() - total_start) / 60
    print("\n" + "="*70)
    print(f"✅ PIPELINE COMPLETE - Total time: {total:.2f} minutes")
    print("="*70)

if __name__ == "__main__":
    main()
