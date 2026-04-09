import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)

def get_metrics(model, X_test, y_test):
    """
    Calculate and return key classification metrics.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc_roc': roc_auc_score(y_test, y_prob)
    }
    return metrics, y_pred, y_prob

def print_evaluation_report(model_name, metrics, y_test, y_pred):
    """
    Print a formatted evaluation report.
    """
    print(f"\n{'='*50}")
    print(f"MODEL: {model_name}")
    print(f"{'='*50}")
    for k, v in metrics.items():
        print(f"{k.capitalize():10}: {v:.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

def plot_confusion_matrix(y_test, y_pred, model_name, save_path=None):
    """
    Plot and optionally save the confusion matrix.
    """
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Default', 'Default'],
                yticklabels=['No Default', 'Default'])
    plt.title(f'Confusion Matrix — {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
