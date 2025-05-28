import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix
)

def get_confusion_matrix_values(y_true, y_pred):
    """Calculate confusion matrix values with error handling"""
    try:
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            return cm.ravel()
        return [0, 0, 0, np.sum(cm)]
    except Exception:
        return [0, 0, 0, 0]

def calculate_binary_metrics(y_true, y_pred):
    """Calculate binary classification metrics"""
    try:
        tn, fp, fn, tp = get_confusion_matrix_values(y_true, y_pred)
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        metrics = {
            "Accuracy": accuracy,
            "Precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
            "Recall/Sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0,
            "Specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
            "F1 Score": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        }
        return metrics, (tn, fp, fn, tp)
    except Exception:
        return None, None

def calculate_roc_metrics(y_true, y_scores):
    """Calculate ROC curve and AUC"""
    try:
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        return fpr, tpr, roc_auc
    except Exception:
        return None, None, None
