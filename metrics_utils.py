import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix
)

def convert_to_binary(y_true, y_pred):
    """Convert continuous values to binary using median threshold"""
    try:
        threshold = np.median(y_true)
        return (y_true >= threshold).astype(int), (y_pred >= threshold).astype(int)
    except Exception:
        return None, None

def calculate_metrics(y_true, y_pred, probas=None):
    """Calculate classification metrics"""
    try:
        # Convert to binary if needed
        if not isinstance(y_true[0], (np.integer, int)):
            y_true, y_pred = convert_to_binary(y_true, y_pred)
        
        if y_true is None:
            return None, None, None
            
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
        
        metrics = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall/Sensitivity": recall_score(y_true, y_pred, zero_division=0),
            "Specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
            "F1 Score": f1_score(y_true, y_pred, zero_division=0)
        }
        
        # Calculate ROC and AUC if probabilities provided
        if probas is not None:
            fpr, tpr, _ = roc_curve(y_true, probas)
            metrics["AUC"] = auc(fpr, tpr)
            return metrics, (tn, fp, fn, tp), (fpr, tpr)
        
        return metrics, (tn, fp, fn, tp), None
        
    except Exception as e:
        print(f"Error in metrics calculation: {e}")
        return None, None, None
