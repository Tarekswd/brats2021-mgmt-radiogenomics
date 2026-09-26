"""
Evaluation Metrics for BraTS 2021 Task 2 (MGMT Radiogenomics).
Author: Tarek Al Ahmadieh (Sole Author & Lead Researcher)
Computes ROC-AUC, Accuracy, F1, Log-Loss, Sensitivity, Specificity, and Brier Score.
"""
from typing import Dict, List, Union
import numpy as np
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, log_loss,
    brier_score_loss, confusion_matrix, precision_score, recall_score
)

def compute_mgmt_metrics(
    y_true: Union[np.ndarray, List[int]],
    y_prob: Union[np.ndarray, List[float]],
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Computes all standard challenge evaluation metrics for binary classification.
    """
    y_true = np.array(y_true).astype(int)
    y_prob = np.array(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    
    # Check for single-class edge case in small batches
    if len(np.unique(y_true)) > 1:
        auc = float(roc_auc_score(y_true, y_prob))
        ll = float(log_loss(y_true, y_prob))
    else:
        auc = 0.5
        ll = 0.693
        
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0)) # Sensitivity
    brier = float(brier_score_loss(y_true, y_prob))
    
    # Specificity
    if len(np.unique(y_true)) > 1:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        spec = float(tn / (tn + fp + 1e-7))
    else:
        spec = 0.5
        
    return {
        "roc_auc": auc,
        "accuracy": acc,
        "f1_score": f1,
        "precision": prec,
        "sensitivity": rec,
        "specificity": spec,
        "log_loss": ll,
        "brier_score": brier
    }
