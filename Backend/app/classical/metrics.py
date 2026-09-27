"""Comprehensive biomedical classification evaluation metrics for Q-BioForge.

Calculates:
- Accuracy
- Precision
- Recall (Sensitivity)
- F1-Score
- ROC-AUC
- Specificity (True Negative Rate)
- Brier Score (Probabilistic Calibration error)
"""

from typing import Dict, Any, Optional, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    brier_score_loss,
)
from pydantic import BaseModel, Field


class ClassificationMetrics(BaseModel):
    """Container for computed evaluation metrics on a single data split."""
    accuracy: float = Field(..., description="Accuracy score")
    precision: float = Field(..., description="Precision score")
    recall: float = Field(..., description="Recall (Sensitivity) score")
    f1_score: float = Field(..., description="Harmonic mean of precision and recall")
    roc_auc: Optional[float] = Field(None, description="Area under ROC curve")
    sensitivity: float = Field(..., description="True Positive Rate: TP / (TP + FN)")
    specificity: float = Field(..., description="True Negative Rate: TN / (TN + FP)")
    brier_score: Optional[float] = Field(None, description="Brier calibration loss: mean((p - y)^2)")
    confusion_matrix: Dict[str, int] = Field(default_factory=dict, description="TP, FP, TN, FN counts")


class BenchmarkRecord(BaseModel):
    """Container for a single benchmark evaluation record."""
    experiment_id: str
    model: str
    dataset: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: Optional[float] = None
    sensitivity: float
    specificity: float
    brier_score: Optional[float] = None
    training_time: float
    inference_time: float
    random_seed: int
    timestamp: str
    status: str = "completed"


def calculate_metrics(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    y_prob: Optional[Union[np.ndarray, list]] = None,
) -> ClassificationMetrics:
    """Calculate all standard and biomedical classification metrics from real predictions.
    
    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_pred: Predicted discrete binary labels (0 or 1).
        y_prob: Predicted probabilities for the positive class (class 1) or full 2D probabilities.
        
    Returns:
        ClassificationMetrics instance with measured values.
    """
    y_t = np.asarray(y_true).astype(int)
    y_p = np.asarray(y_pred).astype(int)

    acc = float(accuracy_score(y_t, y_p))
    prec = float(precision_score(y_t, y_p, zero_division=0))
    rec = float(recall_score(y_t, y_p, zero_division=0))
    f1 = float(f1_score(y_t, y_p, zero_division=0))

    # Confusion matrix extraction: labels=[0, 1] ensures [tn, fp], [fn, tp]
    cm = confusion_matrix(y_t, y_p, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # Edge case if only 1 class in sample
        tn = int(np.sum((y_t == 0) & (y_p == 0)))
        fp = int(np.sum((y_t == 0) & (y_p == 1)))
        fn = int(np.sum((y_t == 1) & (y_p == 0)))
        tp = int(np.sum((y_t == 1) & (y_p == 1)))

    # Sensitivity = TP / (TP + FN)
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    # Specificity = TN / (TN + FP)
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # ROC-AUC and Brier score calculation
    roc_auc_val: Optional[float] = None
    brier_val: Optional[float] = None

    if y_prob is not None:
        y_pr = np.asarray(y_prob, dtype=np.float64)
        try:
            if y_pr.ndim == 2 and y_pr.shape[1] >= 2:
                y_pr_pos = y_pr[:, 1]
            else:
                y_pr_pos = y_pr.ravel()

            # ROC-AUC requires at least 2 distinct classes in y_true
            if len(np.unique(y_t)) > 1:
                roc_auc_val = float(roc_auc_score(y_t, y_pr_pos))
            else:
                roc_auc_val = None

            brier_val = float(brier_score_loss(y_t, y_pr_pos))
        except Exception:
            roc_auc_val = None
            brier_val = None

    return ClassificationMetrics(
        accuracy=round(acc, 4),
        precision=round(prec, 4),
        recall=round(rec, 4),
        f1_score=round(f1, 4),
        roc_auc=round(roc_auc_val, 4) if roc_auc_val is not None else None,
        sensitivity=round(sensitivity, 4),
        specificity=round(specificity, 4),
        brier_score=round(brier_val, 4) if brier_val is not None else None,
        confusion_matrix={"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    )
