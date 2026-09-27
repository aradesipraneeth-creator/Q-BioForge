"""Evaluation metrics and calibration assessment for Quantum Classifiers in Q-BioForge.

Calculates:
- Accuracy
- Precision
- Recall (Sensitivity: True Positive Rate)
- Specificity (True Negative Rate)
- F1-Score
- ROC-AUC
- Brier Calibration Score (when probabilistic calibration is active)
- Raw expectation value calibration notes
"""

from typing import Any, Dict, List, Optional, Union
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


class QuantumClassificationMetrics(BaseModel):
    """Container for evaluated quantum classification metrics."""
    accuracy: float = Field(..., description="Classification accuracy")
    precision: float = Field(..., description="Precision: TP / (TP + FP)")
    recall: float = Field(..., description="Recall (Sensitivity): TP / (TP + FN)")
    f1_score: float = Field(..., description="F1-Score")
    roc_auc: Optional[float] = Field(None, description="Area under ROC curve")
    sensitivity: float = Field(..., description="Sensitivity: TP / (TP + FN)")
    specificity: float = Field(..., description="Specificity: TN / (TN + FP)")
    brier_score: Optional[float] = Field(None, description="Brier calibration loss: mean((p - y)^2)")
    confusion_matrix: Dict[str, int] = Field(default_factory=dict, description="TP, FP, TN, FN counts")
    scientific_notes: str = Field(
        default="Probabilities derived via affine mapping P(y=1) = (1 + <Z>)/2 from PauliZ expectation values.",
        description="Scientific interpretation of quantum readout."
    )


def calculate_quantum_metrics(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    y_prob: Optional[Union[np.ndarray, list]] = None,
) -> QuantumClassificationMetrics:
    """Compute rigorous classification metrics for quantum model predictions."""
    y_t = np.asarray(y_true).astype(int)
    y_p = np.asarray(y_pred).astype(int)

    acc = float(accuracy_score(y_t, y_p))
    prec = float(precision_score(y_t, y_p, zero_division=0))
    rec = float(recall_score(y_t, y_p, zero_division=0))
    f1 = float(f1_score(y_t, y_p, zero_division=0))

    # Confusion Matrix [ [TN, FP], [FN, TP] ]
    cm = confusion_matrix(y_t, y_p, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = int(np.sum((y_t == 0) & (y_p == 0)))
        fp = int(np.sum((y_t == 0) & (y_p == 1)))
        fn = int(np.sum((y_t == 1) & (y_p == 0)))
        tp = int(np.sum((y_t == 1) & (y_p == 1)))

    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    roc_auc_val: Optional[float] = None
    brier_val: Optional[float] = None

    if y_prob is not None:
        y_pr = np.asarray(y_prob, dtype=np.float64)
        if y_pr.ndim == 2 and y_pr.shape[1] >= 2:
            y_pr_pos = y_pr[:, 1]
        else:
            y_pr_pos = y_pr.ravel()

        try:
            if len(np.unique(y_t)) > 1:
                roc_auc_val = float(roc_auc_score(y_t, y_pr_pos))
            brier_val = float(brier_score_loss(y_t, y_pr_pos))
        except Exception:
            roc_auc_val = None
            brier_val = None

    return QuantumClassificationMetrics(
        accuracy=round(acc, 4),
        precision=round(prec, 4),
        recall=round(rec, 4),
        f1_score=round(f1, 4),
        roc_auc=round(roc_auc_val, 4) if roc_auc_val is not None else None,
        sensitivity=round(sensitivity, 4),
        specificity=round(specificity, 4),
        brier_score=round(brier_val, 4) if brier_val is not None else None,
        confusion_matrix={"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        scientific_notes=(
            "Quantum measurement: PauliZ expectation value <Z> in [-1, 1] mapped via affine transform "
            "P(y=1|x) = (1 + <Z>)/2. Evaluated on holdout test partition."
        ),
    )
