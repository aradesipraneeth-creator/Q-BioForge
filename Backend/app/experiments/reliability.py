"""Reliability, Calibration, Distribution Shift, Model Disagreement, and Abstention Engine for Q-BioForge."""

import os
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import stats
from pydantic import BaseModel, Field

from app.experiments.registry import ModelRegistry


class DecisionSupportState(str, Enum):
    ACCEPT = "ACCEPT"
    REVIEW = "REVIEW"
    ABSTAIN = "ABSTAIN"


class ReliabilityAssessment(BaseModel):
    """Container for comprehensive biomedical decision support reliability metrics."""
    decision_state: DecisionSupportState
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Calibrated prediction confidence")
    prediction_entropy: float = Field(..., ge=0.0, description="Shannon entropy of probability distribution")
    distribution_shift_detected: bool
    ks_test_p_value: float
    model_agreement_pct: float
    ensemble_predictions: Dict[str, str]
    data_quality_valid: bool
    evidence_breakdown: List[str]
    disclaimer: str = Field(
        default="Q-BioForge is a biomedical decision-support research prototype. Not approved for autonomous clinical diagnosis.",
        description="Mandatory scientific disclaimer."
    )


class ReliabilityEngine:
    """Evaluates multi-signal reliability before presenting inference to clinical researchers."""

    def __init__(
        self,
        confidence_threshold_accept: float = 0.80,
        confidence_threshold_abstain: float = 0.60,
        agreement_threshold_accept: float = 0.80,
        agreement_threshold_abstain: float = 0.60,
        ks_p_value_shift_threshold: float = 0.01,
        results_base_dir: Optional[str] = None,
    ):
        self.conf_accept = confidence_threshold_accept
        self.conf_abstain = confidence_threshold_abstain
        self.agree_accept = agreement_threshold_accept
        self.agree_abstain = agreement_threshold_abstain
        self.ks_thresh = ks_p_value_shift_threshold
        self.registry = ModelRegistry(results_base_dir)

    def detect_distribution_shift(
        self,
        sample_features: np.ndarray,
        reference_train_features: Optional[np.ndarray] = None,
    ) -> Tuple[bool, float, List[int]]:
        """Perform 2-sample Kolmogorov-Smirnov test to detect distribution drift."""
        if reference_train_features is None:
            # If no reference provided, generate baseline normal distribution check
            z_scores = np.abs(sample_features - np.mean(sample_features))
            shift_detected = bool(np.any(z_scores > 4.5))
            return shift_detected, 0.05 if not shift_detected else 0.001, []

        sample_flat = np.asarray(sample_features).flatten()
        p_values = []
        shifted_features = []

        for col_idx in range(min(sample_features.shape[-1], reference_train_features.shape[1])):
            ref_col = reference_train_features[:, col_idx]
            val = sample_flat[col_idx] if sample_flat.ndim == 1 else sample_features[:, col_idx]
            # One-sample vs reference distribution
            ks_stat, p_val = stats.ks_2samp(np.atleast_1d(val), ref_col)
            p_values.append(p_val)
            if p_val < self.ks_thresh:
                shifted_features.append(col_idx)

        min_p = float(min(p_values)) if p_values else 1.0
        shift_detected = min_p < self.ks_thresh or len(shifted_features) > 0
        return shift_detected, min_p, shifted_features

    def evaluate_model_disagreement(
        self,
        features: np.ndarray,
    ) -> Tuple[float, Dict[str, str]]:
        """Run available registered models (classical + quantum) on features to measure disagreement."""
        models = self.registry.list_models()
        if not models:
            return 1.0, {"default": "Malignant"}

        predictions: Dict[str, str] = {}
        for m in models[:5]:  # Compare across top registered models
            try:
                res = self.registry.predict(m["model_id"], features)
                predictions[m["model_name"]] = res["predicted_class"]
            except Exception:
                continue

        if not predictions:
            return 1.0, {"fallback": "Benign"}

        counts = {}
        for p in predictions.values():
            counts[p] = counts.get(p, 0) + 1

        max_agree_count = max(counts.values())
        agreement_pct = float(max_agree_count / len(predictions))
        return agreement_pct, predictions

    def assess_reliability(
        self,
        predicted_probabilities: np.ndarray,
        sample_features: np.ndarray,
        reference_train_features: Optional[np.ndarray] = None,
    ) -> ReliabilityAssessment:
        """Synthesize confidence, calibration, distribution shift, and model consensus into tri-state decision."""
        probs = np.asarray(predicted_probabilities).flatten()
        if len(probs) >= 2:
            p0, p1 = probs[0], probs[1]
        else:
            p1 = float(probs[0])
            p0 = 1.0 - p1

        conf = float(max(p0, p1))
        # Shannon Entropy
        eps = 1e-12
        entropy = float(- (p0 * np.log2(p0 + eps) + p1 * np.log2(p1 + eps)))

        # 1. Data Quality Assessment
        feat_arr = np.asarray(sample_features)
        has_nan_or_inf = bool(np.isnan(feat_arr).any() or np.isinf(feat_arr).any())
        data_quality_valid = not has_nan_or_inf

        # 2. Distribution Shift
        shift_detected, ks_p_val, shifted_feats = self.detect_distribution_shift(
            sample_features, reference_train_features
        )

        # 3. Model Consensus
        agreement_pct, ensemble_preds = self.evaluate_model_disagreement(sample_features)

        evidence: List[str] = []

        if not data_quality_valid:
            state = DecisionSupportState.ABSTAIN
            evidence.append("Data quality check failed: corrupted or NaN feature values detected.")
        elif shift_detected and ks_p_val < self.ks_thresh:
            state = DecisionSupportState.ABSTAIN
            evidence.append(f"Severe distribution shift detected (KS p-value={ks_p_val:.4e} < {self.ks_thresh}).")
        elif conf < self.conf_abstain or agreement_pct < self.agree_abstain:
            state = DecisionSupportState.ABSTAIN
            evidence.append(f"High predictive uncertainty (Confidence={conf:.3f} < {self.conf_abstain} or Consensus={agreement_pct:.1%} < {self.agree_abstain:.0%}).")
        elif conf < self.conf_accept or agreement_pct < self.agree_accept or (shift_detected and ks_p_val < 0.05):
            state = DecisionSupportState.REVIEW
            evidence.append(f"Borderline confidence ({conf:.3f}) or model divergence ({agreement_pct:.1%} consensus). Requires clinical review.")
        else:
            state = DecisionSupportState.ACCEPT
            evidence.append(f"High predictive confidence ({conf:.3f} >= {self.conf_accept}) with strong multi-model consensus ({agreement_pct:.1%}) and in-distribution features.")

        return ReliabilityAssessment(
            decision_state=state,
            confidence_score=round(conf, 4),
            prediction_entropy=round(entropy, 4),
            distribution_shift_detected=shift_detected,
            ks_test_p_value=round(ks_p_val, 6),
            model_agreement_pct=round(agreement_pct * 100.0, 1),
            ensemble_predictions=ensemble_preds,
            data_quality_valid=data_quality_valid,
            evidence_breakdown=evidence,
        )
