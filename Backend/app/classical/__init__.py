"""Classical machine learning baseline models and training pipeline for Q-BioForge."""

from app.classical.base import ClassicalModel
from app.classical.logistic_regression import LogisticRegressionModel
from app.classical.random_forest import RandomForestModel
from app.classical.svm import SVMModel
from app.classical.xgboost_model import XGBoostModel
from app.classical.metrics import (
    ClassificationMetrics,
    BenchmarkRecord,
    calculate_metrics,
)
from app.classical.trainer import (
    train_and_evaluate_classical_model,
    list_all_experiment_results,
    get_experiment_result,
    generate_classical_benchmark_table,
    MODEL_REGISTRY,
)

__all__ = [
    "ClassicalModel",
    "LogisticRegressionModel",
    "RandomForestModel",
    "SVMModel",
    "XGBoostModel",
    "ClassificationMetrics",
    "BenchmarkRecord",
    "calculate_metrics",
    "train_and_evaluate_classical_model",
    "list_all_experiment_results",
    "get_experiment_result",
    "generate_classical_benchmark_table",
    "MODEL_REGISTRY",
]
