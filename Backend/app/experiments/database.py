"""Lightweight SQLite database storage for Q-BioForge experiment metadata and registry."""

import os
import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def get_db_path(base_dir: Optional[str] = None) -> str:
    """Return path to SQLite database file."""
    from app.config import settings
    if settings.DATABASE_PATH:
        path = settings.DATABASE_PATH
        if not os.path.isabs(path):
            backend_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            path = os.path.join(backend_root, path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    if base_dir is None:
        backend_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        base_dir = os.path.join(backend_root, "results")
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, "q_bioforge.db")


def init_db(db_path: Optional[str] = None) -> None:
    """Initialize SQLite database tables."""
    path = db_path or get_db_path()
    conn = sqlite3.connect(path)
    cur = conn.cursor()

    # Experiments table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS experiments (
        experiment_id TEXT PRIMARY KEY,
        campaign_id TEXT,
        dataset TEXT NOT NULL,
        model_type TEXT NOT NULL,
        model_category TEXT NOT NULL,
        qubit_count INTEGER,
        circuit_depth INTEGER,
        encoding TEXT,
        noise_model TEXT,
        noise_strength REAL,
        optimizer TEXT,
        learning_rate REAL,
        seed INTEGER NOT NULL,
        compute_backend TEXT NOT NULL,
        status TEXT NOT NULL,
        training_time_seconds REAL,
        inference_time_seconds REAL,
        accuracy REAL,
        f1_score REAL,
        roc_auc REAL,
        sensitivity REAL,
        specificity REAL,
        brier_score REAL,
        config_json TEXT,
        metrics_json TEXT,
        artifact_path TEXT,
        created_at TEXT NOT NULL
    )
    """)

    # Campaigns table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS campaigns (
        campaign_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        description TEXT,
        total_experiments INTEGER DEFAULT 0,
        completed_experiments INTEGER DEFAULT 0,
        failed_experiments INTEGER DEFAULT 0,
        resource_limited_experiments INTEGER DEFAULT 0,
        status TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """)

    # Model Registry table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS model_registry (
        model_id TEXT PRIMARY KEY,
        experiment_id TEXT NOT NULL,
        dataset TEXT NOT NULL,
        model_type TEXT NOT NULL,
        model_category TEXT NOT NULL,
        qubit_count INTEGER,
        circuit_depth INTEGER,
        encoding TEXT,
        checkpoint_path TEXT NOT NULL,
        checkpoint_format TEXT NOT NULL,
        accuracy REAL,
        f1_score REAL,
        roc_auc REAL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
    )
    """)

    # Audit log table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS audit_logs (
        log_id INTEGER PRIMARY KEY AUTOINCREMENT,
        action TEXT NOT NULL,
        experiment_id TEXT,
        details TEXT,
        timestamp TEXT NOT NULL
    )
    """)

    conn.commit()
    conn.close()


def insert_or_update_experiment(exp_data: Dict[str, Any], db_path: Optional[str] = None) -> None:
    """Insert or update experiment record in SQLite database."""
    path = db_path or get_db_path()
    init_db(path)
    conn = sqlite3.connect(path)
    cur = conn.cursor()

    metrics = exp_data.get("metrics", {})
    test_metrics = metrics.get("test_metrics", {})
    config = exp_data.get("config", {})

    cur.execute("""
    INSERT OR REPLACE INTO experiments (
        experiment_id, campaign_id, dataset, model_type, model_category,
        qubit_count, circuit_depth, encoding, noise_model, noise_strength,
        optimizer, learning_rate, seed, compute_backend, status,
        training_time_seconds, inference_time_seconds,
        accuracy, f1_score, roc_auc, sensitivity, specificity, brier_score,
        config_json, metrics_json, artifact_path, created_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        exp_data.get("experiment_id"),
        exp_data.get("campaign_id"),
        config.get("dataset", "breast_cancer_wisconsin"),
        config.get("model", config.get("model_type", "unknown")),
        config.get("model_category", "classical"),
        config.get("qubit_count"),
        config.get("circuit_depth"),
        config.get("encoding"),
        config.get("noise_model", "ideal"),
        config.get("noise_strength", 0.0),
        config.get("optimizer"),
        config.get("learning_rate"),
        config.get("random_seed", 42),
        config.get("compute_backend", "local"),
        config.get("status", "completed"),
        metrics.get("training_time_seconds"),
        metrics.get("inference_time_seconds"),
        test_metrics.get("accuracy"),
        test_metrics.get("f1_score"),
        test_metrics.get("roc_auc"),
        test_metrics.get("sensitivity"),
        test_metrics.get("specificity"),
        test_metrics.get("brier_score"),
        json.dumps(config),
        json.dumps(metrics),
        exp_data.get("experiment_dir"),
        config.get("timestamp", datetime.now(timezone.utc).isoformat()),
    ))

    conn.commit()
    conn.close()


def query_experiments(filter_dict: Optional[Dict[str, Any]] = None, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Query experiments from SQLite database."""
    path = db_path or get_db_path()
    init_db(path)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    query = "SELECT * FROM experiments"
    params = []
    if filter_dict:
        conditions = []
        for k, v in filter_dict.items():
            conditions.append(f"{k} = ?")
            params.append(v)
        query += " WHERE " + " AND ".join(conditions)

    query += " ORDER BY experiment_id DESC"
    cur.execute(query, params)
    rows = cur.fetchall()
    results = [dict(r) for r in rows]
    conn.close()
    return results


def log_audit_event(action: str, experiment_id: Optional[str] = None, details: Optional[str] = None, db_path: Optional[str] = None) -> None:
    """Record an audit trail event."""
    path = db_path or get_db_path()
    init_db(path)
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO audit_logs (action, experiment_id, details, timestamp)
    VALUES (?, ?, ?, ?)
    """, (action, experiment_id, details, datetime.now(timezone.utc).isoformat()))
    conn.commit()
    conn.close()
