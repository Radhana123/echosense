from datetime import datetime
from typing import Optional
from sqlalchemy.orm import Session
from . import models


def create_dataset(db: Session, filename: str, filepath: str,
                   row_count: int, col_count: int, file_size: int):
    dataset = models.Dataset(
        filename=filename, filepath=filepath,
        row_count=row_count, col_count=col_count,
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return dataset


def get_dataset(db: Session, dataset_id: int):
    return db.query(models.Dataset).filter(models.Dataset.id == dataset_id).first()


def get_all_datasets(db: Session, skip: int = 0, limit: int = 20):
    return db.query(models.Dataset).order_by(
        models.Dataset.uploaded_at.desc()
    ).offset(skip).limit(limit).all()


def update_dataset_domain(db: Session, dataset_id: int, domain: str, confidence: float):
    dataset = get_dataset(db, dataset_id)
    if dataset:
        dataset.domain = domain
        dataset.domain_confidence = confidence
        db.commit()
        db.refresh(dataset)
    return dataset


def save_column_profiles(db: Session, dataset_id: int, profiles: list):
    saved = []
    for p in profiles:
        profile = models.ColumnProfile(
            dataset_id=dataset_id,
            column_name=p["column_name"],
            inferred_type=p.get("inferred_type"),
            semantic_type=p.get("semantic_type"),
            null_percent=p.get("null_percent", 0.0),
            unique_count=p.get("unique_count"),
            sample_values=p.get("sample_values", []),
            stats=p.get("stats", {}),
        )
        db.add(profile)
        saved.append(profile)
    db.commit()
    return saved


def get_column_profiles(db: Session, dataset_id: int):
    return db.query(models.ColumnProfile).filter(
        models.ColumnProfile.dataset_id == dataset_id
    ).all()


def save_anomalies(db: Session, dataset_id: int, anomalies: list):
    saved = []
    for a in anomalies:
        anomaly = models.Anomaly(
            dataset_id=dataset_id,
            column_name=a.get("column_name"),
            anomaly_type=a.get("anomaly_type"),
            detection_method=a.get("detection_method"),
            row_indices=a.get("row_indices", []),
            anomaly_score=a.get("anomaly_score"),
            p_value=a.get("p_value"),
            description=a.get("description"),
            is_significant=a.get("p_value", 1.0) < 0.05,
        )
        db.add(anomaly)
        saved.append(anomaly)
    db.commit()
    return saved


def get_anomalies(db: Session, dataset_id: int):
    return db.query(models.Anomaly).filter(
        models.Anomaly.dataset_id == dataset_id,
        models.Anomaly.is_significant == True
    ).all()


def save_causal_edges(db: Session, dataset_id: int, edges: list):
    saved = []
    for e in edges:
        edge = models.CausalEdge(
            dataset_id=dataset_id,
            cause_variable=e["cause"],
            effect_variable=e["effect"],
            causal_strength=e.get("strength"),
            confidence=e.get("confidence"),
            p_value=e.get("p_value"),
            is_spurious=e.get("is_spurious", False),
        )
        db.add(edge)
        saved.append(edge)
    db.commit()
    return saved


def get_causal_edges(db: Session, dataset_id: int):
    return db.query(models.CausalEdge).filter(
        models.CausalEdge.dataset_id == dataset_id,
        models.CausalEdge.is_spurious == False
    ).all()


def create_analysis_run(db: Session, dataset_id: int):
    run = models.AnalysisRun(dataset_id=dataset_id, status="running")
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def complete_analysis_run(db: Session, run_id: int, anomaly_count: int,
                           edges_count: int, domain: str,
                           report_text: str, duration: float):
    run = db.query(models.AnalysisRun).filter(models.AnalysisRun.id == run_id).first()
    if run:
        run.status = "completed"
        run.anomaly_count = anomaly_count
        run.causal_edges_found = edges_count
        run.report_text = report_text
        run.duration_seconds = duration
        run.completed_at = datetime.utcnow()
        db.commit()
        db.refresh(run)
    return run


def get_analysis_history(db: Session, limit: int = 10):
    return db.query(models.AnalysisRun).order_by(
        models.AnalysisRun.started_at.desc()
    ).limit(limit).all()