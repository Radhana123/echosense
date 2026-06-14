# models.py — Database Tables ka Blueprint
# Har class = ek SQL table
# SQLAlchemy automatically SQL banata hai Python class se

from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, Boolean,
    DateTime, Text, ForeignKey, JSON
)
from sqlalchemy.orm import declarative_base, relationship

# Base class — sabhi tables isse inherit karenge
Base = declarative_base()


class Dataset(Base):
    """Table 1: Uploaded CSV files ka record."""
    __tablename__ = "datasets"

    id                = Column(Integer, primary_key=True)
    filename          = Column(String(255))
    filepath          = Column(String(500))
    row_count         = Column(Integer)
    col_count         = Column(Integer)
    domain            = Column(String(100))   # "logistics", "healthcare"
    domain_confidence = Column(Float)
    uploaded_at       = Column(DateTime, default=datetime.utcnow)

    # Relationships — ek dataset ke multiple profiles ho sakte hain
    column_profiles = relationship("ColumnProfile", back_populates="dataset", cascade="all, delete")
    anomalies       = relationship("Anomaly",       back_populates="dataset", cascade="all, delete")
    analysis_runs   = relationship("AnalysisRun",   back_populates="dataset", cascade="all, delete")


class ColumnProfile(Base):
    """Table 2: NLP profiler ke results — har column ki info."""
    __tablename__ = "column_profiles"

    id            = Column(Integer, primary_key=True)
    dataset_id    = Column(Integer, ForeignKey("datasets.id"))
    column_name   = Column(String(255))
    inferred_type = Column(String(50))    # numeric, categorical, date, binary
    semantic_type = Column(String(100))   # revenue, target_variable, etc
    null_percent  = Column(Float)
    unique_count  = Column(Integer)
    sample_values = Column(JSON)          # first 5 values
    stats         = Column(JSON)          # mean, std, min, max
    dataset       = relationship("Dataset", back_populates="column_profiles")


class Anomaly(Base):
    """Table 3: ML anomaly detector ke findings."""
    __tablename__    = "anomalies"

    id               = Column(Integer, primary_key=True)
    dataset_id       = Column(Integer, ForeignKey("datasets.id"))
    column_name      = Column(String(255))
    anomaly_type     = Column(String(50))    # outlier, spike, missing_pattern
    detection_method = Column(String(50))    # isolation_forest, zscore, iqr
    row_indices      = Column(JSON)
    anomaly_score    = Column(Float)
    p_value          = Column(Float)
    description      = Column(Text)
    is_significant   = Column(Boolean, default=False)
    dataset          = relationship("Dataset", back_populates="anomalies")


class CausalEdge(Base):
    """Table 4: DoWhy causal graph ke edges.
    Example: rain -> delivery_time (strength=45.2, confidence=0.97)"""
    __tablename__   = "causal_edges"

    id              = Column(Integer, primary_key=True)
    dataset_id      = Column(Integer, ForeignKey("datasets.id"))
    cause_variable  = Column(String(255))   # "rain"
    effect_variable = Column(String(255))   # "delivery_time"
    causal_strength = Column(Float)         # ATE value
    confidence      = Column(Float)
    p_value         = Column(Float)
    is_spurious     = Column(Boolean, default=False)


class AnalysisRun(Base):
    """Table 5: Har complete analysis run ka summary."""
    __tablename__      = "analysis_runs"

    id                 = Column(Integer, primary_key=True)
    dataset_id         = Column(Integer, ForeignKey("datasets.id"))
    status             = Column(String(50), default="pending")
    anomaly_count      = Column(Integer, default=0)
    causal_edges_found = Column(Integer, default=0)
    report_text        = Column(Text)
    duration_seconds   = Column(Float)
    started_at         = Column(DateTime, default=datetime.utcnow)
    completed_at       = Column(DateTime)
    dataset            = relationship("Dataset", back_populates="analysis_runs")