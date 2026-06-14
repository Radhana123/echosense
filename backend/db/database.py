# database.py — Database Connection Manager
# engine = actual DB connection
# SessionLocal = ek transaction (request ke start se end tak)

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .models import Base

# .env se DATABASE_URL lo
from dotenv import load_dotenv
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./echosense.db")

# SQLite ke liye extra setting chahiye
connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args["check_same_thread"] = False

# Engine banao
engine = create_engine(DATABASE_URL, connect_args=connect_args)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Saari tables create karo agar exist nahi karti."""
    Base.metadata.create_all(bind=engine)
    print("✓ Database tables created")


def get_db():
    """FastAPI dependency — har request ko fresh DB session milti hai."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()