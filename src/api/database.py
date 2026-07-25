"""
database.py

Defines the database table where every prediction gets logged, using
SQLAlchemy (an ORM — lets us work with Python classes instead of writing
raw SQL). Starts with SQLite (a single file, zero setup) since that's
plenty for this project's scale.
"""

from datetime import datetime, timezone

from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = "sqlite:///./wildsense.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class Prediction(Base):
    """
    One row per prediction made through the API. This table is what
    Phase 4's trend analysis will aggregate from later — every column
    here exists because some downstream feature needs it.
    """
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    image_filename = Column(String, nullable=False)
    species = Column(String, nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    needs_review = Column(Boolean, default=False)  # confidence-based triage flag

    # Optional location metadata — the client (dashboard, script, etc.)
    # can supply these if known; otherwise they stay null.
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


def init_db():
    """Creates the predictions table if it doesn't already exist."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """
    Yields a database session, and guarantees it gets closed afterward
    even if an error happens. FastAPI calls this automatically per-request
    when used as a dependency (see main.py).
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()