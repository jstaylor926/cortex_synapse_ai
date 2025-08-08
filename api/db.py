from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from settings import get_settings

class Base(DeclarativeBase):
    pass

settings = get_settings()
engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

def get_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
