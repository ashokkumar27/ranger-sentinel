from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from storage.schemas import Base
from configs.settings import settings

engine = create_engine(
    settings.database_url,
    future=True,
    connect_args={"check_same_thread": False} if settings.database_url.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

def init_db() -> None:
    Base.metadata.create_all(bind=engine)
