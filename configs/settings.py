from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()

class Settings(BaseModel):
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///ranger_sentinel.db")
    raw_dir: str = os.getenv("RAW_DIR", "data/raw")
    parquet_dir: str = os.getenv("PARQUET_DIR", "data/parquet")
    jupiter_api_key: str = os.getenv("JUPITER_API_KEY", "")
    drift_base_url: str = os.getenv("DRIFT_BASE_URL", "https://data.api.drift.trade")
    kamino_market_pubkey: str = os.getenv("KAMINO_MARKET_PUBKEY", "YOUR_MARKET_PUBKEY")
    kamino_reserve_pubkey: str = os.getenv("KAMINO_RESERVE_PUBKEY", "YOUR_RESERVE_PUBKEY")

settings = Settings()
