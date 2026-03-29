from sqlalchemy.orm import declarative_base, Mapped, mapped_column
from sqlalchemy import String, Float, DateTime, JSON, Text
import datetime as dt

Base = declarative_base()

class ProtocolSnapshot(Base):
    __tablename__ = "protocol_snapshots"

    ts: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    protocol: Mapped[str] = mapped_column(String, primary_key=True)
    venue: Mapped[str | None] = mapped_column(String, nullable=True)
    market: Mapped[str | None] = mapped_column(String, primary_key=True)
    asset: Mapped[str | None] = mapped_column(String, primary_key=True)
    deposit_apy: Mapped[float | None] = mapped_column(Float, nullable=True)
    borrow_apy: Mapped[float | None] = mapped_column(Float, nullable=True)
    funding_rate_daily: Mapped[float | None] = mapped_column(Float, nullable=True)
    utilization: Mapped[float | None] = mapped_column(Float, nullable=True)
    available_liquidity_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    price_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    raw_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)

class VaultReplay(Base):
    __tablename__ = "vault_replay"

    ts: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    policy_version: Mapped[str] = mapped_column(Text, primary_key=True)
    nav_start: Mapped[float] = mapped_column(Float)
    nav_end: Mapped[float] = mapped_column(Float)
    base_weight: Mapped[float] = mapped_column(Float)
    carry_weight: Mapped[float] = mapped_column(Float)
    reserve_weight: Mapped[float] = mapped_column(Float)
    rebalance_cost_usd: Mapped[float] = mapped_column(Float)
    gross_return: Mapped[float] = mapped_column(Float)
    net_return: Mapped[float] = mapped_column(Float)
