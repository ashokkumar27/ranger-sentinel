from __future__ import annotations

import datetime as dt

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, declarative_base, mapped_column

Base = declarative_base()


class DailyDecision(Base):
    __tablename__ = "daily_decisions"

    date: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    alpha_gate_pass: Mapped[bool] = mapped_column(Boolean, nullable=False)
    gate_fail_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    base_weight: Mapped[float] = mapped_column(Float, nullable=False)
    alpha_weight: Mapped[float] = mapped_column(Float, nullable=False)
    reserve_weight: Mapped[float] = mapped_column(Float, nullable=False)
    expected_net_apy: Mapped[float] = mapped_column(Float, nullable=False)
    persistence_score: Mapped[float] = mapped_column(Float, nullable=False)
    exit_quality_score: Mapped[float] = mapped_column(Float, nullable=False)
    funding_quality_score: Mapped[float] = mapped_column(Float, nullable=False)
    conviction_score: Mapped[float] = mapped_column(Float, nullable=False)
    edge_after_cost: Mapped[float] = mapped_column(Float, nullable=False)
    stress_flags: Mapped[int] = mapped_column(Integer, nullable=False)
    policy_failures: Mapped[int] = mapped_column(Integer, nullable=False)


class DailyAllocation(Base):
    __tablename__ = "daily_allocations"

    date: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    protocol: Mapped[str] = mapped_column(String, primary_key=True)
    venue: Mapped[str] = mapped_column(String, nullable=False)
    market: Mapped[str] = mapped_column(String, primary_key=True)
    asset: Mapped[str] = mapped_column(String, primary_key=True)
    strategy_type: Mapped[str] = mapped_column(String, nullable=False)
    decision_bucket: Mapped[str] = mapped_column(String, nullable=False)
    target_weight: Mapped[float] = mapped_column(Float, nullable=False)
    expected_net_apy: Mapped[float] = mapped_column(Float, nullable=False)
    conviction_score: Mapped[float] = mapped_column(Float, nullable=False)
    policy_pass: Mapped[bool] = mapped_column(Boolean, nullable=False)
    policy_fail_reasons: Mapped[str | None] = mapped_column(Text, nullable=True)
