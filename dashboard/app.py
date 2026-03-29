import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st
import pandas as pd
import plotly.express as px
from engine.score import summarize_run

st.set_page_config(page_title="Ranger Sentinel", layout="wide")
st.title("Ranger Sentinel — Historical + Live Proof")

adaptive = pd.read_parquet("data/parquet/adaptive_replay.parquet")
static = pd.read_parquet("data/parquet/static_replay.parquet")
features = pd.read_parquet("data/parquet/features.parquet")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Adaptive summary")
    st.json(summarize_run(adaptive))
with col2:
    st.subheader("Static summary")
    st.json(summarize_run(static))

nav_df = pd.concat([
    adaptive[["ts", "nav_end"]].assign(policy="adaptive"),
    static[["ts", "nav_end"]].assign(policy="static"),
], ignore_index=True)

st.subheader("NAV over time")
st.plotly_chart(px.line(nav_df, x="ts", y="nav_end", color="policy"), use_container_width=True)

weights = adaptive[["ts", "base_weight", "carry_weight", "reserve_weight"]].melt(
    id_vars="ts", var_name="sleeve", value_name="weight"
)
st.subheader("Adaptive sleeve allocation")
st.plotly_chart(px.area(weights, x="ts", y="weight", color="sleeve"), use_container_width=True)

adaptive_roll = adaptive[["ts", "nav_end"]].copy()
adaptive_roll["rolling_90d_return"] = adaptive_roll["nav_end"] / adaptive_roll["nav_end"].shift(90) - 1
st.subheader("Adaptive rolling 90-day return")
st.plotly_chart(px.line(adaptive_roll, x="ts", y="rolling_90d_return"), use_container_width=True)

st.subheader("Latest live features")
st.dataframe(features.tail(10), use_container_width=True)
