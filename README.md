# Ranger Sentinel

Ranger Sentinel is a fuller local implementation for a historical + live strategy engine for an adaptive carry vault.

## What it does
- pulls historical and live data from Drift, Jupiter Lend, and Kamino
- normalizes everything into one feature table
- runs the same policy in historical replay and live recommendation mode
- compares adaptive vs static baseline
- visualizes results in a dashboard
- uses SQLite by default for simple local setup

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env`:
- `DATABASE_URL=sqlite:///ranger_sentinel.db`
- `JUPITER_API_KEY`
- `KAMINO_MARKET_PUBKEY`
- `KAMINO_RESERVE_PUBKEY`

## Run order

Backfill snapshots:
```bash
python main_backfill.py
```

Run simulation and save parquet outputs:
```bash
python main_simulate.py
```

Launch dashboard:
```bash
streamlit run dashboard/app.py
```

Get current live recommendation:
```bash
python main_live.py
```

Run tests:
```bash
pytest -q
```

## What proves the strategy worked
Use these comparisons:
- adaptive policy vs static baseline
- annualized return
- max drawdown
- rolling 90-day median and worst outcomes
- turnover proxy after rebalance costs

## Practical notes
- Drift should be the easiest source to validate first.
- Jupiter requires an API key.
- Kamino requires valid market/reserve IDs.
- The only likely first-run fixes are field mapping tweaks if an external response schema changed.
