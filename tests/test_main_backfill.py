import pandas as pd

from main_backfill import write_rows


def test_write_rows_returns_int_for_empty_df():
    assert write_rows(pd.DataFrame()) == 0
