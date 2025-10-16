import os
import types

import pandas as pd

import importlib


def test_fetch_historical_data_demo_mode():
    # Ensure environment minimal
    os.environ.pop("POLYGON_API_KEY", None)
    # Import the module from scripts
    mod = importlib.import_module("scripts.data_infrastructure")
    env = mod.get_env()

    df = mod.fetch_historical_data("AAPL", env, use_demo=True)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert set(["date","open","high","low","close","volume","symbol"]).issubset(df.columns)
    assert (df["symbol"] == "AAPL").all()


