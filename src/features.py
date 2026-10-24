"""Load SQL-engineered features into pandas and define the model input columns."""
import numpy as np

from db import SQL_DIR, connect, query

NUMERIC = ["log_amount", "amt_to_hist_avg", "hist_txn_count", "mins_since_prev",
           "txns_last_1h", "amt_last_24h", "is_new_merchant", "is_out_of_city",
           "is_night", "account_age_days"]
CATEGORICAL = ["channel", "category", "income_band"]
TARGET = "is_fraud"


def load_features(con=None):
    own = con is None
    con = con or connect()
    df = query(con, (SQL_DIR / "features.sql").read_text())
    if own:
        con.close()
    df["log_amount"] = np.log1p(df["amount"])
    df["amt_last_24h"] = np.log1p(df["amt_last_24h"])  # heavy-tailed -> log
    return df


def time_split(df, test_frac=0.25):
    """Train on the past, test on the future (like production). No random shuffle."""
    df = df.sort_values("txn_time")
    cut = int(len(df) * (1 - test_frac))
    return df.iloc[:cut], df.iloc[cut:]
