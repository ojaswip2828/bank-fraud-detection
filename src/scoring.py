"""Score ONE new transaction in real time, with the exact SQL used in training.

No re-implementation of features in Python (a classic source of train/serve skew):
a TEMP table named `transactions` shadows the real one for this connection only,
holding just this customer's history plus the new transaction. sql/features.sql
then runs unchanged and we keep the new row. ~100 rows instead of 220K -> milliseconds.
"""
import sqlite3

import pandas as pd

import explain
from db import DB_PATH, SQL_DIR
from features import add_derived

FEATURES_SQL = (SQL_DIR / "features.sql").read_text()


class UnknownEntity(ValueError):
    pass


def connect_readonly(path=DB_PATH):
    """Open one per request: the shadow TEMP table is private to its connection."""
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def features_for(con, txn):
    """txn: dict with customer_id, merchant_id, amount, txn_time, channel, txn_city."""
    if not con.execute("SELECT 1 FROM customers WHERE customer_id = ?",
                       (txn["customer_id"],)).fetchone():
        raise UnknownEntity(f"unknown customer_id {txn['customer_id']}")
    if not con.execute("SELECT 1 FROM merchants WHERE merchant_id = ?",
                       (txn["merchant_id"],)).fetchone():
        raise UnknownEntity(f"unknown merchant_id {txn['merchant_id']}")

    ts = pd.Timestamp(txn["txn_time"]).strftime("%Y-%m-%d %H:%M:%S")   # stored format
    new_id = con.execute("SELECT COALESCE(MAX(txn_id), 0) + 1 FROM main.transactions").fetchone()[0]
    try:
        con.execute("CREATE TEMP TABLE transactions AS "
                    "SELECT * FROM main.transactions WHERE customer_id = ? AND txn_time <= ?",
                    (txn["customer_id"], ts))
        # is_fraud is a placeholder: the label is never a feature (see schema.sql)
        con.execute("INSERT INTO temp.transactions VALUES (?, ?, ?, ?, ?, ?, ?, 0, NULL)",
                    (new_id, txn["customer_id"], txn["merchant_id"], ts, float(txn["amount"]),
                     txn["channel"], txn["txn_city"]))
        df = pd.read_sql_query(FEATURES_SQL, con)
    finally:
        con.execute("DROP TABLE IF EXISTS temp.transactions")
    return add_derived(df[df.txn_id == new_id].copy())


def score(con, bundle, txn, k=3):
    row = features_for(con, txn)
    X = row[bundle["features"]]
    proba = float(bundle["pipeline"].predict_proba(X)[0, 1])
    sv = explain.shap_frame(bundle, X)
    reasons = sv.iloc[0].sort_values(ascending=False)
    top = [{"feature": f, "value": _plain(X.iloc[0][f]), "contribution": round(float(c), 4)}
           for f, c in reasons.items() if c > 0][:k]
    return {
        "fraud_probability": round(proba, 4),
        "flagged": proba >= bundle["threshold"],
        "threshold": round(bundle["threshold"], 4),
        "reasons": top,
        "features": {c: _plain(X.iloc[0][c]) for c in bundle["features"]},
    }


def _plain(v):
    v = v.item() if hasattr(v, "item") else v
    return round(v, 3) if isinstance(v, float) else v
