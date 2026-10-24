import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import db  # noqa: E402
from generate_data import generate  # noqa: E402
from features import load_features, time_split  # noqa: E402
from train import threshold_for_recall  # noqa: E402


@pytest.fixture(scope="module")
def con(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    cu, me, tx = generate(n_customers=150, n_merchants=60, days=20, fraud_cases=20, seed=1)
    for name, df in [("customers", cu), ("merchants", me), ("transactions", tx)]:
        df.to_csv(d / f"{name}.csv", index=False)
    return db.build_db(data_dir=d, path=d / "test.db")


def test_data_has_both_classes(con):
    n, f = con.execute("SELECT COUNT(*), SUM(is_fraud) FROM transactions").fetchone()
    assert n > 1000 and 0 < f < n * 0.05


def test_schema_constraints_reject_bad_rows(con):
    with pytest.raises(sqlite3.IntegrityError):
        con.execute("INSERT INTO transactions VALUES (999999,1,1,'2026-01-01',-5,'POS','Pune',0,'none')")
    with pytest.raises(sqlite3.IntegrityError):  # foreign key
        con.execute("INSERT INTO transactions VALUES (999998,99999,1,'2026-01-01',5,'POS','Pune',0,'none')")


def test_all_analytics_queries_run(con):
    for name, sql in db.named_queries("analytics.sql").items():
        assert len(db.query(con, sql)) > 0, name


def test_features_no_nulls_and_one_row_per_txn(con):
    df = load_features(con)
    n = con.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    assert len(df) == n and df.txn_id.is_unique
    assert not df.drop(columns=["txn_time"]).isna().any().any()


def test_first_txn_has_no_history(con):
    """Leakage check: a customer's first txn must not see any history."""
    df = load_features(con).sort_values("txn_time")
    first = df.groupby("customer_id").head(1)
    assert (first.hist_txn_count == 0).all()
    assert (first.amt_to_hist_avg == 1.0).all()


def test_time_split_is_chronological(con):
    tr, te = time_split(load_features(con))
    assert tr.txn_time.max() <= te.txn_time.min()


def test_threshold_hits_target_recall():
    y = np.array([0] * 90 + [1] * 10)
    p = np.linspace(0, 1, 100)
    thr = threshold_for_recall(y, p, 0.9)
    assert ((p >= thr) & (y == 1)).sum() / 10 >= 0.9
