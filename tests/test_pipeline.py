import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import db  # noqa: E402
import explain  # noqa: E402
import paysim  # noqa: E402
from generate_data import generate  # noqa: E402
from features import CATEGORICAL, NUMERIC, load_features, time_split  # noqa: E402
from train import build_models, fit_and_evaluate, threshold_for_recall  # noqa: E402


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


def test_fraud_is_not_confined_to_a_few_categories(con):
    """Regression guard: category alone must not be a perfect separator."""
    n_cats = con.execute("""SELECT COUNT(DISTINCT m.category) FROM transactions t
                            JOIN merchants m ON m.merchant_id = t.merchant_id
                            WHERE t.is_fraud = 1""").fetchone()[0]
    assert n_cats >= 5


@pytest.fixture(scope="module")
def small_bundle(con):
    df = load_features(con)
    tr, te = time_split(df)
    cols = NUMERIC + CATEGORICAL
    pipe = build_models(rf_kwargs={"n_estimators": 20})["random_forest"]
    _, thr, _ = fit_and_evaluate(pipe, tr, te, cols)
    bg = explain._dense(pipe.named_steps["pre"].transform(tr[cols].head(50)))
    return {"pipeline": pipe, "threshold": thr, "features": cols, "background": bg}, te


def test_ablation_without_features_still_trains(con):
    df = load_features(con)
    tr, te = time_split(df)
    num = [c for c in NUMERIC if c != "is_new_merchant"]
    pipe = build_models(num, [], rf_kwargs={"n_estimators": 20})["random_forest"]
    m, _, _ = fit_and_evaluate(pipe, tr, te, num)
    assert 0 <= m["pr_auc"] <= 1
    assert pipe.named_steps["pre"].get_feature_names_out().tolist() == [f"num__{c}" for c in num]


def test_shap_maps_back_to_raw_features_and_sums_to_prediction(small_bundle):
    bundle, te = small_bundle
    X = te[bundle["features"]].head(20)
    sv = explain.shap_frame(bundle, X)
    assert list(sv.columns) == sorted(bundle["features"])
    # additivity: base value + sum of SHAP = predicted fraud probability
    clf = bundle["pipeline"].named_steps["clf"]
    import shap
    base = np.ravel(shap.TreeExplainer(clf).expected_value)[-1]
    proba = bundle["pipeline"].predict_proba(X)[:, 1]
    assert np.allclose(base + sv.sum(axis=1), proba, atol=1e-6)
    reasons = explain.top_reasons(sv, X)
    assert len(reasons) == len(X)


def test_paysim_features_use_only_past_rows():
    raw = pd.DataFrame({
        "step": [1, 1, 2, 3, 3], "type": ["TRANSFER", "PAYMENT", "CASH_OUT", "TRANSFER", "CASH_OUT"],
        "amount": [100.0, 5.0, 300.0, 50.0, 1000.0],
        "nameOrig": ["C1", "C2", "C3", "C4", "C5"],
        "oldbalanceOrg": [100.0, 10.0, 0.0, 500.0, 2000.0],
        "newbalanceOrig": [0.0, 5.0, 0.0, 450.0, 1000.0],
        "nameDest": ["D1", "M1", "D1", "D2", "D1"],
        "oldbalanceDest": [0.0, 0.0, 100.0, 0.0, 400.0],
        "newbalanceDest": [100.0, 0.0, 400.0, 50.0, 1400.0],
        "isFraud": [1, 0, 0, 0, 1], "isFlaggedFraud": [0, 0, 0, 0, 0],
    })
    f = paysim.build_features(raw)
    assert "PAYMENT" not in set(f["type"])                     # only TRANSFER/CASH_OUT kept
    d1 = f[f.nameDest == "D1"].sort_values("step")
    assert d1["dest_hist_count"].tolist() == [0, 1, 2]
    assert d1["amt_to_dest_hist_avg"].tolist() == [1.0, 3.0, 1000 / 200]   # past avg only
    assert d1["dest_hours_since_prev"].tolist() == [1000, 1, 1]
    assert f.loc[f.nameOrig == "C1", "drains_orig"].item() == 1
    assert not {"isFlaggedFraud", "newbalanceOrig", "newbalanceDest"} & set(paysim.NUMERIC)
