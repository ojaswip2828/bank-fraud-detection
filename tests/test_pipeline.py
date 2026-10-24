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


def _bundle(con, model, calibrate=True):
    df = load_features(con)
    tr, te = time_split(df)
    cols = NUMERIC + CATEGORICAL
    pipe = build_models(rf_kwargs={"n_estimators": 20}, calibrate=calibrate, only=[model])[model]
    _, thr, _ = fit_and_evaluate(pipe, tr, te, cols)
    bg = explain._dense(pipe.named_steps["pre"].transform(tr[cols].head(50)))
    return {"pipeline": pipe, "threshold": thr, "features": cols, "background": bg}, te


@pytest.fixture(scope="module")
def lgbm_bundle(con):
    return _bundle(con, "lightgbm")


def test_ablation_without_features_still_trains(con):
    df = load_features(con)
    tr, te = time_split(df)
    num = [c for c in NUMERIC if c != "is_new_merchant"]
    pipe = build_models(num, [], rf_kwargs={"n_estimators": 20})["random_forest"]
    m, _, _ = fit_and_evaluate(pipe, tr, te, num)
    assert 0 <= m["pr_auc"] <= 1
    assert pipe.named_steps["pre"].get_feature_names_out().tolist() == [f"num__{c}" for c in num]


def test_rf_shap_maps_back_to_raw_features_and_sums_to_prediction(con):
    bundle, te = _bundle(con, "random_forest", calibrate=False)
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


def test_lgbm_shap_explains_the_model_under_calibration(lgbm_bundle):
    bundle, te = lgbm_bundle
    X = te[bundle["features"]].head(20)
    sv = explain.shap_frame(bundle, X)
    pipe = bundle["pipeline"]
    base = explain.base_model(pipe.named_steps["clf"])
    assert base is not pipe.named_steps["clf"]            # really unwrapped the calibrator
    Xt = explain._dense(pipe.named_steps["pre"].transform(X))
    bias = base.predict(Xt, pred_contrib=True, raw_score=True)[:, -1]
    assert np.allclose(bias + sv.sum(axis=1), base.predict(Xt, raw_score=True), atol=1e-6)
    # calibrated output is a valid probability
    p = pipe.predict_proba(X)[:, 1]
    assert ((p >= 0) & (p <= 1)).all()


def _db_path(con):
    return con.execute("PRAGMA database_list").fetchone()[2]


def test_serving_features_match_training_features(con, tmp_path):
    """No train/serve skew: scoring a txn live must give the features training saw."""
    import shutil
    import scoring
    train_feats = load_features(con)
    # a customer's last txn, with no other txn at the same timestamp
    last = train_feats.sort_values(["txn_time", "txn_id"]).groupby("customer_id").tail(1)
    tx = con.execute("SELECT * FROM transactions WHERE txn_id = ?",
                     (int(last.iloc[len(last) // 2].txn_id),)).fetchone()
    copy = tmp_path / "serve.db"
    shutil.copy(_db_path(con), copy)
    c2 = sqlite3.connect(copy)
    c2.execute("DELETE FROM transactions WHERE txn_id = ?", (tx[0],))
    c2.commit()
    c2.close()
    ro = scoring.connect_readonly(copy)
    txn = dict(zip(["customer_id", "merchant_id", "txn_time", "amount", "channel", "txn_city"],
                   tx[1:7]))
    live = scoring.features_for(ro, txn).iloc[0]
    ro.close()
    expected = train_feats.loc[train_feats.txn_id == tx[0]].iloc[0]
    for col in NUMERIC:
        assert live[col] == pytest.approx(expected[col]), col
    for col in CATEGORICAL:
        assert live[col] == expected[col], col


def test_api_scores_and_validates(con, lgbm_bundle, tmp_path, monkeypatch):
    import importlib
    import joblib
    from fastapi.testclient import TestClient
    joblib.dump(lgbm_bundle[0], tmp_path / "lightgbm.joblib")
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("DB_PATH", _db_path(con))
    import app as app_module
    client = TestClient(importlib.reload(app_module).app)

    assert client.get("/health").json()["status"] == "ok"
    body = {"customer_id": 1, "merchant_id": 1, "amount": 42.5, "txn_time": "2026-01-15T03:10:00",
            "channel": "ONLINE", "txn_city": "Pune"}
    r = client.post("/score", json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    assert 0 <= out["fraud_probability"] <= 1 and isinstance(out["flagged"], bool)
    assert set(out["features"]) == set(NUMERIC + CATEGORICAL)
    assert client.post("/score", json={**body, "customer_id": 999999}).status_code == 404
    assert client.post("/score", json={**body, "channel": "CHEQUE"}).status_code == 422


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
