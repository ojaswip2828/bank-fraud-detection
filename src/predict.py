"""Score transactions with a trained model and explain each alert with SHAP.

Usage:
  python src/predict.py --txn-id 1234            # score + explain one stored transaction
  python src/predict.py --top 20                 # 20 riskiest transactions (alert queue) + reasons
  python src/predict.py --model logistic_regression --top 10
  python src/predict.py --flagged                # explain every flagged txn -> reports/alerts_explained.csv
"""
import argparse

import joblib

import explain
from db import ROOT
from features import load_features


def score(model="random_forest"):
    b = joblib.load(ROOT / "models" / f"{model}.joblib")
    df = load_features()
    df["fraud_score"] = b["pipeline"].predict_proba(df[b["features"]])[:, 1]
    df["flagged"] = (df["fraud_score"] >= b["threshold"]).astype(int)
    return df, b


def explain_rows(df, bundle, k=3):
    """Adds a 'reasons' column: top-k SHAP contributions toward fraud for each row."""
    X = df[bundle["features"]]
    df = df.copy()
    df["reasons"] = explain.top_reasons(explain.shap_frame(bundle, X), X, k=k)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="random_forest",
                    choices=["random_forest", "logistic_regression"])
    ap.add_argument("--txn-id", type=int)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--flagged", action="store_true",
                    help="explain every flagged transaction and save to reports/alerts_explained.csv")
    a = ap.parse_args()
    df, bundle = score(a.model)
    cols = ["txn_id", "customer_id", "txn_time", "amount", "channel", "category",
            "fraud_score", "flagged", "is_fraud", "reasons"]
    if a.txn_id:
        row = explain_rows(df.loc[df.txn_id == a.txn_id], bundle, k=5)
        print(row[cols].T.to_string())
    elif a.flagged:
        alerts = explain_rows(df[df.flagged == 1], bundle)
        out = ROOT / "reports" / "alerts_explained.csv"
        alerts[cols].to_csv(out, index=False)
        print(f"{len(alerts):,} flagged transactions explained -> {out}")
    else:
        print(f"threshold={bundle['threshold']:.4f}")
        top = explain_rows(df.nlargest(a.top, "fraud_score"), bundle)
        print(top[cols].round(3).to_string(index=False))
