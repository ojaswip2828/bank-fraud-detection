"""Score transactions with a trained model.

Usage:
  python src/predict.py --txn-id 1234            # score one stored transaction
  python src/predict.py --top 20                 # 20 riskiest transactions (alert queue)
  python src/predict.py --model logistic_regression --top 10
"""
import argparse

import joblib

from db import ROOT
from features import load_features


def score(model="random_forest"):
    b = joblib.load(ROOT / "models" / f"{model}.joblib")
    df = load_features()
    df["fraud_score"] = b["pipeline"].predict_proba(df[b["features"]])[:, 1]
    df["flagged"] = (df["fraud_score"] >= b["threshold"]).astype(int)
    return df, b["threshold"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="random_forest",
                    choices=["random_forest", "logistic_regression"])
    ap.add_argument("--txn-id", type=int)
    ap.add_argument("--top", type=int, default=10)
    a = ap.parse_args()
    df, thr = score(a.model)
    cols = ["txn_id", "customer_id", "txn_time", "amount", "channel", "category",
            "amt_to_hist_avg", "txns_last_1h", "is_out_of_city", "fraud_score", "flagged", "is_fraud"]
    if a.txn_id:
        print(df.loc[df.txn_id == a.txn_id, cols].T.to_string())
    else:
        print(f"threshold={thr:.4f}")
        print(df.nlargest(a.top, "fraud_score")[cols].round(3).to_string(index=False))
