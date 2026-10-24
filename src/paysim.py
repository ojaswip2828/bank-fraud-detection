"""Run the same modelling approach on PaySim, a public mobile-money fraud dataset.

Dataset: https://www.kaggle.com/datasets/ealaxi/paysim1 (6.36M rows, ~0.13% fraud).
PaySim is itself simulated, but calibrated on a month of real mobile-money logs
from an African operator, so its fraud patterns were not written by this project.

Usage:
  python src/paysim.py --csv data/paysim/PS_20174392719_1491204439457_log.csv

Honesty rules (same spirit as sql/features.sql):
- Only TRANSFER and CASH_OUT rows: PaySim's fraud occurs only in those two types.
- Never use `isFlaggedFraud` (it's PaySim's own rule output) or `newbalance*` /
  "balance error" columns. Those are post-transaction simulator artifacts that
  make the problem trivially separable and wouldn't be clean at authorisation time.
- History features for a destination account use only its *earlier* rows.
- Time split on `step` (hours): train on the past, test on the future.
- To keep RF training to a few minutes, legit rows used for *fitting* are
  downsampled. Threshold tuning and test metrics always use the full class mix.
"""
import argparse
import json

import numpy as np
import pandas as pd

from db import ROOT
from train import build_models, fit_and_evaluate

TARGET = "isFraud"
NUMERIC = ["log_amount", "amt_to_orig_balance", "drains_orig", "orig_balance_zero",
           "log_old_balance_dest", "dest_hist_count", "amt_to_dest_hist_avg",
           "dest_hours_since_prev", "dest_txns_same_hour", "hour"]
CATEGORICAL = ["type"]
REPORT = ROOT / "reports" / "paysim_report.md"


def build_features(df):
    df = df[df["type"].isin(["TRANSFER", "CASH_OUT"])].copy()
    df["row"] = np.arange(len(df))                 # file order breaks ties within a step
    df = df.sort_values(["step", "row"], kind="stable")
    df["log_amount"] = np.log1p(df["amount"])
    df["amt_to_orig_balance"] = (df["amount"] / (df["oldbalanceOrg"] + 1)).clip(upper=100)
    df["drains_orig"] = ((df["oldbalanceOrg"] > 0) & (df["amount"] >= df["oldbalanceOrg"])).astype(int)
    df["orig_balance_zero"] = (df["oldbalanceOrg"] == 0).astype(int)
    df["log_old_balance_dest"] = np.log1p(df["oldbalanceDest"])
    df["hour"] = df["step"] % 24
    g = df.groupby("nameDest", sort=False)
    df["dest_hist_count"] = g.cumcount()
    prev_sum = g["amount"].cumsum() - df["amount"]
    hist_avg = prev_sum / df["dest_hist_count"].replace(0, np.nan)
    df["amt_to_dest_hist_avg"] = (df["amount"] / hist_avg).fillna(1.0).clip(upper=100)
    df["dest_hours_since_prev"] = (df["step"] - g["step"].shift()).fillna(1000)
    df["dest_txns_same_hour"] = df.groupby(["nameDest", "step"], sort=False).cumcount()
    return df.reset_index(drop=True)


def step_split(df, test_frac=0.25):
    df = df.sort_values(["step", "row"])
    cut = int(len(df) * (1 - test_frac))
    return df.iloc[:cut], df.iloc[cut:]


def main(csv, legit_frac=0.05, n_estimators=150, seed=42):
    raw = pd.read_csv(csv)
    df = build_features(raw)
    # fit_and_evaluate's validation split sorts on txn_time; step is PaySim's clock
    df["txn_time"] = df["step"] * 1_000_000 + df["row"]
    train, test = step_split(df)
    rng = np.random.default_rng(seed)

    def fit_rows(d):
        keep = (d[TARGET] == 1) | (rng.random(len(d)) < legit_frac)
        return d[keep]

    cols = NUMERIC + CATEGORICAL
    results = {}
    print(f"PaySim TRANSFER+CASH_OUT: {len(df):,} rows, fraud {df[TARGET].mean():.3%} | "
          f"train {len(train):,} / test {len(test):,} (fraud {int(test[TARGET].sum())})")
    for name, pipe in build_models(NUMERIC, CATEGORICAL,
                                   rf_kwargs={"n_estimators": n_estimators}).items():
        m, _, _ = fit_and_evaluate(pipe, train, test, cols, target=TARGET, fit_rows=fit_rows)
        results[name] = m
        print(f"{name}: {json.dumps({k: v for k, v in m.items() if k != 'at_default_0.5'})}")

    md = ["# PaySim results", "",
          f"Rows: {len(df):,} TRANSFER/CASH_OUT txns, fraud rate {df[TARGET].mean():.3%}. "
          f"Split on `step` (train first 75%, test last 25%: {len(test):,} txns, "
          f"{int(test[TARGET].sum())} fraud). Threshold tuned on validation for recall >= 0.9.", "",
          "| model | ROC-AUC | PR-AUC | precision | recall | FP | FN |",
          "|---|---|---|---|---|---|---|"]
    for n, m in results.items():
        c = m["confusion"]
        md.append(f"| {n} | {m['roc_auc']} | {m['pr_auc']} | {m['precision']} | {m['recall']} "
                  f"| {c['fp']} | {c['fn']} |")
    md += ["", "Excluded on purpose: `isFlaggedFraud`, `newbalanceOrig`, `newbalanceDest`, "
               "balance-error features (post-transaction simulator artifacts)."]
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text("\n".join(md) + "\n", encoding="utf-8")
    (ROOT / "reports" / "paysim_metrics.json").write_text(json.dumps(results, indent=2))
    print(f"-> {REPORT}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="path to the PaySim CSV from Kaggle")
    ap.add_argument("--legit-frac", type=float, default=0.05,
                    help="fraction of legit rows kept when fitting (val/test never sampled)")
    a = ap.parse_args()
    main(a.csv, a.legit_frac)
