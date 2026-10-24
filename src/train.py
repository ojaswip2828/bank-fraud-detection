"""Train & compare Logistic Regression vs Random Forest for fraud detection.

Key choices (be ready to defend these):
- Time-based split, not random: avoids training on the future.
- class_weight='balanced': fraud is <1% of data; without it the model predicts "not fraud" always.
- Metric = PR-AUC + recall/precision, NOT accuracy: 99% accuracy is trivial on imbalanced data.
- Threshold tuned on a validation slice (last 20% of train) to hit a target recall,
  because in banking a missed fraud costs far more than a false alert.
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, precision_score, recall_score,
                             f1_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from db import ROOT
from features import CATEGORICAL, NUMERIC, TARGET, load_features, time_split

REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
TARGET_RECALL = 0.90


def build_models():
    pre_scaled = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    pre_raw = ColumnTransformer([
        ("num", "passthrough", NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return {
        "logistic_regression": Pipeline([
            ("pre", pre_scaled),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0)),
        ]),
        "random_forest": Pipeline([
            ("pre", pre_raw),
            ("clf", RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=3,
                                           class_weight="balanced_subsample",
                                           n_jobs=-1, random_state=42)),
        ]),
    }


def threshold_for_recall(y, proba, target=TARGET_RECALL):
    """Highest threshold that still achieves the target recall (=> best precision)."""
    prec, rec, thr = precision_recall_curve(y, proba)
    ok = np.where(rec[:-1] >= target)[0]
    return float(thr[ok[-1]]) if len(ok) else 0.5


def evaluate(y, proba, thr):
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(thr, 4),
        "roc_auc": round(roc_auc_score(y, proba), 4),
        "pr_auc": round(average_precision_score(y, proba), 4),
        "precision": round(precision_score(y, pred, zero_division=0), 4),
        "recall": round(recall_score(y, pred), 4),
        "f1": round(f1_score(y, pred), 4),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def feature_names(pipe):
    return list(pipe.named_steps["pre"].get_feature_names_out())


def explain(name, pipe):
    names = feature_names(pipe)
    clf = pipe.named_steps["clf"]
    vals = clf.coef_[0] if name == "logistic_regression" else clf.feature_importances_
    s = pd.Series(vals, index=names)
    return s.reindex(s.abs().sort_values(ascending=False).index).head(10)


def main():
    REPORTS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    df = load_features()
    train, test = time_split(df, 0.25)
    fit_part, val_part = time_split(train, 0.20)

    X_cols = NUMERIC + CATEGORICAL
    results, lines = {}, []
    print(f"train={len(train):,} (fraud {train[TARGET].sum()}) | "
          f"test={len(test):,} (fraud {test[TARGET].sum()})")

    for name, pipe in build_models().items():
        # 1) tune threshold on validation using a model fit on the earlier slice
        pipe.fit(fit_part[X_cols], fit_part[TARGET])
        thr = threshold_for_recall(val_part[TARGET], pipe.predict_proba(val_part[X_cols])[:, 1])
        # 2) refit on full train, evaluate once on untouched future test set
        pipe.fit(train[X_cols], train[TARGET])
        proba = pipe.predict_proba(test[X_cols])[:, 1]
        m = evaluate(test[TARGET], proba, thr)
        m["at_default_0.5"] = evaluate(test[TARGET], proba, 0.5)
        results[name] = m
        joblib.dump({"pipeline": pipe, "threshold": thr, "features": X_cols},
                    MODELS / f"{name}.joblib")

        top = explain(name, pipe)
        label = "coefficient" if name == "logistic_regression" else "importance"
        lines += [f"\n### {name}", f"Top features ({label}):", "",
                  "| feature | value |", "|---|---|"]
        lines += [f"| {k} | {v:.4f} |" for k, v in top.items()]
        print(f"\n{name}: {json.dumps({k: v for k, v in m.items() if k != 'at_default_0.5'})}")

    # which fraud types slip through? (uses best model by PR-AUC)
    best = max(results, key=lambda k: results[k]["pr_auc"])
    bundle = joblib.load(MODELS / f"{best}.joblib")
    test = test.copy()
    test["flag"] = (bundle["pipeline"].predict_proba(test[X_cols])[:, 1] >= bundle["threshold"]).astype(int)
    from db import connect, query
    ftype = query(connect(), "SELECT txn_id, fraud_type FROM transactions WHERE is_fraud = 1")
    caught = test[test[TARGET] == 1].merge(ftype, on="txn_id").groupby("fraud_type")["flag"].agg(["sum", "count"])
    caught["recall"] = (caught["sum"] / caught["count"]).round(3)

    (REPORTS / "metrics.json").write_text(json.dumps(results, indent=2))
    md = ["# Model Report", "",
          f"Time-based split. Threshold tuned on validation for recall >= {TARGET_RECALL}.", "",
          "| model | ROC-AUC | PR-AUC | precision | recall | F1 | FP | FN |",
          "|---|---|---|---|---|---|---|---|"]
    for n, m in results.items():
        c = m["confusion"]
        md.append(f"| {n} | {m['roc_auc']} | {m['pr_auc']} | {m['precision']} | {m['recall']} "
                  f"| {m['f1']} | {c['fp']} | {c['fn']} |")
    md += ["", f"**Best by PR-AUC:** `{best}`", "", "### Recall by fraud type (best model)", "",
           caught.rename(columns={"sum": "caught", "count": "total"}).to_markdown()]
    md += lines
    (REPORTS / "model_report.md").write_text("\n".join(md) + "\n")
    print(f"\nBest: {best}\n{caught}")


if __name__ == "__main__":
    main()
