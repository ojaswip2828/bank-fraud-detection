"""Train & compare Logistic Regression vs Random Forest for fraud detection.

Key choices (be ready to defend these):
- Time-based split, not random: avoids training on the future.
- class_weight='balanced': fraud is <1% of data; without it the model predicts "not fraud" always.
- Metric = PR-AUC + recall/precision, NOT accuracy: 99% accuracy is trivial on imbalanced data.
- Threshold tuned on a validation slice (last 20% of train) to hit a target recall,
  because in banking a missed fraud costs far more than a false alert.
- Ablation: retrain without the strongest "shortcut" features to show the model
  isn't leaning on a single column.
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

import explain
from db import ROOT
from features import CATEGORICAL, NUMERIC, TARGET, load_features, time_split

REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
TARGET_RECALL = 0.90
ABLATIONS = {
    "full model": [],
    "no category": ["category"],
    "no is_new_merchant": ["is_new_merchant"],
    "no category + no is_new_merchant": ["category", "is_new_merchant"],
}


def _pre(numeric, categorical, scale):
    parts = [("num", StandardScaler() if scale else "passthrough", numeric)]
    if categorical:
        parts.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical))
    return ColumnTransformer(parts)


def build_models(numeric=NUMERIC, categorical=CATEGORICAL, rf_kwargs=None):
    rf = dict(n_estimators=300, max_depth=12, min_samples_leaf=3,
              class_weight="balanced_subsample", n_jobs=-1, random_state=42)
    rf.update(rf_kwargs or {})
    return {
        "logistic_regression": Pipeline([
            ("pre", _pre(numeric, categorical, scale=True)),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0)),
        ]),
        "random_forest": Pipeline([
            ("pre", _pre(numeric, categorical, scale=False)),
            ("clf", RandomForestClassifier(**rf)),
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


def fit_and_evaluate(pipe, train, test, cols, target=TARGET, fit_rows=None):
    """Tune threshold on a validation slice, refit on all of train, score the test set once.

    fit_rows(df) -> df optionally subsamples the rows a model is *fit* on (validation and
    test are never subsampled, so the threshold and metrics reflect the real class mix)."""
    fit_rows = fit_rows or (lambda d: d)
    fit_part, val_part = time_split(train, 0.20)
    fp = fit_rows(fit_part)
    pipe.fit(fp[cols], fp[target])
    thr = threshold_for_recall(val_part[target], pipe.predict_proba(val_part[cols])[:, 1])
    tr = fit_rows(train)
    pipe.fit(tr[cols], tr[target])
    proba = pipe.predict_proba(test[cols])[:, 1]
    m = evaluate(test[target], proba, thr)
    m["at_default_0.5"] = evaluate(test[target], proba, 0.5)
    return m, thr, proba


def run_ablation(train, test):
    """Retrain both models with feature groups removed; report the drop vs the full model."""
    rows = []
    for label, drop in ABLATIONS.items():
        num = [c for c in NUMERIC if c not in drop]
        cat = [c for c in CATEGORICAL if c not in drop]
        for name, pipe in build_models(num, cat).items():
            m, _, _ = fit_and_evaluate(pipe, train, test, num + cat)
            rows.append({"features": label, "model": name, "roc_auc": m["roc_auc"],
                         "pr_auc": m["pr_auc"], "precision": m["precision"],
                         "recall": m["recall"], "false_alerts": m["confusion"]["fp"]})
    df = pd.DataFrame(rows)
    base = df[df.features == "full model"].set_index("model")["pr_auc"]
    df["pr_auc_drop"] = (df["pr_auc"] - df["model"].map(base)).round(4)
    return df


def save_bundle(name, pipe, thr, cols, train):
    sample = train[cols].sample(min(500, len(train)), random_state=0)
    bg = explain._dense(pipe.named_steps["pre"].transform(sample))
    bundle = {"pipeline": pipe, "threshold": thr, "features": cols, "background": bg}
    joblib.dump(bundle, MODELS / f"{name}.joblib")
    return bundle


def main(ablation=True):
    REPORTS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    df = load_features()
    train, test = time_split(df, 0.25)

    X_cols = NUMERIC + CATEGORICAL
    results, bundles, lines = {}, {}, []
    print(f"train={len(train):,} (fraud {train[TARGET].sum()}) | "
          f"test={len(test):,} (fraud {test[TARGET].sum()})")

    for name, pipe in build_models().items():
        m, thr, _ = fit_and_evaluate(pipe, train, test, X_cols)
        results[name] = m
        bundles[name] = save_bundle(name, pipe, thr, X_cols, train)
        print(f"\n{name}: {json.dumps({k: v for k, v in m.items() if k != 'at_default_0.5'})}")

        top = explain.global_importance(bundles[name], test[X_cols]).head(10)
        unit = "fraud-probability units" if name == "random_forest" else "log-odds units"
        lines += [f"\n### {name}", f"Top features (mean |SHAP|, {unit}, 500 test rows):", "",
                  "| feature | mean abs SHAP |", "|---|---|"]
        lines += [f"| {k} | {v:.4f} |" for k, v in top.items()]

    # which fraud types slip through? (uses best model by PR-AUC)
    best = max(results, key=lambda k: results[k]["pr_auc"])
    bundle = bundles[best]
    test = test.copy()
    test["flag"] = (bundle["pipeline"].predict_proba(test[X_cols])[:, 1] >= bundle["threshold"]).astype(int)
    from db import connect, query
    ftype = query(connect(), "SELECT txn_id, fraud_type FROM transactions WHERE is_fraud = 1")
    caught = test[test[TARGET] == 1].merge(ftype, on="txn_id").groupby("fraud_type")["flag"].agg(["sum", "count"])
    caught["recall"] = (caught["sum"] / caught["count"]).round(3)

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

    if ablation:
        abl = run_ablation(train, test)
        results["ablation"] = abl.to_dict(orient="records")
        abl.to_csv(REPORTS / "ablation.csv", index=False)
        md += ["", "## Ablation: removing the strongest shortcut features", "",
               "Same split, same threshold rule; `pr_auc_drop` is vs the full model.", "",
               abl.to_markdown(index=False)]
        print(f"\nAblation\n{abl.to_string(index=False)}")

    md += ["", "## Global feature importance (SHAP)"] + lines
    (REPORTS / "metrics.json").write_text(json.dumps(results, indent=2))
    (REPORTS / "model_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nBest: {best}\n{caught}")


if __name__ == "__main__":
    main()
