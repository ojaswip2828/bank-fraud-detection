"""Train & compare Logistic Regression vs Random Forest vs LightGBM for fraud detection.

Key choices (be ready to defend these):
- Time-based split, not random: avoids training on the future.
- class_weight='balanced': fraud is <1% of data; without it the model predicts "not fraud" always.
- Metric = PR-AUC + recall/precision, NOT accuracy: 99% accuracy is trivial on imbalanced data.
- Isotonic calibration: balanced weights distort probabilities; calibration makes
  "0.30" mean a 30% chance again (Brier score) and puts every model on one scale.
- Threshold chosen for a target recall on out-of-time predictions pooled over
  3 expanding folds, because in banking a missed fraud costs far more than a false alert.
- Ablation: retrain without the strongest "shortcut" features to show the model
  isn't leaning on a single column.
"""
import json

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
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


def _calibrated(clf, calibrate):
    """Isotonic calibration (3 CV folds inside train) so a score means a probability.
    ensemble=False keeps ONE base model fit on all rows, which is what SHAP explains."""
    if not calibrate:
        return clf
    return CalibratedClassifierCV(clf, method="isotonic", cv=3, ensemble=False)


def build_models(numeric=NUMERIC, categorical=CATEGORICAL, rf_kwargs=None,
                 calibrate=True, only=None):
    rf = dict(n_estimators=300, max_depth=12, min_samples_leaf=3,
              class_weight="balanced_subsample", n_jobs=-1, random_state=42)
    rf.update(rf_kwargs or {})
    lgbm = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=50,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1)
    models = {
        "logistic_regression": Pipeline([
            ("pre", _pre(numeric, categorical, scale=True)),
            ("clf", _calibrated(LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0),
                                calibrate)),
        ]),
        "random_forest": Pipeline([
            ("pre", _pre(numeric, categorical, scale=False)),
            ("clf", _calibrated(RandomForestClassifier(**rf), calibrate)),
        ]),
        "lightgbm": Pipeline([
            ("pre", _pre(numeric, categorical, scale=False)),
            ("clf", _calibrated(LGBMClassifier(**lgbm), calibrate)),
        ]),
    }
    return {k: v for k, v in models.items() if only is None or k in only}


def threshold_for_recall(y, proba, target=TARGET_RECALL):
    """Highest threshold that still achieves the target recall (=> best precision)."""
    prec, rec, thr = precision_recall_curve(y, proba)
    ok = np.where(rec[:-1] >= target)[0]
    return float(thr[ok[-1]]) if len(ok) else 0.5


def precision_at_recall(y, proba, target=TARGET_RECALL):
    """Best precision achievable at >= target recall on this set. Threshold-free, so it
    compares models fairly; the operational threshold below is what you'd actually get."""
    prec, rec, _ = precision_recall_curve(y, proba)
    return float(prec[rec >= target].max())


def evaluate(y, proba, thr):
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(thr, 4),
        "roc_auc": round(roc_auc_score(y, proba), 4),
        "pr_auc": round(average_precision_score(y, proba), 4),
        "brier": round(brier_score_loss(y, proba), 5),
        "precision_at_90_recall": round(precision_at_recall(y, proba), 4),
        "precision": round(precision_score(y, pred, zero_division=0), 4),
        "recall": round(recall_score(y, pred), 4),
        "f1": round(f1_score(y, pred), 4),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


OOF_FOLDS = (0.55, 0.70, 0.85)   # expanding windows: fit on train[:k], predict the next 15%


def oof_threshold(pipe, train, cols, target=TARGET, fit_rows=None):
    """Threshold from out-of-time predictions pooled over 3 expanding-window folds.

    One validation slice holds too few (and bursty) frauds for a stable threshold;
    pooling ~2.5x more frauds steadies it. Models are calibrated, so scores from
    folds trained on different amounts of data share one probability scale."""
    fit_rows = fit_rows or (lambda d: d)
    n, ys, ps = len(train), [], []
    for k in OOF_FOLDS:
        lo, hi = int(n * k), int(n * (k + 0.15))
        past = fit_rows(train.iloc[:lo])
        m = clone(pipe).fit(past[cols], past[target])
        ps.append(m.predict_proba(train.iloc[lo:hi][cols])[:, 1])
        ys.append(train.iloc[lo:hi][target].to_numpy())
    return threshold_for_recall(np.concatenate(ys), np.concatenate(ps))


def fit_and_evaluate(pipe, train, test, cols, target=TARGET, fit_rows=None):
    """Pick the threshold out-of-time inside train, fit on all of train, score the test set once.

    fit_rows(df) -> df optionally subsamples the rows a model is *fit* on (threshold-tuning
    and test rows are never subsampled, so both reflect the real class mix)."""
    train = train.sort_values("txn_time")
    thr = oof_threshold(pipe, train, cols, target, fit_rows)
    tr = (fit_rows or (lambda d: d))(train)
    pipe.fit(tr[cols], tr[target])
    proba = pipe.predict_proba(test[cols])[:, 1]
    m = evaluate(test[target], proba, thr)
    m["at_default_0.5"] = evaluate(test[target], proba, 0.5)
    return m, thr, proba


def run_ablation(train, test, models=("logistic_regression", "lightgbm")):
    """Retrain models with feature groups removed; report the drop vs the full model.
    RF is left out: calibrated + 4 fits per run makes it ~5 min per row for no new insight."""
    rows = []
    for label, drop in ABLATIONS.items():
        num = [c for c in NUMERIC if c not in drop]
        cat = [c for c in CATEGORICAL if c not in drop]
        for name, pipe in build_models(num, cat, only=models).items():
            m, _, _ = fit_and_evaluate(pipe, train, test, num + cat)
            rows.append({"features": label, "model": name, "roc_auc": m["roc_auc"],
                         "pr_auc": m["pr_auc"],
                         "precision_at_90_recall": m["precision_at_90_recall"],
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
        unit = "raw-score units" if name == "random_forest" else "log-odds units"
        lines += [f"\n### {name}", f"Top features (mean |SHAP|, {unit}, 500 test rows):", "",
                  "| feature | mean abs SHAP |", "|---|---|"]
        lines += [f"| {k} | {v:.4f} |" for k, v in top.items()]

    # what calibration buys: same LightGBM, same threshold rule, no isotonic step
    raw = build_models(calibrate=False, only=["lightgbm"])["lightgbm"]
    calib_ref, _, _ = fit_and_evaluate(raw, train, test, X_cols)
    print(f"\nlightgbm uncalibrated: brier={calib_ref['brier']} recall={calib_ref['recall']}")

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
          "Time-based split. All models isotonic-calibrated. Threshold chosen for recall >= "
          f"{TARGET_RECALL} on out-of-time predictions pooled over {len(OOF_FOLDS)} expanding "
          "folds inside train, then applied once to the untouched test set.", "",
          "`P@90R` = best precision reachable at 90% recall on test (threshold-free model "
          "comparison). `precision` / `recall` / `FP` / `FN` = what the chosen threshold actually gave.", "",
          "| model | ROC-AUC | PR-AUC | P@90R | Brier | precision | recall | FP | FN |",
          "|---|---|---|---|---|---|---|---|---|"]
    rows = {**results, "lightgbm (uncalibrated)": calib_ref}
    for n, m in rows.items():
        c = m["confusion"]
        md.append(f"| {n} | {m['roc_auc']} | {m['pr_auc']} | {m['precision_at_90_recall']} "
                  f"| {m['brier']} | {m['precision']} | {m['recall']} | {c['fp']} | {c['fn']} |")
    results["lightgbm_uncalibrated"] = calib_ref
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
