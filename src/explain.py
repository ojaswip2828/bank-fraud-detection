"""SHAP explanations: why did the model flag this transaction?

One-hot columns (e.g. channel_ONLINE, channel_UPI) are summed back into their
source feature ("channel") so each reason maps to a column an analyst knows.
SHAP explains the model under the probability calibration.
Units: Random Forest -> change in raw score; LightGBM / Logistic Regression -> log-odds.
"""
import numpy as np
import pandas as pd


def _dense(X):
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)


def raw_feature(transformed_name, raw_cols):
    """'cat__channel_ONLINE' -> 'channel', 'num__log_amount' -> 'log_amount'."""
    name = transformed_name.split("__", 1)[1]
    if name in raw_cols:
        return name
    return max((c for c in raw_cols if name.startswith(c + "_")), key=len)


def base_model(clf):
    """The fitted model under an isotonic CalibratedClassifierCV (ensemble=False), or clf itself.
    Calibration is a monotonic re-mapping, so the base model's SHAP reasons rank the same."""
    cc = getattr(clf, "calibrated_classifiers_", None)
    return cc[0].estimator if cc else clf


def is_tree(model):
    return hasattr(model, "estimators_") or hasattr(model, "booster_")


def shap_frame(bundle, X):
    """SHAP values per *raw* feature for rows X (DataFrame of bundle['features'])."""
    pipe = bundle["pipeline"]
    pre, clf = pipe.named_steps["pre"], base_model(pipe.named_steps["clf"])
    Xt = _dense(pre.transform(X))
    if hasattr(clf, "booster_"):
        # LightGBM's built-in TreeSHAP: same algorithm, no shap dependency (keeps the API image small)
        vals = clf.predict(Xt, pred_contrib=True, raw_score=True)[:, :-1]
    elif is_tree(clf):
        import shap
        vals = shap.TreeExplainer(clf).shap_values(Xt)
        vals = vals[1] if isinstance(vals, list) else vals
        vals = vals[:, :, 1] if vals.ndim == 3 else vals   # (n, features, classes) -> fraud class
    else:
        import shap
        bg = bundle["background"]
        masker = shap.maskers.Independent(bg, max_samples=len(bg))
        vals = shap.LinearExplainer(clf, masker).shap_values(Xt)
    names = pre.get_feature_names_out()
    sv = pd.DataFrame(vals, columns=names, index=X.index)
    groups = [raw_feature(n, bundle["features"]) for n in names]
    return sv.T.groupby(groups).sum().T


def top_reasons(sv, X, k=3):
    """['txns_last_1h=6 (+0.214)', ...] for each row: the k features pushing hardest toward fraud."""
    out = []
    for idx, row in sv.iterrows():
        parts = []
        for feat in row.sort_values(ascending=False).index[:k]:
            if row[feat] <= 0:
                break
            v = X.at[idx, feat]
            v = f"{v:.2f}" if isinstance(v, (float, np.floating)) else v
            parts.append(f"{feat}={v} ({row[feat]:+.3f})")
        out.append("; ".join(parts))
    return out


def global_importance(bundle, X, n=500, seed=0):
    """Mean |SHAP| per raw feature on a sample — a less biased ranking than RF impurity."""
    sample = X.sample(min(n, len(X)), random_state=seed)
    return shap_frame(bundle, sample).abs().mean().sort_values(ascending=False)
