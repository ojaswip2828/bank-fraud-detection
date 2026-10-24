"""Build only what the API needs: data -> SQLite -> calibrated LightGBM bundle.

Used by the Dockerfile at image-build time (no analytics, ablation or RF), so the
deployed image is reproducible from the repo with seed=42 and no binary artifacts in git.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import db  # noqa: E402
import train  # noqa: E402
from features import CATEGORICAL, NUMERIC, load_features, time_split  # noqa: E402
from generate_data import generate  # noqa: E402

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    cu, me, tx = generate()
    for name, df in [("customers", cu), ("merchants", me), ("transactions", tx)]:
        df.to_csv(f"data/{name}.csv", index=False)
    db.DB_PATH.unlink(missing_ok=True)
    db.build_db().close()
    print(f"DB built: {len(tx):,} transactions")

    train.MODELS.mkdir(exist_ok=True)
    df = load_features()
    tr, te = time_split(df, 0.25)
    cols = NUMERIC + CATEGORICAL
    pipe = train.build_models(only=["lightgbm"])["lightgbm"]
    m, thr, _ = train.fit_and_evaluate(pipe, tr, te, cols)
    train.save_bundle("lightgbm", pipe, thr, cols, tr)
    print(f"lightgbm: pr_auc={m['pr_auc']} recall={m['recall']} precision={m['precision']} thr={thr:.4f}")
