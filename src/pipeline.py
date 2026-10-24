"""One command end-to-end: generate data -> build DB -> SQL analytics -> train models."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import analytics  # noqa: E402
import db  # noqa: E402
import train  # noqa: E402
from generate_data import generate  # noqa: E402

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    cu, me, tx = generate()
    cu.to_csv("data/customers.csv", index=False)
    me.to_csv("data/merchants.csv", index=False)
    tx.to_csv("data/transactions.csv", index=False)
    print(f"[1/4] data: {len(tx):,} txns, fraud rate {tx.is_fraud.mean():.3%}")
    if db.DB_PATH.exists():
        db.DB_PATH.unlink()
    db.build_db()
    print("[2/4] SQLite DB built")
    print("[3/4] SQL analytics"); analytics.main()
    print("\n[4/4] training"); train.main()
