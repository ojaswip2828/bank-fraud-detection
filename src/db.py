"""SQLite helpers: build the database from CSVs and run .sql files."""
import re
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "bank.db"
SQL_DIR = ROOT / "sql"


def connect(path=DB_PATH):
    con = sqlite3.connect(path)
    con.execute("PRAGMA foreign_keys = ON")
    return con


def build_db(data_dir=ROOT / "data", path=DB_PATH):
    con = connect(path)
    con.executescript((SQL_DIR / "schema.sql").read_text())
    for table in ["customers", "merchants", "transactions"]:
        df = pd.read_csv(Path(data_dir) / f"{table}.csv")
        df.to_sql(table, con, if_exists="append", index=False)
    con.commit()
    return con


def named_queries(sql_file):
    """Split a file on '-- name: x' markers -> {name: sql}."""
    text = (SQL_DIR / sql_file).read_text()
    parts = re.split(r"^-- name:\s*(\w+)\s*$", text, flags=re.M)
    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts), 2)}


def query(con, sql, params=None):
    return pd.read_sql_query(sql, con, params=params)


if __name__ == "__main__":
    c = build_db()
    n = c.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    print(f"Built {DB_PATH} with {n:,} transactions")
