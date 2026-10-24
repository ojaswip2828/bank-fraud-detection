"""Run every business query in sql/analytics.sql and save results to reports/."""
from pathlib import Path

from db import ROOT, connect, named_queries, query

OUT = ROOT / "reports" / "sql"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    con = connect()
    for name, sql in named_queries("analytics.sql").items():
        df = query(con, sql)
        df.to_csv(OUT / f"{name}.csv", index=False)
        print(f"\n=== {name} ===")
        print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
