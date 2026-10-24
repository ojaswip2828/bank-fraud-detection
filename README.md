# Bank Transaction Analyzer — Fraud Detection (ML · SQL · BFSI)

End-to-end fraud-detection pipeline on ~220K bank transactions: relational data in **SQLite**, business analytics and **leakage-safe feature engineering in SQL window functions**, and a **Logistic Regression vs Random Forest** comparison tuned for the cost structure of banking (missed fraud ≫ false alert).

## Results (time-based split, threshold tuned for ≥90% recall)

| Model | ROC-AUC | PR-AUC | Precision | Recall | False alerts | Missed fraud |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.997 | 0.773 | 0.31 | 0.92 | 461 | 19 |
| **Random Forest** | **0.9996** | **0.928** | **0.70** | **0.95** | **92** | **11** |

Random Forest cut false alerts **5×** at higher recall. Per fraud type: card testing 99%, unusual amount 94%, account takeover 90% recall.
*(Numbers regenerate with `seed=42`; see `reports/model_report.md`.)*

## Architecture

```
generate_data.py ──► CSVs ──► SQLite (schema.sql: PK/FK/CHECK + indexes)
                                   │
                ┌──────────────────┼────────────────────┐
         analytics.sql        features.sql          (labels)
     fraud by channel/hour,   window functions:          │
     velocity bursts,         velocity, amount vs        │
     outliers vs baseline     history, new merchant…     │
                                   └────► train.py: LR vs RF ──► models/*.joblib
                                                                    │
                                                              predict.py (alert queue)
```

## Run it

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/pipeline.py            # data → DB → SQL reports → train both models (~1 min)
python src/predict.py --top 20    # riskiest transactions (analyst alert queue)
python src/predict.py --txn-id 98215
pytest -q                         # 7 tests: schema constraints, leakage, split, threshold
```

Outputs: `reports/sql/*.csv` (analytics), `reports/metrics.json`, `reports/model_report.md`.

## Data

Synthetic but behaviour-based — every column has a business meaning (unlike Kaggle's anonymised PCA `V1..V28`). Tables: `customers`, `merchants`, `transactions` (fraud rate ≈ 0.42%).

Injected fraud typologies:
- **Account takeover** — burst of high-value online/UPI txns, often at night or in another city
- **Card testing** — many tiny online txns within minutes
- **Unusual amount** — one txn far above the customer's normal spend

Deliberate legit noise so it isn't trivially separable: travellers, night-owl customers, occasional big purchases, rapid small UPI bursts (bill-splitting).

## Features (sql/features.sql)

| Feature | Fraud signal |
|---|---|
| `amt_to_hist_avg` | amount ÷ customer's *past* average |
| `txns_last_1h`, `mins_since_prev` | velocity (card testing, takeover bursts) |
| `amt_last_24h` | spend spike |
| `is_new_merchant` | first-ever txn at that merchant |
| `is_out_of_city`, `is_night` | location / time anomaly |
| `channel`, `category`, `account_age_days`, `income_band` | context |

All history features use `ROWS … 1 PRECEDING` → a transaction never sees itself or the future.

## Design decisions (interview talking points)

1. **Why not accuracy?** At 0.42% fraud, predicting "never fraud" scores 99.6%. I report **PR-AUC**, precision and recall.
2. **Imbalance:** `class_weight='balanced'` instead of SMOTE — no synthetic rows, works with time ordering.
3. **Time-based split:** train on Jan–early Mar, test on later data. A random split would leak future behaviour into training.
4. **Threshold tuning:** chose the threshold on a validation slice (last 20% of train) to reach ≥90% recall, then evaluated once on the untouched test set. The default 0.5 isn't meaningful for business cost.
5. **Why RF beat LR:** fraud is about *interactions* (small amount **and** online **and** many txns in an hour). LR is linear in features — it leaned on merchant category; RF picked up velocity + new-merchant combinations.
6. **Why keep LR:** coefficients are explainable to compliance/regulators (RBI model-risk expectations); it's the baseline.

## Limitations & next steps
- Synthetic data — next: validate on PaySim or IEEE-CIS public datasets.
- Add XGBoost/LightGBM and SHAP explanations per alert.
- Serve via FastAPI `/score` endpoint + Docker; stream txns via Kafka for real-time scoring.
- Monitor drift (fraud patterns change) and retrain on a schedule.

## Structure
```
sql/      schema.sql · analytics.sql · features.sql
src/      generate_data · db · analytics · features · train · predict · pipeline
tests/    test_pipeline.py
reports/  generated metrics & SQL outputs
```
