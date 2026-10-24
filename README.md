# Bank Transaction Analyzer — Fraud Detection (ML · SQL · BFSI)

> **TL;DR**: calibrated LightGBM fraud model with SQL-built features, SHAP reasons for every alert, and a Dockerised scoring API.
> - **PR-AUC 0.72**, vs 0.66 for Random Forest and 0.26 for Logistic Regression, on 55K future transactions with 0.4% fraud
> - **94% of fraud caught**, with a threshold chosen in advance on past data only
> - **12.9% precision at 90% recall**, 60% better than Random Forest
>
> Same model ranking on the public **PaySim** dataset. Near-perfect scores (0.9996 here, 1.000 on PaySim) were traced to data shortcuts and removed.
>
> **Live demo:** [bank-fraud-detection-19rh.onrender.com/docs](https://bank-fraud-detection-19rh.onrender.com/docs). Open `POST /score` → *Try it out* → pick an example → *Execute*. It runs on a free instance, so the first request after 15 idle minutes takes ~1 minute to wake up.

End-to-end fraud detection on ~220K bank transactions:
- relational data in **SQLite**
- business analytics and **leakage-safe feature engineering in SQL window functions**
- **Logistic Regression vs Random Forest vs LightGBM**, calibrated and tuned for the cost structure of banking (missed fraud ≫ false alert)
- a **feature ablation** and a **public-data cross-check** (PaySim)
- **SHAP reasons for every alert**
- a **Dockerised FastAPI service** that scores a raw transaction in real time

## Results (time-based split, threshold set for ≥90% recall)

Test set: 55,449 future transactions, 210 fraud.

| Model | ROC-AUC | PR-AUC | Precision @ 90% recall | Brier | Recall (chosen threshold) | False alerts |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.964 | 0.262 | 2.5% | 0.00313 | 0.95 | 10,605 |
| Random Forest | 0.981 | 0.658 | 8.0% | 0.00181 | 0.93 | 3,407 |
| **LightGBM** | **0.987** | **0.721** | **12.9%** | **0.00156** | **0.94** | **2,886** |

- **LightGBM wins on every ranking metric.** At the same 90% recall it's 60% more precise than Random Forest. Recall per fraud type: card testing 100%, account takeover 88%, unusual amount 84%.
- **Precision @ 90% recall** is the best precision any threshold could reach on test, so it compares *models* fairly. The last two columns are what the threshold chosen *in advance* actually delivered.
- **Reading the numbers honestly.** Even the best model raises ~15 alerts per fraud caught at this recall level. That's the trade-off when fraud is 0.4% of traffic and some fraud deliberately looks normal. At the default 0.5 threshold, LightGBM is 95% precise but catches only 58% of fraud. The threshold is a business decision. *(All numbers regenerate exactly with `seed=42`. The Docker build on Linux reproduces them to 4 decimals; see `reports/model_report.md`.)*

An earlier version of the data generator gave ROC-AUC 0.9996. That number was mostly the model learning the generator's rules back: fraud only ever happened in 3 of 8 merchant categories, always at never-seen merchants, at a fixed night-time window. The current generator removes those shortcuts (see [Data](#data)).

## Calibration and the threshold

`class_weight='balanced'` pushes scores up, so a raw "0.3" isn't a 30% chance. Every model is wrapped in **isotonic calibration** (3 CV folds inside train).

The **threshold** is chosen on *out-of-time* predictions from 3 expanding-window folds inside the training period (340 frauds), then applied once to the test set. Results for LightGBM:

| | PR-AUC | Brier (lower = truer probabilities) | Recall | False alerts |
|---|---|---|---|---|
| uncalibrated | 0.742 | 0.00271 | 0.95 | 3,842 |
| **calibrated** | 0.721 | **0.00156** | 0.94 | **2,886** |

- **Calibration bought honest probabilities** (Brier −42%) and **25% fewer false alerts** at about the same recall. It cost 0.02 PR-AUC, because isotonic regression flattens some scores into ties.
- **What didn't work:** tuning the threshold on a single validation slice (only 136 frauds, many in bursts). Test recall swung between 87% and 95% depending on the setup, with or without calibration. Pooling three out-of-time folds steadied it. Calibration is what makes pooling valid, because it puts models trained on different amounts of data onto one probability scale.

## Ablation — is the model leaning on one column?

Retrain with the two strongest "shortcut" features removed, same split and same threshold rule (`reports/ablation.csv`):

| Features | LightGBM PR-AUC | LightGBM precision @ 90% recall | LR PR-AUC |
|---|---|---|---|
| full model | 0.721 | 12.9% | 0.262 |
| − `category` | 0.675 | 1.8% | 0.344 |
| − `is_new_merchant` | 0.661 | 8.1% | 0.184 |
| − both | 0.609 | 2.1% | 0.194 |

- **The ranking doesn't hinge on either feature.** Removing both costs 0.11 PR-AUC: velocity, amount-vs-history and location carry most of the signal.
- **But LightGBM needs `category` at the high-recall end.** Without it, precision at 90% recall falls from 12.9% to 1.8%. The hardest frauds sit in a few category/amount combinations, and global SHAP ranks `category` first for LightGBM.
- **LR got *better* without `category`** (+0.08 PR-AUC). As a linear model it over-trusts category, now a weak signal on its own since fraud appears in all 8 categories.

## Explaining alerts (SHAP)

Every alert comes with its top reasons. One-hot columns are summed back into their source column, so analysts see `channel`, not `channel_ONLINE`. For LightGBM this uses its built-in TreeSHAP. That's the same algorithm as the `shap` package, which keeps the API image small.

```
$ python src/predict.py --top 3
 txn_id   amount channel category fraud_score  reasons
     40 25758.65     UPI   travel         1.0  mins_since_prev=5.00 (+3.093); is_night=1 (+3.048); log_amount=10.16 (+2.322)
     41   217.60     POS     fuel         1.0  is_night=1 (+2.960); log_amount=5.39 (+2.955); mins_since_prev=8.00 (+2.918)
     43   100.22     UPI  grocery         1.0  is_night=1 (+3.254); log_amount=4.62 (+2.844); is_new_merchant=1 (+2.140)
```

Transaction 40 is a high-value travel booking 5 minutes after the previous one at 1:47am: an account-takeover burst. Transactions 41 and 43 are small probes at a fuel pump and a grocery store minutes apart at night: card testing outside the usual online channel. Units are log-odds. The model report also ranks features globally by mean |SHAP|.

## Real-time scoring API

`POST /score` takes a **raw transaction**, not features:

```json
{"customer_id": 1411, "merchant_id": 35, "amount": 49.0,
 "txn_time": "2026-03-31T03:12:00", "channel": "ONLINE", "txn_city": "Delhi"}
```

```json
{"fraud_probability": 1.0, "flagged": true, "threshold": 0.0082, "model": "lightgbm",
 "reasons": [{"feature": "log_amount", "value": 3.912, "contribution": 4.9831},
             {"feature": "is_new_merchant", "value": 1, "contribution": 2.9225},
             {"feature": "channel", "value": "ONLINE", "contribution": 2.3357}],
 "features": {"...": "all 13 model inputs"}}
```

**No train/serve skew.** The API doesn't re-implement features in Python. A SQLite `TEMP` table named `transactions` shadows the real one for that request only. It holds the customer's history plus the new transaction, and the *same* `sql/features.sql` used in training runs on it, about 100 rows instead of 220K. A test removes a stored transaction, scores it through this path, and checks every feature matches what training saw.

Measured in Docker with a 512 MB memory cap: **123 MB RAM, 20–100 ms per request.** The routine example (a grocery UPI payment) scores 0.0002. The card-testing example above scores 1.0.

```bash
# local, without Docker
uvicorn app:app --app-dir src --port 8000           # → http://localhost:8000/docs

# Docker: build generates data, builds SQLite and trains LightGBM inside the image (~5–15 min)
docker build -t bank-fraud-detection .
docker run -p 8000:8000 bank-fraud-detection         # → http://localhost:8000/docs
```

**Deployed** on Render's free tier from `render.yaml`: [live API docs](https://bank-fraud-detection-19rh.onrender.com/docs). To deploy your own copy, choose **New → Blueprint** on [Render](https://render.com) and point it at this repo. The free tier sleeps after 15 minutes idle, so the first request after that takes ~1 minute.

## Public data: PaySim

To check the approach on fraud patterns this project didn't write, `src/paysim.py` runs the same models, calibration, threshold rule and time split on [PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1). PaySim is a 6.4M-row mobile-money simulation calibrated on real operator logs. Features are rebuilt for its schema: amount vs origin balance, account-draining transfers, destination-account history and velocity.

Excluded on purpose: `isFlaggedFraud` (PaySim's own rule output) and the post-transaction `newbalance*` / balance-error columns. Those make PaySim trivially separable and wouldn't be clean at authorisation time.

Test set: the last 25% of PaySim's timeline, 692,603 TRANSFER/CASH_OUT transactions with 4,480 fraud (`reports/paysim_report.md`):

| Features | Model | ROC-AUC | PR-AUC | Recall | False alerts |
|---|---|---|---|---|---|
| all | LightGBM | 1.000 | 1.000 | 1.00 | 1 |
| **no origin-balance features** | Logistic Regression | 0.868 | 0.111 | 0.90 | 293,467 |
| **no origin-balance features** | Random Forest | 0.942 | 0.434 | 0.88 | 93,796 |
| **no origin-balance features** | **LightGBM** | **0.939** | **0.530** | **0.88** | **91,002** |

- **The "all features" row is a simulator artifact, not a result.** 97.8% of PaySim fraud transfers *exactly* the origin account's balance (0.0% of legit transactions do). PaySim's fraud agent always empties the account, so `amt_to_orig_balance == 1.0` alone nearly identifies fraud. Real fraudsters don't follow that rule, so I drop the three origin-balance features and treat those rows as the real numbers.
- **The model ranking holds on data I didn't write.** LightGBM > Random Forest > Logistic Regression on PR-AUC (0.53 / 0.43 / 0.11), using only destination-account history, velocity, amount and transaction type. This mirrors the synthetic results.
- **There's a time shift.** Fraud is 0.18% of the training period but 0.65% of the test period, because PaySim's legit volume drops late in the month. Thresholds set on the past land just under the 90% recall target (88%). A production model would need monitoring for this kind of drift.

```bash
# download the CSV from Kaggle into data/paysim/ first (needs a free Kaggle account)
python src/paysim.py --csv data/paysim/PS_20174392719_1491204439457_log.csv   # ~5 min
```

## Architecture

```
generate_data.py ──► CSVs ──► SQLite (schema.sql: PK/FK/CHECK + indexes)
                                   │
                ┌──────────────────┼────────────────────┐
         analytics.sql        features.sql          (labels)
     fraud by channel/hour,   window functions:          │
     velocity bursts,         velocity, amount vs        │
     outliers vs baseline     history, new merchant…     │
                                   ├────► train.py: LR / RF / LightGBM, calibrated,
                                   │      out-of-time threshold, ablation ──► models/*.joblib
                                   │                                              │
                                   │                   predict.py (alert queue + SHAP reasons)
                                   │                                              │
                                   └──── scoring.py (same SQL, one customer) ◄── app.py  POST /score

paysim.py: same models / calibration / threshold rule / time split on public PaySim data
```

## Run it

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/pipeline.py              # data → DB → SQL reports → 3 models → SHAP → ablation (~12 min)
python src/predict.py --top 20      # riskiest transactions with SHAP reasons (analyst alert queue)
python src/predict.py --txn-id 40
python src/predict.py --flagged     # explain every alert → reports/alerts_explained.csv
uvicorn app:app --app-dir src       # scoring API → http://localhost:8000/docs
pytest -q                           # 14 tests (see below)
```

The tests cover schema constraints, leakage, time split, threshold, ablation, SHAP additivity (RF and calibrated LightGBM), PaySim features, train/serve feature parity, and API validation.

Outputs: `reports/sql/*.csv` (analytics), `reports/metrics.json`, `reports/ablation.csv`, `reports/model_report.md`.

## Data

Synthetic but behaviour-based. Every column has a business meaning, unlike Kaggle's anonymised PCA `V1..V28`. Tables: `customers`, `merchants`, `transactions` (fraud rate ≈ 0.40%).

Injected fraud typologies:
- **Account takeover:** a burst of high-value transactions, often at night, in another city or online. In 35% of cases the fraudster reuses the victim's *usual* merchants (stolen phone / hijacked session).
- **Card testing:** many tiny transactions within minutes. Mostly online, but 30% are UPI/POS probes at grocery, fuel or restaurants.
- **Unusual amount:** one transaction far above the customer's norm. It uses *the same amount distribution as legitimate big purchases*, half of them at the victim's regular shops.

Built so no single column gives fraud away:
- Fraud appears in all 8 merchant categories (gift-card and fuel-card fraud included).
- Legit customers include travellers, night owls, people visiting new merchants (6%), big purchases at new shops, rapid small UPI bursts (bill-splitting), and online travel/electronics/utility payments.
- The amount-outlier SQL report is dominated by *legit* transactions. Amount alone isn't enough, which is the point.

## Features (sql/features.sql)

| Feature | Fraud signal |
|---|---|
| `amt_to_hist_avg` | amount ÷ customer's *past* average |
| `txns_last_1h`, `mins_since_prev` | velocity (card testing, takeover bursts) |
| `amt_last_24h` | spend spike |
| `is_new_merchant` | first-ever txn at that merchant |
| `is_out_of_city`, `is_night` | location / time anomaly |
| `channel`, `category`, `account_age_days`, `income_band` | context |

All history features use `ROWS … 1 PRECEDING` (ordered by time, then `txn_id` for ties), so a transaction never sees itself or the future.

## Design decisions (interview talking points)

1. **Why not accuracy?** At 0.40% fraud, predicting "never fraud" scores 99.6%. I report **PR-AUC**, precision at a fixed recall, and recall.
2. **Imbalance:** `class_weight='balanced'` instead of SMOTE. It adds no synthetic rows and works with time ordering. Calibration then undoes its distortion of the probabilities.
3. **Time-based split:** train on Jan–early Mar, test on later data. A random split would leak future behaviour into training.
4. **Threshold:** chosen for ≥90% recall on pooled out-of-time predictions inside train, evaluated once on the untouched test set. A single validation slice was too noisy (see [Calibration](#calibration-and-the-threshold)).
5. **Why boosting won:** fraud is about *interactions* (small amount **and** online **and** many transactions in an hour). LR is linear in its features. RF and LightGBM both capture interactions, and boosting focuses each new tree on the cases the previous ones got wrong, which are the hard frauds.
6. **Why I didn't trust 0.9996 (or PaySim's 1.000):** near-perfect metrics usually mean the data generator has leaked into the label. I found the shortcut both times, removed it, and ran an ablation to show what's left.
7. **Why SHAP:** a flag with no reason can't be actioned by an analyst or defended to a regulator (RBI model-risk expectations). LR stays as the transparent baseline.
8. **Why the API reuses the training SQL:** re-implementing features for serving is the classic source of train/serve skew. One SQL file, two callers, and a test that proves they agree.

## Limitations & next steps
- The headline results are on synthetic data. PaySim is a public cross-check, but it is also simulated (and has its own shortcut, above). IEEE-CIS (real but anonymised) is the next step.
- The API scores against a fixed demo history (Jan–Mar 2026) and doesn't store new transactions. Production would write each scored transaction back and stream them in (Kafka).
- Thresholds drift when the fraud rate shifts (seen on PaySim). Monitor alert volume and recall, and retrain on a schedule.

## Structure
```
sql/      schema.sql · analytics.sql · features.sql
src/      generate_data · db · analytics · features · train · explain · predict · paysim · pipeline
          scoring · app (API) · build_serving (Docker build step)
tests/    test_pipeline.py
reports/  generated metrics, ablation & SQL outputs
Dockerfile · render.yaml · requirements-serve.txt
```
