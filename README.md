# Bank Transaction Analyzer — Fraud Detection (ML · SQL · BFSI)

End-to-end fraud-detection pipeline on ~220K bank transactions: relational data in **SQLite**, business analytics and **leakage-safe feature engineering in SQL window functions**, a **Logistic Regression vs Random Forest** comparison tuned for the cost structure of banking (missed fraud ≫ false alert), a **feature ablation**, and **SHAP explanations for every alert**.

## Results (time-based split, threshold tuned for ≥90% recall)

| Model | ROC-AUC | PR-AUC | Precision | Recall | False alerts | Missed fraud |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.964 | 0.299 | 0.03 | 0.91 | 5,554 | 18 |
| **Random Forest** | **0.981** | **0.666** | **0.09** | **0.90** | **2,025** | **21** |

Test set: 55,449 future transactions, 210 fraud. Random Forest more than doubles PR-AUC and cuts false alerts by 64% at the same recall. Recall per fraud type: card testing 100%, account takeover 77%, unusual amount 76%.

**Reading the numbers honestly.** Holding 90% recall costs ~12 alerts for every fraud caught. That's the trade-off a bank faces when fraud is 0.4% of traffic and some fraud deliberately looks normal. At the default 0.5 threshold, the RF is 72% precise but catches only 56% of fraud. The threshold is a business decision, not a modelling one. *(Numbers regenerate with `seed=42`; see `reports/model_report.md`.)*

An earlier version of the data generator gave ROC-AUC 0.9996. That number was mostly the model learning the generator's rules back: fraud only ever happened in 3 of 8 merchant categories, always at never-seen merchants, at a fixed night-time window. The current generator removes those shortcuts (see [Data](#data)).

## Ablation — is the model leaning on one column?

Retrain both models with the two strongest "shortcut" features removed, same split and same threshold rule (`reports/ablation.csv`):

| Features | RF ROC-AUC | RF PR-AUC | RF false alerts @ 90% recall | LR PR-AUC |
|---|---|---|---|---|
| full model | 0.981 | 0.666 | 2,025 | 0.299 |
| − `category` | 0.962 | 0.658 | 6,412 | 0.359 |
| − `is_new_merchant` | 0.977 | 0.633 | 3,259 | 0.197 |
| − both | 0.940 | 0.608 | 13,745 | 0.221 |

- **The RF's ranking doesn't hinge on either feature.** Removing both costs only 0.058 PR-AUC: velocity, amount-vs-history and location carry most of the signal.
- **Those features matter at the high-recall end.** Without them, holding 90% recall needs ~7× more false alerts. The last, hardest frauds are the ones category and merchant-novelty help separate.
- **LR got *better* without `category`** (+0.06 PR-AUC). As a linear model it was over-trusting category, which is a weak signal now that fraud appears in all 8 categories.

## Explaining alerts (SHAP)

Every alert comes with its top reasons. SHAP values are computed per feature, with one-hot columns summed back into their source column so analysts see `channel`, not `channel_ONLINE`:

```
$ python src/predict.py --top 3
 txn_id  amount channel      category  fraud_score  reasons
 123248   41.85  ONLINE online_retail          1.0  log_amount=3.76 (+0.140); txns_last_1h=5 (+0.089); is_new_merchant=1 (+0.082)
 162681   57.50  ONLINE online_retail          1.0  log_amount=4.07 (+0.141); txns_last_1h=7 (+0.092); is_new_merchant=1 (+0.082)
```

That reads as "tiny amount, 5th transaction this hour, never used this merchant": a card-testing burst. The model report also ranks features globally by mean |SHAP|, which is less biased than Random Forest impurity importance. The top RF features are `log_amount`, `mins_since_prev`, `is_new_merchant` and `amt_to_hist_avg`.

## Public data: PaySim

To check the approach on fraud patterns this project didn't write, `src/paysim.py` runs the same models, threshold rule and time split on [PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1). PaySim is a 6.4M-row mobile-money simulation calibrated on real operator logs. Features are rebuilt for its schema: amount vs origin balance, account-draining transfers, destination-account history and velocity.

Excluded on purpose: `isFlaggedFraud` (PaySim's own rule output) and the post-transaction `newbalance*` / balance-error columns. Those make PaySim trivially separable and wouldn't be clean at authorisation time.

Test set: the last 25% of PaySim's timeline, 692,603 TRANSFER/CASH_OUT transactions with 4,480 fraud (`reports/paysim_report.md`):

| Features | Model | ROC-AUC | PR-AUC | Precision | Recall | False alerts |
|---|---|---|---|---|---|---|
| all | Logistic Regression | 0.992 | 0.646 | 0.18 | 0.90 | 18,499 |
| all | Random Forest | 1.000 | 0.999 | 1.00 | 0.85 | 9 |
| **no origin-balance features** | Logistic Regression | 0.868 | 0.105 | 0.01 | 0.90 | 286,133 |
| **no origin-balance features** | **Random Forest** | **0.944** | **0.480** | **0.03** | **0.89** | **116,396** |

- **The "all features" row is a simulator artifact, not a result.** 97.8% of PaySim fraud transfers *exactly* the origin account's balance (0.0% of legit transactions do). PaySim's fraud agent always empties the account, so `amt_to_orig_balance == 1.0` alone nearly identifies fraud. Real fraudsters don't follow that rule, so I drop the three origin-balance features and treat the second pair of rows as the real number.
- **The same story holds on data I didn't write.** Random Forest beats Logistic Regression by 4.6× on PR-AUC (0.48 vs 0.10), using only destination-account history, velocity, amount and transaction type. That lines up with the synthetic results (0.67 vs 0.30).
- **There's a time shift.** Fraud is 0.18% of the training period but 0.65% of the test period, because PaySim's legit volume drops late in the month. Thresholds tuned on the past land just under the 90% recall target (89%), the kind of drift a production model would need monitoring for.

```bash
# download the CSV from Kaggle into data/paysim/ first (needs a free Kaggle account)
python src/paysim.py --csv data/paysim/PS_20174392719_1491204439457_log.csv   # ~1.5 min
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
                                   └────► train.py: LR vs RF + ablation ──► models/*.joblib
                                                                    │
                                              predict.py (alert queue + SHAP reasons, explain.py)

paysim.py: same models / threshold rule / time split on public PaySim data
```

## Run it

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/pipeline.py              # data → DB → SQL reports → models → SHAP → ablation (~6 min)
python src/predict.py --top 20      # riskiest transactions with SHAP reasons (analyst alert queue)
python src/predict.py --txn-id 123248
python src/predict.py --flagged     # explain every alert → reports/alerts_explained.csv
pytest -q                           # 11 tests: schema, leakage, split, threshold, ablation, SHAP, PaySim features
```

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

1. **Why not accuracy?** At 0.40% fraud, predicting "never fraud" scores 99.6%. I report **PR-AUC**, precision and recall.
2. **Imbalance:** `class_weight='balanced'` instead of SMOTE. It adds no synthetic rows and works with time ordering.
3. **Time-based split:** train on Jan–early Mar, test on later data. A random split would leak future behaviour into training.
4. **Threshold tuning:** the threshold is chosen on a validation slice (last 20% of train) to reach ≥90% recall, then evaluated once on the untouched test set. The default 0.5 isn't meaningful for business cost.
5. **Why RF beat LR:** fraud is about *interactions* (small amount **and** online **and** many transactions in an hour). LR is linear in its features, so it leaned on merchant category and channel. RF combines velocity, amount and merchant novelty.
6. **Why I didn't trust 0.9996:** near-perfect metrics on synthetic data usually mean the generator has leaked into the label. I removed the shortcuts, then ran an ablation to show what's left.
7. **Why SHAP:** a flag with no reason can't be actioned by an analyst or defended to a regulator (RBI model-risk expectations). LR stays as the transparent baseline.

## Limitations & next steps
- The headline results are on synthetic data. PaySim is a public cross-check, but it is also simulated (and has its own shortcut, above). IEEE-CIS (real but anonymised) is the next step.
- Add XGBoost/LightGBM.
- The threshold is tuned on a model fit on 80% of train, then applied after refitting on all of it. Calibrating probabilities would make that transfer more reliable.
- Serve via a FastAPI `/score` endpoint + Docker; stream transactions via Kafka for real-time scoring.
- Monitor drift (fraud patterns change) and retrain on a schedule.

## Structure
```
sql/      schema.sql · analytics.sql · features.sql
src/      generate_data · db · analytics · features · train · explain · predict · paysim · pipeline
tests/    test_pipeline.py
reports/  generated metrics, ablation & SQL outputs
```
