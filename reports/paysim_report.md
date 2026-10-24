# PaySim results

Rows: 2,770,409 TRANSFER/CASH_OUT txns, fraud rate 0.296%. Split on `step` (train first 75%, test last 25%: 692,603 txns, 4480 fraud). Threshold tuned on validation for recall >= 0.9.

Fraud rate is 0.18% in train vs 0.65% in test (PaySim's legit volume falls off late in the month), so thresholds tuned on the past can undershoot the recall target.

| features | model | ROC-AUC | PR-AUC | precision | recall | FP | FN |
|---|---|---|---|---|---|---|---|
| all features | logistic_regression | 0.9922 | 0.6654 | 0.1801 | 0.9009 | 18379 | 444 |
| all features | random_forest | 1.0 | 0.9985 | 0.9987 | 0.9969 | 6 | 14 |
| all features | lightgbm | 1.0 | 0.9996 | 0.9998 | 0.9987 | 1 | 6 |
| no origin-balance features | logistic_regression | 0.8681 | 0.1112 | 0.0135 | 0.8989 | 293467 | 453 |
| no origin-balance features | random_forest | 0.9416 | 0.434 | 0.0404 | 0.8821 | 93796 | 528 |
| no origin-balance features | lightgbm | 0.9387 | 0.5298 | 0.0416 | 0.8808 | 91002 | 534 |

**Why the full set is near-perfect:** 97.8% of PaySim fraud has `amount == oldbalanceOrg` exactly, vs 0.0% of legit txns. The simulator's fraud agent always empties the account, so `amt_to_orig_balance` hits exactly 1.0. That's a generator artifact, not a pattern real fraud follows, so the row without origin-balance features is the honest number.

Excluded from every run: `isFlaggedFraud`, `newbalanceOrig`, `newbalanceDest`, balance-error features (post-transaction simulator artifacts).
