# PaySim results

Rows: 2,770,409 TRANSFER/CASH_OUT txns, fraud rate 0.296%. Split on `step` (train first 75%, test last 25%: 692,603 txns, 4480 fraud). Threshold tuned on validation for recall >= 0.9.

Fraud rate is 0.18% in train vs 0.65% in test (PaySim's legit volume falls off late in the month), so thresholds tuned on the past can undershoot the recall target.

| features | model | ROC-AUC | PR-AUC | precision | recall | FP | FN |
|---|---|---|---|---|---|---|---|
| all features | logistic_regression | 0.9922 | 0.6459 | 0.1791 | 0.9007 | 18499 | 445 |
| all features | random_forest | 1.0 | 0.9991 | 0.9976 | 0.8475 | 9 | 683 |
| no origin-balance features | logistic_regression | 0.8675 | 0.1049 | 0.0139 | 0.8973 | 286133 | 460 |
| no origin-balance features | random_forest | 0.9439 | 0.4795 | 0.0333 | 0.894 | 116396 | 475 |

**Why the full set is near-perfect:** 97.8% of PaySim fraud has `amount == oldbalanceOrg` exactly, vs 0.0% of legit txns. The simulator's fraud agent always empties the account, so `amt_to_orig_balance` hits exactly 1.0. That's a generator artifact, not a pattern real fraud follows, so the row without origin-balance features is the honest number.

Excluded from every run: `isFlaggedFraud`, `newbalanceOrig`, `newbalanceDest`, balance-error features (post-transaction simulator artifacts).
