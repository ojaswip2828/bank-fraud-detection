-- Business analytics queries. Each block starts with "-- name: <title>" so
-- src/analytics.py can run them individually and save results.

-- name: overall_fraud_rate
SELECT COUNT(*)                                   AS total_txns,
       SUM(is_fraud)                              AS fraud_txns,
       ROUND(100.0 * AVG(is_fraud), 3)            AS fraud_rate_pct,
       ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount END), 2) AS fraud_value
FROM transactions;

-- name: fraud_by_channel
SELECT channel,
       COUNT(*)                          AS txns,
       SUM(is_fraud)                     AS fraud_txns,
       ROUND(100.0 * AVG(is_fraud), 3)   AS fraud_rate_pct
FROM transactions
GROUP BY channel
ORDER BY fraud_rate_pct DESC;

-- name: fraud_by_hour
SELECT CAST(strftime('%H', txn_time) AS INTEGER) AS hour,
       COUNT(*)                                  AS txns,
       SUM(is_fraud)                             AS fraud_txns,
       ROUND(100.0 * AVG(is_fraud), 3)           AS fraud_rate_pct
FROM transactions
GROUP BY hour
ORDER BY hour;

-- name: fraud_by_merchant_category
SELECT m.category,
       COUNT(*)                                   AS txns,
       SUM(t.is_fraud)                            AS fraud_txns,
       ROUND(SUM(CASE WHEN t.is_fraud = 1 THEN t.amount ELSE 0 END), 2) AS fraud_value
FROM transactions t
JOIN merchants m ON m.merchant_id = t.merchant_id
GROUP BY m.category
ORDER BY fraud_value DESC;

-- name: out_of_home_city_risk
SELECT CASE WHEN t.txn_city = c.home_city THEN 'home_city' ELSE 'other_city' END AS location,
       COUNT(*)                        AS txns,
       ROUND(100.0 * AVG(t.is_fraud), 3) AS fraud_rate_pct
FROM transactions t
JOIN customers c ON c.customer_id = t.customer_id
GROUP BY location;

-- name: top_customers_by_spend
SELECT customer_id,
       COUNT(*)              AS txns,
       ROUND(SUM(amount), 2) AS total_spend,
       ROUND(AVG(amount), 2) AS avg_txn
FROM transactions
WHERE is_fraud = 0
GROUP BY customer_id
ORDER BY total_spend DESC
LIMIT 10;

-- name: velocity_bursts
-- Customers with >= 5 transactions inside any 10-minute window (window function + self-range)
WITH ordered AS (
    SELECT customer_id, txn_time, is_fraud,
           LAG(txn_time, 4) OVER (PARTITION BY customer_id ORDER BY txn_time) AS t_minus_4
    FROM transactions
)
SELECT customer_id,
       txn_time AS burst_end,
       ROUND((julianday(txn_time) - julianday(t_minus_4)) * 24 * 60, 1) AS minutes_for_5_txns,
       is_fraud
FROM ordered
WHERE t_minus_4 IS NOT NULL
  AND (julianday(txn_time) - julianday(t_minus_4)) * 24 * 60 <= 10
ORDER BY minutes_for_5_txns
LIMIT 20;

-- name: amount_outliers_vs_customer_baseline
-- Transactions > 5x the customer's own median-ish (average) spend
WITH base AS (
    SELECT customer_id, AVG(amount) AS avg_amt
    FROM transactions
    GROUP BY customer_id
)
SELECT t.txn_id, t.customer_id, t.amount,
       ROUND(b.avg_amt, 2)              AS customer_avg,
       ROUND(t.amount / b.avg_amt, 1)   AS x_times_avg,
       t.is_fraud
FROM transactions t
JOIN base b ON b.customer_id = t.customer_id
WHERE t.amount > 5 * b.avg_amt
ORDER BY x_times_avg DESC
LIMIT 20;

-- name: monthly_trend
SELECT strftime('%Y-%m', txn_time)        AS month,
       COUNT(*)                           AS txns,
       ROUND(SUM(amount), 2)              AS volume,
       SUM(is_fraud)                      AS fraud_txns
FROM transactions
GROUP BY month
ORDER BY month;
