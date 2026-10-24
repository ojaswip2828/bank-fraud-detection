-- Feature engineering done IN SQL (window functions).
-- Rule: every feature for a txn uses ONLY that customer's PAST transactions
-- (ROWS ... PRECEDING / 1 PRECEDING) -> no data leakage from the future or the label.
WITH base AS (
    SELECT t.txn_id, t.customer_id, t.txn_time, t.amount, t.channel,
           t.merchant_id, m.category, t.txn_city, c.home_city,
           c.account_age_days, c.income_band, t.is_fraud,
           julianday(t.txn_time) AS jd
    FROM transactions t
    JOIN customers c ON c.customer_id = t.customer_id
    JOIN merchants m ON m.merchant_id = t.merchant_id
),
win AS (
    SELECT b.*,
        -- customer's historical average amount (excluding current txn)
        AVG(amount) OVER (PARTITION BY customer_id ORDER BY jd, txn_id
                          ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS hist_avg_amt,
        COUNT(*)    OVER (PARTITION BY customer_id ORDER BY jd, txn_id
                          ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS hist_txn_count,
        -- time since previous txn (minutes)
        (jd - LAG(jd) OVER (PARTITION BY customer_id ORDER BY jd, txn_id)) * 1440       AS mins_since_prev,
        -- velocity: txns in the last 1 hour (range on julian day; 1h = 1/24)
        COUNT(*) OVER (PARTITION BY customer_id ORDER BY jd
                       RANGE BETWEEN 0.0416667 PRECEDING AND CURRENT ROW)       AS txns_last_1h,
        SUM(amount) OVER (PARTITION BY customer_id ORDER BY jd
                       RANGE BETWEEN 1.0 PRECEDING AND CURRENT ROW)            AS amt_last_24h,
        -- first time this customer uses this merchant?
        ROW_NUMBER() OVER (PARTITION BY customer_id, merchant_id ORDER BY jd, txn_id)  AS merchant_visit_no
    FROM base b
)
SELECT txn_id,
       customer_id,
       txn_time,
       amount,
       COALESCE(amount / NULLIF(hist_avg_amt, 0), 1.0)              AS amt_to_hist_avg,
       COALESCE(hist_txn_count, 0)                                  AS hist_txn_count,
       COALESCE(mins_since_prev, 10000)                             AS mins_since_prev,
       txns_last_1h,
       amt_last_24h,
       CASE WHEN merchant_visit_no = 1 THEN 1 ELSE 0 END            AS is_new_merchant,
       CASE WHEN txn_city <> home_city THEN 1 ELSE 0 END            AS is_out_of_city,
       CAST(strftime('%H', txn_time) AS INTEGER)                    AS hour,
       CASE WHEN CAST(strftime('%H', txn_time) AS INTEGER) < 5 THEN 1 ELSE 0 END AS is_night,
       account_age_days,
       income_band,
       channel,
       category,
       is_fraud
FROM win
ORDER BY txn_time, txn_id;
