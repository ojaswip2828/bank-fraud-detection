-- Normalised schema: customers 1-N transactions N-1 merchants
DROP TABLE IF EXISTS transactions;
DROP TABLE IF EXISTS customers;
DROP TABLE IF EXISTS merchants;

CREATE TABLE customers (
    customer_id       INTEGER PRIMARY KEY,
    home_city         TEXT    NOT NULL,
    account_age_days  INTEGER NOT NULL CHECK (account_age_days >= 0),
    income_band       TEXT    NOT NULL CHECK (income_band IN ('low','mid','high'))
);

CREATE TABLE merchants (
    merchant_id  INTEGER PRIMARY KEY,
    category     TEXT NOT NULL,
    city         TEXT NOT NULL
);

CREATE TABLE transactions (
    txn_id       INTEGER PRIMARY KEY,
    customer_id  INTEGER NOT NULL REFERENCES customers(customer_id),
    merchant_id  INTEGER NOT NULL REFERENCES merchants(merchant_id),
    txn_time     TEXT    NOT NULL,          -- ISO-8601
    amount       REAL    NOT NULL CHECK (amount > 0),
    channel      TEXT    NOT NULL CHECK (channel IN ('POS','ONLINE','UPI','ATM')),
    txn_city     TEXT    NOT NULL,
    is_fraud     INTEGER NOT NULL CHECK (is_fraud IN (0,1)),  -- label: never used as a feature
    fraud_type   TEXT                                         -- analysis only
);

-- Indexes for the window/feature queries (customer timeline lookups)
CREATE INDEX idx_txn_customer_time ON transactions(customer_id, txn_time);
CREATE INDEX idx_txn_merchant      ON transactions(merchant_id);
