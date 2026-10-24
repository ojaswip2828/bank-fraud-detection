"""Generate a synthetic, realistic-ish bank transaction dataset with injected fraud.

Why synthetic? Real bank data is confidential. Public sets (Kaggle creditcard.csv)
use anonymised PCA features (V1..V28) that you can't explain in an interview.
Here every column has a business meaning, so every feature is defensible.

Fraud patterns injected (each mirrors a real BFSI fraud typology):
  1. Account takeover  - burst of high-value txns at night, new merchant/device, far from home
  2. Card testing      - many tiny txns in minutes at online merchants
  3. Unusual amount    - single txn many times the customer's normal spend
"""
import argparse
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

CITIES = ["Bengaluru", "Mumbai", "Delhi", "Chennai", "Hyderabad", "Pune", "Kolkata", "Jaipur"]
CATEGORIES = {
    "grocery": (300, 2500), "fuel": (500, 3000), "restaurant": (200, 3000),
    "electronics": (2000, 60000), "travel": (1500, 25000), "utilities": (500, 5000),
    "online_retail": (300, 8000), "atm_withdrawal": (500, 10000),
}
CHANNELS = ["POS", "ONLINE", "UPI", "ATM"]


def make_customers(n, rng):
    rows = []
    for cid in range(1, n + 1):
        rows.append({
            "customer_id": cid,
            "home_city": rng.choice(CITIES),
            "account_age_days": int(rng.integers(30, 3650)),
            "income_band": rng.choice(["low", "mid", "high"], p=[0.35, 0.45, 0.20]),
        })
    return pd.DataFrame(rows)


def make_merchants(n, rng):
    cats = list(CATEGORIES)
    return pd.DataFrame({
        "merchant_id": np.arange(1, n + 1),
        "category": rng.choice(cats, n),
        "city": rng.choice(CITIES, n),
    })


def normal_txns(customers, merchants, start, days, rng):
    rows = []
    mult = {"low": 0.6, "mid": 1.0, "high": 1.8}
    for c in customers.itertuples():
        # each customer has a small set of "usual" merchants, mostly in home city
        local = merchants[merchants.city == c.home_city]
        pool = local.sample(min(8, len(local)), random_state=int(rng.integers(1e9)))
        pool = pd.concat([pool, merchants.sample(2, random_state=int(rng.integers(1e9)))])
        n_txn = int(rng.poisson(days * 1.2))
        night_owl = rng.random() < 0.10          # some people genuinely shop at night
        for _ in range(n_txn):
            # legit noise so the problem isn't trivially separable:
            if rng.random() < 0.04:              # travel / one-off new merchant anywhere
                m = merchants.iloc[int(rng.integers(len(merchants)))]
            else:
                m = pool.iloc[int(rng.integers(len(pool)))]
            lo, hi = CATEGORIES[m.category]
            amt = rng.uniform(lo, hi) * mult[c.income_band] * rng.lognormal(0, 0.35)
            if rng.random() < 0.01:              # legit big purchase (phone, flight, rent)
                amt *= rng.uniform(4, 12)
            amt = round(float(amt), 2)
            if night_owl and rng.random() < 0.25:
                hour = int(rng.integers(0, 5))
            else:
                hour = int(np.clip(rng.normal(14, 4.5), 0, 23))
            ts = start + timedelta(days=int(rng.integers(days)), hours=hour,
                                   minutes=int(rng.integers(60)))
            channel = "ATM" if m.category == "atm_withdrawal" else (
                "ONLINE" if m.category == "online_retail" else rng.choice(["POS", "UPI"]))
            rows.append((c.customer_id, int(m.merchant_id), ts, amt, channel, m.city, 0, "none"))
        # legit bursts: splitting a bill / market run -> several small UPI txns in minutes
        for _ in range(int(rng.poisson(0.6))):
            t = start + timedelta(days=int(rng.integers(days)), hours=int(rng.integers(9, 22)))
            for _ in range(int(rng.integers(3, 7))):
                m = pool.iloc[int(rng.integers(len(pool)))]
                rows.append((c.customer_id, int(m.merchant_id), t,
                             round(float(rng.uniform(20, 600)), 2), "UPI", m.city, 0, "none"))
                t += timedelta(minutes=int(rng.integers(1, 8)))
    return rows


def fraud_txns(customers, merchants, start, days, rng, n_cases):
    rows = []
    victims = customers.sample(n_cases, random_state=int(rng.integers(1e9)))
    for c in victims.itertuples():
        kind = rng.choice(["takeover", "card_testing", "unusual_amount"], p=[0.4, 0.35, 0.25])
        t0 = start + timedelta(days=int(rng.integers(days)))
        if kind == "takeover":
            # fraudsters don't always look obvious: some act in daytime / victim's own city
            same_city = rng.random() < 0.3
            cand = merchants[merchants.category.isin(["electronics", "travel", "online_retail"])]
            cand = cand[(cand.city == c.home_city) if same_city else (cand.city != c.home_city)]
            hour = int(rng.integers(0, 5)) if rng.random() < 0.6 else int(rng.integers(9, 22))
            t = t0 + timedelta(hours=hour)
            for _ in range(int(rng.integers(1, 5))):
                m = cand.iloc[int(rng.integers(len(cand)))]
                amt = round(float(rng.uniform(5000, 90000)), 2)
                rows.append((c.customer_id, int(m.merchant_id), t, amt,
                             rng.choice(["ONLINE", "UPI"]), m.city, 1, kind))
                t += timedelta(minutes=int(rng.integers(2, 40)))
        elif kind == "card_testing":
            online = merchants[merchants.category == "online_retail"]
            t = t0 + timedelta(hours=int(rng.integers(0, 24)))
            for _ in range(int(rng.integers(3, 10))):
                m = online.iloc[int(rng.integers(len(online)))]
                amt = round(float(rng.uniform(1, 500)), 2)
                rows.append((c.customer_id, int(m.merchant_id), t, amt, "ONLINE", m.city, 1, kind))
                t += timedelta(minutes=int(rng.integers(1, 6)))
        else:
            m = merchants[merchants.category.isin(["electronics", "travel"])].sample(
                1, random_state=int(rng.integers(1e9))).iloc[0]
            t = t0 + timedelta(hours=int(rng.integers(8, 22)))
            amt = round(float(rng.uniform(25000, 250000)), 2)
            rows.append((c.customer_id, int(m.merchant_id), t, amt, rng.choice(CHANNELS[:3]),
                         m.city, 1, kind))
    return rows


def generate(n_customers=2000, n_merchants=400, days=90, fraud_cases=300, seed=42):
    rng = np.random.default_rng(seed)
    random.seed(seed)
    start = datetime(2026, 1, 1)
    customers = make_customers(n_customers, rng)
    merchants = make_merchants(n_merchants, rng)
    rows = normal_txns(customers, merchants, start, days, rng)
    rows += fraud_txns(customers, merchants, start, days, rng, fraud_cases)
    tx = pd.DataFrame(rows, columns=["customer_id", "merchant_id", "txn_time", "amount",
                                     "channel", "txn_city", "is_fraud", "fraud_type"])
    tx = tx.sort_values("txn_time").reset_index(drop=True)
    tx.insert(0, "txn_id", np.arange(1, len(tx) + 1))
    return customers, merchants, tx


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--customers", type=int, default=2000)
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--fraud-cases", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    cu, me, tx = generate(a.customers, 400, a.days, a.fraud_cases, a.seed)
    import os
    os.makedirs("data", exist_ok=True)
    cu.to_csv("data/customers.csv", index=False)
    me.to_csv("data/merchants.csv", index=False)
    tx.to_csv("data/transactions.csv", index=False)
    print(f"{len(tx):,} transactions | fraud rate {tx.is_fraud.mean():.3%}")
