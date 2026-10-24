"""Generate a synthetic, realistic-ish bank transaction dataset with injected fraud.

Why synthetic? Real bank data is confidential. Public sets (Kaggle creditcard.csv)
use anonymised PCA features (V1..V28) that you can't explain in an interview.
Here every column has a business meaning, so every feature is defensible.

Fraud patterns injected (each mirrors a real BFSI fraud typology):
  1. Account takeover  - burst of high-value txns, often at night / far from home / new merchant
  2. Card testing      - many tiny txns in minutes, mostly online
  3. Unusual amount    - single txn many times the customer's normal spend

The generator deliberately avoids "giveaway" rules. Fraud can hit any merchant
category (including grocery/fuel gift-card and fuel-card fraud), can reuse the
victim's own usual merchants (stolen phone / session hijack), and legit big
purchases come from the same amount distribution as "unusual amount" fraud.
A model therefore can't just memorise one column; it has to combine signals.
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
INCOME_MULT = {"low": 0.6, "mid": 1.0, "high": 1.8}


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


def pick_channel(category, rng):
    """Legit channel mix: ATM/online_retail are fixed, big-ticket categories are often online."""
    if category == "atm_withdrawal":
        return "ATM"
    if category == "online_retail":
        return "ONLINE"
    if category in ("travel", "utilities", "electronics") and rng.random() < 0.3:
        return "ONLINE"
    return rng.choice(["POS", "UPI"])


def big_purchase_amount(category, income_band, rng):
    """Amount far above a customer's norm. Shared by legit big buys and 'unusual amount'
    fraud so amount alone can't separate them."""
    lo, hi = CATEGORIES[category]
    return rng.uniform(lo, hi) * INCOME_MULT[income_band] * rng.lognormal(0, 0.35) * rng.uniform(4, 12)


def normal_txns(customers, merchants, start, days, rng):
    rows, pools = [], {}
    big_cats = merchants[merchants.category.isin(["electronics", "travel", "grocery", "online_retail"])]
    for c in customers.itertuples():
        # each customer has a small set of "usual" merchants, mostly in home city
        local = merchants[merchants.city == c.home_city]
        pool = local.sample(min(8, len(local)), random_state=int(rng.integers(1e9)))
        pool = pd.concat([pool, merchants.sample(2, random_state=int(rng.integers(1e9)))])
        pools[c.customer_id] = pool
        n_txn = int(rng.poisson(days * 1.2))
        night_owl = rng.random() < 0.10          # some people genuinely shop at night
        for _ in range(n_txn):
            # legit noise so the problem isn't trivially separable:
            if rng.random() < 0.06:              # travel / one-off new merchant anywhere
                m = merchants.iloc[int(rng.integers(len(merchants)))]
            else:
                m = pool.iloc[int(rng.integers(len(pool)))]
            lo, hi = CATEGORIES[m.category]
            amt = rng.uniform(lo, hi) * INCOME_MULT[c.income_band] * rng.lognormal(0, 0.35)
            if rng.random() < 0.015:             # legit big purchase (phone, flight, rent)
                if rng.random() < 0.4:           # ...often at a shop never visited before
                    m = big_cats.iloc[int(rng.integers(len(big_cats)))]
                amt = big_purchase_amount(m.category, c.income_band, rng)
            amt = round(float(amt), 2)
            if night_owl and rng.random() < 0.25:
                hour = int(rng.integers(0, 5))
            else:
                hour = int(np.clip(rng.normal(14, 4.5), 0, 23))
            ts = start + timedelta(days=int(rng.integers(days)), hours=hour,
                                   minutes=int(rng.integers(60)))
            rows.append((c.customer_id, int(m.merchant_id), ts, amt, pick_channel(m.category, rng),
                         m.city, 0, "none"))
        # legit bursts: splitting a bill / market run -> several small UPI txns in minutes
        for _ in range(int(rng.poisson(0.6))):
            t = start + timedelta(days=int(rng.integers(days)), hours=int(rng.integers(9, 22)))
            for _ in range(int(rng.integers(3, 7))):
                m = pool.iloc[int(rng.integers(len(pool)))]
                rows.append((c.customer_id, int(m.merchant_id), t,
                             round(float(rng.uniform(20, 600)), 2), "UPI", m.city, 0, "none"))
                t += timedelta(minutes=int(rng.integers(1, 8)))
    return rows, pools


def fraud_txns(customers, merchants, pools, start, days, rng, n_cases):
    rows = []
    victims = customers.sample(n_cases, random_state=int(rng.integers(1e9)))
    for c in victims.itertuples():
        kind = rng.choice(["takeover", "card_testing", "unusual_amount"], p=[0.4, 0.35, 0.25])
        pool = pools[c.customer_id]
        t0 = start + timedelta(days=int(rng.integers(days)))
        if kind == "takeover":
            if rng.random() < 0.35:
                # stolen phone / hijacked session: fraudster reuses the victim's usual merchants
                cand = pool
            else:
                # resellable goods: electronics, travel, gift cards & fuel cards at grocery/fuel
                cand = merchants[merchants.category.isin(
                    ["electronics", "travel", "online_retail", "grocery", "fuel"])]
                # fraudsters don't always look obvious: some act in the victim's own city
                same_city = rng.random() < 0.3
                cand = cand[(cand.city == c.home_city) if same_city else (cand.city != c.home_city)]
            hour = int(rng.integers(0, 5)) if rng.random() < 0.5 else int(rng.integers(8, 23))
            t = t0 + timedelta(hours=hour, minutes=int(rng.integers(60)))
            for _ in range(int(rng.integers(1, 5))):
                m = cand.iloc[int(rng.integers(len(cand)))]
                lo, hi = CATEGORIES[m.category]
                amt = round(float(rng.uniform(0.6 * hi, 2.5 * hi) * INCOME_MULT[c.income_band]), 2)
                channel = pick_channel(m.category, rng)
                if channel in ("POS", "UPI") and rng.random() < 0.4:
                    channel = "ONLINE"           # card-not-present is the easiest route
                rows.append((c.customer_id, int(m.merchant_id), t, amt, channel, m.city, 1, kind))
                t += timedelta(minutes=int(rng.integers(2, 40)))
        elif kind == "card_testing":
            if rng.random() < 0.7:
                cand, channel = merchants[merchants.category == "online_retail"], "ONLINE"
            else:                                # small-ticket UPI/POS probes at everyday merchants
                cand = merchants[merchants.category.isin(["grocery", "fuel", "restaurant"])]
                channel = None
            t = t0 + timedelta(hours=int(rng.integers(0, 24)), minutes=int(rng.integers(60)))
            for _ in range(int(rng.integers(3, 9))):
                m = cand.iloc[int(rng.integers(len(cand)))]
                amt = round(float(rng.uniform(1, 300)), 2)
                rows.append((c.customer_id, int(m.merchant_id), t, amt,
                             channel or rng.choice(["UPI", "POS"]), m.city, 1, kind))
                t += timedelta(minutes=int(rng.integers(1, 10)))
        else:
            if rng.random() < 0.5:               # big spend at one of the victim's usual shops
                m = pool.iloc[int(rng.integers(len(pool)))]
                if m.category == "atm_withdrawal":
                    m = merchants[merchants.category == "electronics"].iloc[0]
            else:
                cand = merchants[merchants.category.isin(
                    ["electronics", "travel", "online_retail", "grocery"])]
                m = cand.iloc[int(rng.integers(len(cand)))]
            t = t0 + timedelta(hours=int(rng.integers(8, 22)), minutes=int(rng.integers(60)))
            amt = round(float(big_purchase_amount(m.category, c.income_band, rng)), 2)
            rows.append((c.customer_id, int(m.merchant_id), t, amt, pick_channel(m.category, rng),
                         m.city, 1, kind))
    return rows


def generate(n_customers=2000, n_merchants=400, days=90, fraud_cases=300, seed=42):
    rng = np.random.default_rng(seed)
    random.seed(seed)
    start = datetime(2026, 1, 1)
    customers = make_customers(n_customers, rng)
    merchants = make_merchants(n_merchants, rng)
    rows, pools = normal_txns(customers, merchants, start, days, rng)
    rows += fraud_txns(customers, merchants, pools, start, days, rng, fraud_cases)
    tx = pd.DataFrame(rows, columns=["customer_id", "merchant_id", "txn_time", "amount",
                                     "channel", "txn_city", "is_fraud", "fraud_type"])
    # stable sort + txn_id tiebreak keeps same-timestamp rows in a deterministic order
    tx = tx.sort_values("txn_time", kind="stable").reset_index(drop=True)
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
