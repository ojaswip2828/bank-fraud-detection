"""Fraud-scoring REST API.

  uvicorn app:app --app-dir src --port 8000      ->  http://localhost:8000/docs

POST /score takes a raw transaction (no features), builds features from the
customer's history with sql/features.sql, and returns a calibrated fraud
probability, the alert decision and the top SHAP reasons.
"""
import os
from datetime import datetime
from typing import Literal

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

import db
import scoring

MODEL = os.environ.get("MODEL", "lightgbm")
MODEL_DIR = os.environ.get("MODEL_DIR", db.ROOT / "models")
DB_PATH = os.environ.get("DB_PATH", db.DB_PATH)
bundle = joblib.load(os.path.join(MODEL_DIR, f"{MODEL}.joblib"))

app = FastAPI(
    title="Bank Fraud Detection API",
    description="Scores a card/UPI transaction against the customer's history. "
                "Customers 1-2000 and merchants 1-400 exist in the demo database "
                "(Jan-Mar 2026). Try the examples on POST /score.",
    version="1.0",
)


class Transaction(BaseModel):
    customer_id: int = Field(ge=1)
    merchant_id: int = Field(ge=1)
    amount: float = Field(gt=0, description="INR")
    txn_time: datetime
    channel: Literal["POS", "ONLINE", "UPI", "ATM"]
    txn_city: str

    model_config = {"json_schema_extra": {"examples": [
        # routine: Kolkata customer, their usual grocery store, evening UPI
        {"customer_id": 1411, "merchant_id": 400, "amount": 1450.0,
         "txn_time": "2026-03-31T19:05:00", "channel": "UPI", "txn_city": "Kolkata"},
        # card-testing pattern: tiny online payment, 3am, never-used merchant in another city
        {"customer_id": 1411, "merchant_id": 35, "amount": 49.0,
         "txn_time": "2026-03-31T03:12:00", "channel": "ONLINE", "txn_city": "Delhi"},
    ]}}


class Reason(BaseModel):
    feature: str
    value: float | int | str
    contribution: float


class Score(BaseModel):
    fraud_probability: float
    flagged: bool
    threshold: float
    model: str
    reasons: list[Reason]
    features: dict[str, float | int | str]


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL, "threshold": round(bundle["threshold"], 4)}


@app.post("/score", response_model=Score)
def score(txn: Transaction):
    con = scoring.connect_readonly(DB_PATH)   # per request: the temp table is per connection
    try:
        out = scoring.score(con, bundle, txn.model_dump())
    except scoring.UnknownEntity as e:
        raise HTTPException(status_code=404, detail=str(e))
    finally:
        con.close()
    return {**out, "model": MODEL}
