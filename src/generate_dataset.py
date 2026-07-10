"""
Synthetic telecom customer dataset generator
----------------------------------------------
Produces two linked CSV files:
  customers.csv - one row per customer (profile + real-style attributes)
  events.csv    - one row per interaction (support ticket, billing event, or CSAT survey),
                   spread across each customer's tenure, with actual calendar dates

Design notes:
- planted_outcome and hidden risk_level are the "ground truth" used to condition event
  generation. They are NOT meant to be fed into your detection pipeline as input features -
  hold them back and use them afterward to sanity-check whether your risk scores correctly
  rank planted high-risk / churned customers above planted low-risk / retained ones.
- Noise is deliberately added so outcomes aren't perfectly predictable from risk_level alone -
  this creates genuinely ambiguous cases (false alarms, surprise saves) for a more convincing demo.
- No API calls / no cost - all text is generated from hand-written template pools with
  randomised combination, so it's free to (re)run as many times as you like.
- Event dates are anchored to "today" (the data snapshot date) and each customer's join_date
  is derived backwards from their tenure_months, so events fall at realistic calendar dates
  within that customer's actual relationship window.
"""

import random
import pandas as pd
import numpy as np
from datetime import timedelta

# ---- reproducibility ----
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

N_CUSTOMERS = 180

# "today" for the purposes of this dataset - all tenure/dates are calculated backwards from this
SNAPSHOT_DATE = pd.Timestamp("2026-07-09")

# ---------------------------------------------------------------------------
# 1. CUSTOMER PROFILES
# ---------------------------------------------------------------------------

plan_types = ["Month-to-month", "One year", "Two year"]
payment_methods = ["Direct debit", "Credit card", "Bank transfer", "Mailed cheque"]
regions = ["VIC", "NSW", "QLD", "WA", "SA"]

# hidden risk_level drives event generation but is not a perfect predictor of outcome
risk_levels = ["Low", "Medium", "High"]
risk_probs_by_outcome = {
    "Churned":  [0.05, 0.30, 0.65],   # mostly high risk, but not all
    "Retained": [0.65, 0.28, 0.07],   # mostly low risk, some false-alarm-looking cases
}

customers = []
for cid in range(1001, 1001 + N_CUSTOMERS):
    tenure_months = int(np.clip(np.random.exponential(scale=24), 1, 72))

    # month-to-month customers are realistically shorter tenure / higher underlying churn base rate
    plan_type = random.choices(plan_types, weights=[0.55, 0.25, 0.20])[0]

    base_charge = {"Month-to-month": 75, "One year": 65, "Two year": 55}[plan_type]
    monthly_charge = round(base_charge + np.random.normal(0, 12), 2)
    monthly_charge = max(25, monthly_charge)

    payment_method = random.choices(
        payment_methods, weights=[0.45, 0.30, 0.15, 0.10]
    )[0]
    region = random.choice(regions)

    # outcome probability skews with plan type and payment method (mirrors real churn drivers)
    base_churn_p = {"Month-to-month": 0.38, "One year": 0.18, "Two year": 0.08}[plan_type]
    if payment_method == "Mailed cheque":
        base_churn_p += 0.05  # weak manual payment methods correlate with disengagement
    planted_outcome = "Churned" if random.random() < base_churn_p else "Retained"

    risk_level = random.choices(risk_levels, weights=risk_probs_by_outcome[planted_outcome])[0]

    join_date = SNAPSHOT_DATE - pd.DateOffset(months=int(tenure_months))

    customers.append({
        "customer_id": cid,
        "join_date": join_date.date().isoformat(),
        "tenure_months": tenure_months,
        "plan_type": plan_type,
        "monthly_charge": monthly_charge,
        "payment_method": payment_method,
        "region": region,
        "risk_level": risk_level,          # hidden ground truth - exclude from pipeline input
        "planted_outcome": planted_outcome  # hidden ground truth - exclude from pipeline input
    })

customers_df = pd.DataFrame(customers)

# ---------------------------------------------------------------------------
# 2. EVENT TEXT / CONTENT POOLS
# ---------------------------------------------------------------------------

issue_topics = [
    "a billing error", "dropped calls", "slow data speeds", "an unexpected price increase",
    "the app crashing repeatedly", "a network outage in their area", "confusion over plan inclusions",
    "roaming charges from a recent trip", "long wait times reaching support", "unclear contract terms",
    "a delayed refund", "trouble accessing their account online", "a faulty modem/router",
    "an incorrect charge on their statement", "coverage issues at their new address",
]

support_ticket_templates = {
    "Low": [
        "Asked a quick question about {issue}, resolved without any fuss.",
        "Reached out about {issue} - nothing urgent, just wanted clarity.",
        "Mentioned {issue} in passing, seemed satisfied once explained.",
        "Called about {issue}, said the support experience was quick and easy.",
        "Followed up on {issue} from a prior chat, all sorted now.",
        "Asked how {issue} would affect their next bill, no complaints raised.",
    ],
    "Medium": [
        "Raised {issue} and wasn't fully satisfied with how it was handled.",
        "Called about {issue} for the second time this month.",
        "Queried {issue} and asked whether a better plan was available.",
        "Mentioned {issue}, and referenced a competitor's cheaper offer.",
        "Reported {issue}, said the first fix attempt didn't fully work.",
        "Asked about {issue}, sounded a little frustrated but stayed polite.",
    ],
    "High": [
        "Called again about {issue} - this is the third time it hasn't been resolved.",
        "Expressed real frustration over {issue}, said they're at their limit.",
        "Asked directly what it would take to cancel after ongoing {issue}.",
        "Said they're actively comparing providers because of {issue}.",
        "Reported {issue}, said they feel like a long-standing customer isn't valued.",
        "Escalated {issue} after two unresolved prior contacts.",
    ],
}

def generate_support_ticket(risk):
    template = random.choice(support_ticket_templates[risk])
    issue = random.choice(issue_topics)
    return template.format(issue=issue)

csat_score_ranges = {
    "Low": (7, 10),
    "Medium": (4, 7),
    "High": (1, 5),
}

billing_events_pool = {
    "Low": ["on_time_payment", "on_time_payment", "on_time_payment", "plan_upgrade"],
    "Medium": ["on_time_payment", "late_payment", "on_time_payment", "plan_change"],
    "High": ["missed_payment", "late_payment", "plan_downgrade", "missed_payment"],
}

# ---------------------------------------------------------------------------
# 3. EVENTS - spread across each customer's tenure
# ---------------------------------------------------------------------------

events = []
event_id = 5001

for _, cust in customers_df.iterrows():
    cid = cust["customer_id"]
    tenure = cust["tenure_months"]
    risk = cust["risk_level"]
    join_date = pd.Timestamp(cust["join_date"])

    # 3-12 events per customer, scaled loosely by tenure but not capped by it -
    # short-tenure customers can still have several events packed into a short window
    n_events = int(np.clip(round(tenure / 5), 3, 12))

    # sample a day offset for each event, somewhere between join_date and snapshot_date
    total_days = max((SNAPSHOT_DATE - join_date).days, 1)
    day_offsets = sorted(np.random.choice(
        range(0, total_days + 1),
        size=n_events,
        replace=True  # allow multiple events in the same month for short-tenure customers
    ))

    for offset in day_offsets:
        event_date = join_date + timedelta(days=int(offset))

        signal_type = random.choices(
            ["support_ticket", "billing_event", "csat_survey"],
            weights=[0.4, 0.35, 0.25]
        )[0]

        # avoid two CSAT surveys landing on the exact same day for the same customer -
        # a customer wouldn't realistically submit two satisfaction surveys on one date
        if signal_type == "csat_survey":
            already_used_dates = {
                ev["event_date"] for ev in events
                if ev["customer_id"] == cid and ev["signal_type"] == "csat_survey"
            }
            if event_date.date().isoformat() in already_used_dates:
                signal_type = random.choice(["support_ticket", "billing_event"])

        if signal_type == "support_ticket":
            detail = generate_support_ticket(risk)
        elif signal_type == "billing_event":
            detail = random.choice(billing_events_pool[risk])
        else:  # csat_survey
            low, high = csat_score_ranges[risk]
            detail = str(random.randint(low, high))

        events.append({
            "event_id": event_id,
            "customer_id": cid,
            "event_date": event_date.date().isoformat(),
            "signal_type": signal_type,
            "detail": detail,
        })
        event_id += 1

events_df = pd.DataFrame(events).sort_values(["customer_id", "event_date"]).reset_index(drop=True)

# ---------------------------------------------------------------------------
# 4. SAVE OUTPUT
# ---------------------------------------------------------------------------

# Full versions (with hidden ground truth) - for YOUR validation use only
customers_df.to_csv("data/customers_with_ground_truth.csv", index=False)
events_df.to_csv("data/events.csv", index=False)

# Pipeline-input version - ground truth columns stripped out, this is what your
# detection pipeline should actually ingest, so it isn't "cheating" off the answer key
customers_input_df = customers_df.drop(columns=["risk_level", "planted_outcome"])
customers_input_df.to_csv("data/customers_pipeline_input.csv", index=False)

print("Customers generated:", len(customers_df))
print("Events generated:", len(events_df))
print("\nOutcome distribution:")
print(customers_df["planted_outcome"].value_counts())
print("\nRisk level distribution:")
print(customers_df["risk_level"].value_counts())
print("\nEvents per customer - min/median/max:",
      events_df.groupby("customer_id").size().min(),
      events_df.groupby("customer_id").size().median(),
      events_df.groupby("customer_id").size().max())
print("\nSample customer:")
print(customers_df.head(3).to_string(index=False))
print("\nSample events for customer", customers_df.iloc[0]["customer_id"], ":")
print(events_df[events_df["customer_id"] == customers_df.iloc[0]["customer_id"]].to_string(index=False))
