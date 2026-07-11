"""
Customer risk scoring
------------------------
Combines four signal categories - CSAT, sentiment, churn intent, and billing -
into a single 0-100 risk score per customer.

METHOD (same for all four categories):
  1. For each customer, gather their events for that category, sorted most
     recent first.
  2. Assign each event a rank-based weight: rank 1 (most recent) gets weight
     r^0, rank 2 gets r^1, rank 3 gets r^2, etc. (r = DECAY_RATIO, e.g. 0.6).

     NOTE - this is RANK-based recency, not TIME-based recency. Two events
     one day apart and two events two years apart receive the exact same
     relative weighting, as long as they occupy the same two rank positions
     (e.g. most-recent and second-most-recent). This is a deliberate
     simplification: a single shared DECAY_RATIO can be applied uniformly
     across all four categories without needing a separate, harder-to-justify
     half-life constant tuned per category. The trade-off is that this
     pipeline cannot distinguish "customer went quiet for a week" from
     "customer went quiet for two years" - only their relative event order
     matters, not the actual elapsed time. For a production version, a
     time-based (e.g. exponential half-life) decay would be a reasonable
     refinement once there's a real basis for tuning per-category half-lives.

  3. Normalise those weights to sum to 1, and take the weighted average of
     the event values. This is a genuine average (not a decayed sum), so a
     customer with a long, all-positive history doesn't get penalised just
     because their events are old - only actual recent negative evidence
     pulls the score toward risk. This also means an old warning sign can
     fade in influence if the customer's more recent behaviour has improved
     ("redemption").
  4. Each category's weighted average is rescaled to a common 0-100 RISK
     scale, where 100 = highest risk, 0 = no risk, regardless of the
     category's original units.
  5. The four category risk scores are combined using hand-picked category
     weights into one final risk_score per customer. See CATEGORY_WEIGHTS
     below for the weighting and the reasoning behind each value.

Customers with no events in a given category are excluded from that
category's contribution, and the remaining category weights are
renormalised - so a customer with e.g. no billing events isn't penalised
or rewarded for a category with no evidence either way.

Each customer's output also records which categories were actually present
vs missing (categories_present / categories_missing), and a data_coverage
label (Low/Medium/High) based on how many of the four categories had any
evidence at all. IMPORTANT: this label describes how much EVIDENCE was
available, not how confident the model is that its prediction is correct -
those are different things, and the naming is deliberately "coverage" rather
than "confidence" to avoid implying a predictive-certainty claim this
pipeline doesn't actually make. Customers with Low coverage get a
suggested_data_action recommending proactive outreach (e.g. a CSAT
check-in) rather than a retention-specific action, since their risk score
is based on very little evidence and shouldn't be acted on with the same
weight as a fully-evidenced score.

This category/coverage information is carried forward into the output
table specifically so the downstream rationale-generation LLM step has
enough context to distinguish "this customer looks fine" from "we simply
have no data for this customer" - a distinction the risk score alone
cannot convey.

Ground truth (risk_level, planted_outcome) is NEVER used as an input here -
it's read separately, at the end, purely to validate the resulting scores.

PERFORMANCE NOTE: compute_customer_risk() currently filters the full
events/sentiment/intent dataframes once per customer, rather than
pre-grouping by customer_id upfront. At this project's scale (~180
customers, ~900 events) this runs in well under a second, so it hasn't
been optimised. For a production deployment scoring a much larger customer
base, pre-grouping (e.g. via groupby) before the per-customer loop would
be the appropriate next step to avoid repeated full-table scans.
"""

import pandas as pd
import numpy as np

DECAY_RATIO = 0.6

# Category weights - hand-picked, not learned from data (see README for the
# option of validating/tuning these against planted_outcome as a future step).
# Reasoning behind the ordering:
#   CSAT (30%)      - the customer's own DIRECT, self-reported statement of
#                      satisfaction. No other category is stated outright by
#                      the customer; everything else is inferred or behavioural.
#   Intent (25%)    - an explicit, forward-looking signal when present
#                      (e.g. "comparing providers"), and the redemption-aware
#                      rank-weighted average means an old signal fades if not
#                      repeated - a strong, if inferred, indicator.
#   Billing (25%)   - revealed behaviour rather than a stated opinion. A
#                      single missed payment could be a one-off (e.g. simple
#                      forgetfulness) rather than dissatisfaction, so it's
#                      NOT weighted above CSAT/intent - but sustained or
#                      recent billing trouble is hard to explain away as
#                      coincidence, and revealed behaviour is not subject to
#                      the same social-desirability bias that can make survey
#                      responses (CSAT) overly polite relative to actual
#                      future behaviour ("silent churn").
#   Sentiment (20%) - the weakest signal of the four. Lexicon-based tone
#                      scoring (VADER) was directly observed, during this
#                      project's build, to miss real risk when a customer
#                      states a churn-relevant fact calmly (e.g. "comparing
#                      providers" scored as mildly POSITIVE by VADER alone).
#                      Kept in the model as a useful supporting signal, but
#                      weighted lowest given this known reliability gap.
CATEGORY_WEIGHTS = {
    "csat": 0.30,
    "intent": 0.25,
    "billing": 0.25,
    "sentiment": 0.20,
}

# Billing event severity mapping (0 = no risk, 1 = maximum risk within this
# category). Hand-picked based on how unambiguous a signal each event type is:
#   on_time_payment / plan_upgrade -> 0.0  (positive/neutral, no risk)
#   plan_change     -> 0.3  (a lateral move; mildly ambiguous, so scored low)
#   late_payment    -> 0.6  (a real but recoverable friction point)
#   plan_downgrade  -> 0.7  (customer actively reducing spend/commitment)
#   missed_payment  -> 1.0  (the most concrete, unambiguous negative event)
BILLING_SEVERITY = {
    "on_time_payment": 0.0,
    "plan_upgrade": 0.0,
    "plan_change": 0.3,
    "late_payment": 0.6,
    "plan_downgrade": 0.7,
    "missed_payment": 1.0,
}


def rank_weighted_average(dated_values, r=DECAY_RATIO):
    """
    dated_values: list of (date, value) tuples for ONE customer, ONE category.
    Returns the rank-weighted average value, or None if the list is empty.
    """
    if not dated_values:
        return None

    # sort most recent first
    sorted_vals = sorted(dated_values, key=lambda x: x[0], reverse=True)
    values = [v for _, v in sorted_vals]

    raw_weights = [r ** i for i in range(len(values))]
    total_weight = sum(raw_weights)
    norm_weights = [w / total_weight for w in raw_weights]

    weighted_avg = sum(v * w for v, w in zip(values, norm_weights))
    return weighted_avg


def csat_to_risk(weighted_avg_csat):
    """CSAT is 1-10, higher = better. Convert to 0-100 risk (higher = riskier)."""
    return (10 - weighted_avg_csat) / 9 * 100


def sentiment_to_risk(weighted_avg_sentiment):
    """Sentiment compound score is -1 to +1, higher = better. Convert to 0-100 risk."""
    return (1 - weighted_avg_sentiment) / 2 * 100


def intent_to_risk(weighted_avg_intent):
    """Intent score is already 0-1, higher = riskier. Just rescale to 0-100."""
    return weighted_avg_intent * 100


def billing_to_risk(weighted_avg_severity):
    """Billing severity is already 0-1, higher = riskier. Just rescale to 0-100."""
    return weighted_avg_severity * 100


def compute_customer_risk(customer_id, events_df, sentiment_df, intent_df):
    # --- CSAT ---
    csat_events = events_df[
        (events_df["customer_id"] == customer_id) & (events_df["signal_type"] == "csat_survey")
    ]
    csat_dated_values = list(zip(csat_events["event_date"], csat_events["detail"].astype(float)))
    csat_weighted = rank_weighted_average(csat_dated_values)

    # --- Billing ---
    billing_events = events_df[
        (events_df["customer_id"] == customer_id) & (events_df["signal_type"] == "billing_event")
    ]
    billing_dated_values = [
        (row["event_date"], BILLING_SEVERITY.get(row["detail"], 0.0))
        for _, row in billing_events.iterrows()
    ]
    billing_weighted = rank_weighted_average(billing_dated_values)

    # --- Sentiment (from ticket_sentiment.csv) ---
    sent_rows = sentiment_df[sentiment_df["customer_id"] == customer_id]
    sent_dated_values = list(zip(sent_rows["event_date"], sent_rows["compound_score"].astype(float)))
    sentiment_weighted = rank_weighted_average(sent_dated_values)

    # --- Intent (from ticket_intent.csv) ---
    intent_rows = intent_df[intent_df["customer_id"] == customer_id]
    intent_dated_values = list(zip(intent_rows["event_date"], intent_rows["intent_score"].astype(float)))
    intent_weighted = rank_weighted_average(intent_dated_values)

    # --- Convert each available category to a 0-100 risk score ---
    category_risks = {}
    if csat_weighted is not None:
        category_risks["csat"] = csat_to_risk(csat_weighted)
    if sentiment_weighted is not None:
        category_risks["sentiment"] = sentiment_to_risk(sentiment_weighted)
    if intent_weighted is not None:
        category_risks["intent"] = intent_to_risk(intent_weighted)
    if billing_weighted is not None:
        category_risks["billing"] = billing_to_risk(billing_weighted)

    # --- Combine with category weights, renormalised over available categories ---
    if not category_risks:
        # No evidence in ANY category - this is genuinely unknown risk, not
        # zero risk. Returning 0.0 here would silently rank a total-unknown
        # customer as equivalent to "definitely safe", which is incorrect -
        # so we return None and handle this explicitly in main().
        return None, category_risks

    available_weight_sum = sum(CATEGORY_WEIGHTS[c] for c in category_risks)
    final_score = sum(
        category_risks[c] * (CATEGORY_WEIGHTS[c] / available_weight_sum)
        for c in category_risks
    )

    return final_score, category_risks


def main():
    customers = pd.read_csv("data/customers_pipeline_input.csv")
    events = pd.read_csv("data/events.csv")
    sentiment = pd.read_csv("data/ticket_sentiment.csv")
    intent = pd.read_csv("data/ticket_intent.csv")

    results = []
    for _, cust in customers.iterrows():
        cid = cust["customer_id"]
        score, cat_risks = compute_customer_risk(cid, events, sentiment, intent)

        categories_present = sorted(cat_risks.keys())
        n_categories = len(categories_present)

        if n_categories == 0:
            data_coverage = "No data"
        elif n_categories == 1:
            data_coverage = "Low"
        elif n_categories in (2, 3):
            data_coverage = "Medium"
        else:
            data_coverage = "High"

        missing_categories = sorted(set(CATEGORY_WEIGHTS.keys()) - set(cat_risks.keys()))

        suggested_data_action = None
        if data_coverage in ("No data", "Low"):
            if "csat" in missing_categories:
                suggested_data_action = "Insufficient signal for confident assessment - recommend proactive CSAT outreach"
            else:
                suggested_data_action = "Insufficient signal for confident assessment - recommend proactive check-in"

        results.append({
            "customer_id": cid,
            # risk_score is left as None/NaN (not 0.0) when there's zero
            # evidence - this customer's risk is genuinely UNKNOWN, not
            # confirmed-safe, and should not be sorted to the bottom of a
            # risk-ranked list as if they were the lowest-risk customer.
            "risk_score": round(score, 1) if score is not None else None,
            "data_coverage": data_coverage,
            "categories_present": ",".join(categories_present) if categories_present else "none",
            "categories_missing": ",".join(missing_categories) if missing_categories else "none",
            "suggested_data_action": suggested_data_action,
            "csat_risk": round(cat_risks.get("csat", np.nan), 1) if "csat" in cat_risks else None,
            "sentiment_risk": round(cat_risks.get("sentiment", np.nan), 1) if "sentiment" in cat_risks else None,
            "intent_risk": round(cat_risks.get("intent", np.nan), 1) if "intent" in cat_risks else None,
            "billing_risk": round(cat_risks.get("billing", np.nan), 1) if "billing" in cat_risks else None,
        })

    output = pd.DataFrame(results)
    # sort risk-scored customers highest-first; None-score (unknown) customers
    # go to their own separate group at the end, not silently treated as 0
    output = output.sort_values("risk_score", ascending=False, na_position="last").reset_index(drop=True)
    output.to_csv("data/risk_scores.csv", index=False)

    print("Risk scores computed for", len(output), "customers")
    n_unknown = output["risk_score"].isna().sum()
    if n_unknown > 0:
        print(f"NOTE: {n_unknown} customer(s) had zero evidence in any category - "
              f"risk_score left as unknown (None), not scored as 0/safe.")
    print("\nScore distribution (scored customers only):")
    print(output["risk_score"].describe())
    print("\nData coverage distribution:")
    print(output["data_coverage"].value_counts())
    print("\nTop 10 highest risk:")
    print(output[["customer_id","risk_score","data_coverage","categories_present"]].head(10).to_string(index=False))
    print("\nLow/No-data-coverage customers (sample):")
    print(output[output["data_coverage"].isin(["Low","No data"])][["customer_id","risk_score","categories_present","suggested_data_action"]].head(5).to_string(index=False))


if __name__ == "__main__":
    main()