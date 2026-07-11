"""
Rationale and suggested action generation
---------------------------------------------
Produces a rationale (plain-language explanation) and suggested_action for
EVERY customer in risk_scores.csv, so the final dashboard has full coverage -
but only spends LLM calls where genuine reasoning actually adds value.
 
THREE TIERS:
 
1. LOW/NO-DATA COVERAGE (any risk_score)
   -> No LLM call. Reuses suggested_data_action already computed in
      scoring.py (recommend proactive outreach/CSAT check-in). This
      customer's risk is genuinely uncertain due to thin evidence, not
      something an LLM can reason its way around.
 
2. MEDIUM/HIGH COVERAGE, risk_score >= LLM_THRESHOLD (elevated risk)
   -> LLM call (Claude Haiku). This is the only tier where a language
      model is actually needed: turning multi-signal evidence into a
      nuanced, human-readable explanation and a specific suggested action.
 
3. MEDIUM/HIGH COVERAGE, risk_score < LLM_THRESHOLD (stable/moderate)
   -> Deterministic, rule-based text. No LLM call - there's no real
      ambiguity here for a model to reason about, so a templated summary
      referencing the actual category scores is used instead. This keeps
      the dashboard fully populated without spending LLM calls on
      customers who don't need nuanced reasoning.
 
This mirrors the same "use the LLM only where it earns its keep" principle
used throughout the pipeline (VADER for cheap sentiment scoring, LLM only
for churn-intent judgement; LLM only for elevated-risk rationale here).
"""
 
import os
import time
import pandas as pd
from dotenv import load_dotenv
import anthropic
 
load_dotenv()
client = anthropic.Anthropic()
 
LLM_THRESHOLD = 60  # risk_score cutoff for LLM-generated rationale (Medium/High coverage only)
STABLE_THRESHOLD = 30  # below this = "stable", between this and LLM_THRESHOLD = "monitor"
 
MAX_RATIONALE_CHARS = 150
MAX_ACTION_CHARS = 100
 
SYSTEM_PROMPT = """You are helping a telecom customer retention team understand why a customer has been flagged as at-risk.
 
Each customer's risk score is calculated from four types of evidence: how satisfied the customer says they are (CSAT surveys), the tone of their support interactions, whether their support messages suggest they're considering leaving, and their billing/payment behaviour.
 
The system weighs each piece of evidence based on how recent it is relative to that customer's OTHER evidence - not the actual amount of time that has passed. A customer's most recent interaction always carries the most weight, and each interaction before that carries progressively less, regardless of whether their most recent interaction was last week or a year ago. This means a customer who once had a problem but has since had several calmer, more recent interactions is treated as lower risk than a customer showing that same problem in their most recent interaction.
 
The system also treats a customer's own stated satisfaction (CSAT), and explicit statements about considering leaving, as somewhat stronger evidence than tone alone, since tone can be misleading - a calmly-worded message can still describe a serious problem.
 
You will be given this customer's actual evidence: their billing history, satisfaction scores, and support interaction notes. Based on this evidence, write a short, plain-language explanation of why they've been flagged, and one concrete suggested action for the retention team.
 
Respond in exactly this format, nothing else:
Rationale: <plain-language explanation, under {max_rationale} characters>
Suggested Action: <one concrete action, under {max_action} characters>
""".format(max_rationale=MAX_RATIONALE_CHARS, max_action=MAX_ACTION_CHARS)
 
USER_MESSAGE_TEMPLATE = """Customer risk profile:
 
Category risk levels (0-100, higher = more concerning):
- CSAT: {csat_risk}
- Sentiment: {sentiment_risk}
- Intent: {intent_risk}
- Billing: {billing_risk}
Categories with no data available: {categories_missing}
 
Billing history (most recent first):
{billing_history}
 
CSAT scores (most recent first):
{csat_history}
 
Support interaction notes (most recent first):
{intent_history}
"""
 
 
def format_billing_history(rows):
    if not rows:
        return "No billing data available"
    return "\n".join(f"- {date}: {detail}" for date, detail in rows)
 
 
def format_csat_history(rows):
    if not rows:
        return "No CSAT data available"
    return "\n".join(f"- {date}: {score}/10" for date, score in rows)
 
 
def format_intent_history(rows):
    if not rows:
        return "No support interaction data available"
    return "\n".join(f"- {date}: {reason}" for date, reason in rows)
 
 
def gather_customer_evidence(customer_id, events_df, intent_df):
    billing_rows = events_df[
        (events_df["customer_id"] == customer_id) & (events_df["signal_type"] == "billing_event")
    ].sort_values("event_date", ascending=False)
    billing_history = format_billing_history(
        list(zip(billing_rows["event_date"], billing_rows["detail"]))
    )
 
    csat_rows = events_df[
        (events_df["customer_id"] == customer_id) & (events_df["signal_type"] == "csat_survey")
    ].sort_values("event_date", ascending=False)
    csat_history = format_csat_history(
        list(zip(csat_rows["event_date"], csat_rows["detail"]))
    )
 
    intent_rows = intent_df[intent_df["customer_id"] == customer_id].sort_values(
        "event_date", ascending=False
    )
    intent_history = format_intent_history(
        list(zip(intent_rows["event_date"], intent_rows["intent_reason"]))
    )
 
    return billing_history, csat_history, intent_history
 
 
def generate_llm_rationale(customer_row, events_df, intent_df, retries=3):
    billing_history, csat_history, intent_history = gather_customer_evidence(
        customer_row["customer_id"], events_df, intent_df
    )
 
    user_message = USER_MESSAGE_TEMPLATE.format(
        csat_risk=customer_row["csat_risk"] if pd.notna(customer_row["csat_risk"]) else "N/A",
        sentiment_risk=customer_row["sentiment_risk"] if pd.notna(customer_row["sentiment_risk"]) else "N/A",
        intent_risk=customer_row["intent_risk"] if pd.notna(customer_row["intent_risk"]) else "N/A",
        billing_risk=customer_row["billing_risk"] if pd.notna(customer_row["billing_risk"]) else "N/A",
        categories_missing=customer_row["categories_missing"],
        billing_history=billing_history,
        csat_history=csat_history,
        intent_history=intent_history,
    )
 
    for attempt in range(retries):
        try:
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=200,
                temperature=0,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
            text = response.content[0].text.strip()
 
            rationale = ""
            action = ""
            for line in text.splitlines():
                if line.lower().startswith("rationale:"):
                    rationale = line.split(":", 1)[1].strip()
                elif line.lower().startswith("suggested action:"):
                    action = line.split(":", 1)[1].strip()
 
            if not rationale:
                rationale = "Elevated risk based on multiple signals - see category scores"
            if not action:
                action = "Escalate to retention specialist for manual review"
 
            return rationale, action
 
        except Exception as e:
            print(f"    Retry {attempt+1}/{retries} after error: {e}")
            time.sleep(2)
 
    return "Classification failed after retries", "Escalate to retention specialist for manual review"
 
 
def generate_rule_based_text(customer_row):
    """Deterministic rationale/action for Medium/High coverage customers below the LLM threshold."""
    score = customer_row["risk_score"]
 
    # build a short list of which categories actually looked concerning (risk >= 50),
    # so even the templated text reflects real evidence, not just a generic line
    concerning = []
    for cat, label in [("csat_risk", "satisfaction"), ("sentiment_risk", "support tone"),
                        ("intent_risk", "switching intent"), ("billing_risk", "billing")]:
        val = customer_row[cat]
        if pd.notna(val) and val >= 50:
            concerning.append(label)
 
    if score < STABLE_THRESHOLD:
        rationale = "Stable - no significant concerns across available signals."
        action = "No action needed - continue standard engagement."
    else:
        if concerning:
            rationale = f"Some risk signals present ({', '.join(concerning)}), below the threshold for escalation."
        else:
            rationale = "Moderate risk signals present but none individually concerning."
        action = "Monitor - no immediate action required, revisit if signals increase."
 
    return rationale, action
 
 
def main():
    risk_scores = pd.read_csv("data/risk_scores.csv")
    events = pd.read_csv("data/events.csv")
    intent = pd.read_csv("data/ticket_intent.csv")
 
    results = []
    llm_calls = 0
 
    for _, row in risk_scores.iterrows():
        cid = row["customer_id"]
 
        if row["data_coverage"] in ("Low", "No data"):
            # Tier 1: reuse the data-gap action already computed in scoring.py
            rationale = "Insufficient signal for a confident assessment."
            action = row["suggested_data_action"]
 
        elif pd.notna(row["risk_score"]) and row["risk_score"] >= LLM_THRESHOLD:
            # Tier 2: elevated risk, sufficient coverage - genuine LLM reasoning
            rationale, action = generate_llm_rationale(row, events, intent)
            llm_calls += 1
            if llm_calls % 10 == 0:
                print(f"  {llm_calls} LLM calls made...")
 
        else:
            # Tier 3: sufficient coverage, but not elevated - deterministic template
            rationale, action = generate_rule_based_text(row)
 
        results.append({
            "customer_id": cid,
            "rationale": rationale,
            "suggested_action": action,
        })
 
    output = pd.DataFrame(results)
    output.to_csv("data/rationale_and_actions.csv", index=False)
 
    print(f"\nDone. {llm_calls} customers required an LLM call out of {len(risk_scores)} total.")
    print("\nSample output:")
    print(output.head(10).to_string(index=False))
 
 
if __name__ == "__main__":
    main()