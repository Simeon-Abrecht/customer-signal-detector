"""
Churn-intent detection for support ticket text
--------------------------------------------------
Reads events.csv, and for every support_ticket, asks an LLM (Claude Haiku)
to judge whether the ticket signals churn intent - independent of emotional
tone, which VADER already captures separately in sentiment.py.

Rather than asking the LLM for a raw continuous score (which it can't
produce reliably/consistently), it returns one of three categories with a
short justification. Categories are then mapped to a 0-1 numeric value:

  No signal     -> 0.0
  Mild signal   -> 0.5
  Strong signal -> 1.0

This keeps the score defensible (every value traces back to a labelled
judgement + a stated reason) and directionally consistent with the rest
of the risk scoring pipeline (higher number = higher risk, no sign-flipping
needed anywhere in scoring.py).
"""

import os
import time
import pandas as pd
from dotenv import load_dotenv
import anthropic

load_dotenv()  # reads ANTHROPIC_API_KEY from .env

client = anthropic.Anthropic()  # picks up ANTHROPIC_API_KEY automatically

INTENT_MAP = {
    "No signal": 0.0,
    "Mild signal": 0.5,
    "Strong signal": 1.0,
}

PROMPT_TEMPLATE = """A telecom customer wrote the following support ticket note:

"{ticket_text}"

Does this suggest the customer may be considering leaving/switching providers,
independent of how angry or calm they sound? Judge based on content/intent,
not tone.

Respond in exactly this format, nothing else. Keep the reason to a maximum
of 15 words.

Category: <No signal|Mild signal|Strong signal>
Reason: <short reason, 15 words or fewer>
"""

def classify_intent(ticket_text, retries=3):
    prompt = PROMPT_TEMPLATE.format(ticket_text=ticket_text)

    for attempt in range(retries):
        try:
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=120,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            text = response.content[0].text.strip()

            category_raw = ""
            reason = ""
            for line in text.splitlines():
                if line.lower().startswith("category:"):
                    category_raw = line.split(":", 1)[1].strip().rstrip(".")
                elif line.lower().startswith("reason:"):
                    reason = line.split(":", 1)[1].strip()

            # robust matching - handles case/punctuation variation, falls back
            # to substring matching on the key word rather than exact match
            category_lower = category_raw.lower()
            if "strong" in category_lower:
                category, score = "Strong signal", 1.0
            elif "mild" in category_lower:
                category, score = "Mild signal", 0.5
            elif "no signal" in category_lower or "no" == category_lower:
                category, score = "No signal", 0.0
            else:
                # unexpected format - flag it clearly rather than silently defaulting
                category, score = f"UNPARSED: {category_raw}", 0.0

            if not reason:
                reason = "No reason provided by model"

            return score, category, reason

        except Exception as e:
            print(f"  Retry {attempt+1}/{retries} after error: {e}")
            time.sleep(2)

    return 0.0, "No signal", "Classification failed after retries"

def main():
    events = pd.read_csv("data/events.csv")
    tickets = events[events["signal_type"] == "support_ticket"].copy()

    # TEST MODE: set to True first, confirm output looks right, then set to False
    TEST_MODE = False
    if TEST_MODE:
        tickets = tickets.head(10)  # only run on first 10 tickets for testing
        print("TEST MODE: running on 10 tickets only")

    print(f"Classifying churn intent for {len(tickets)} tickets...")
    results = []
    for i, row in tickets.iterrows():
        score, category, reason = classify_intent(row["detail"])
        results.append({
            "event_id": row["event_id"],
            "customer_id": row["customer_id"],
            "event_date": row["event_date"],
            "intent_score": score,
            "intent_category": category,
            "intent_reason": reason,
        })
        if (len(results)) % 25 == 0:
            print(f"  {len(results)}/{len(tickets)} done")

    output = pd.DataFrame(results)
    output.to_csv("data/ticket_intent.csv", index=False)

    print("\nDone. Intent category distribution:")
    print(output["intent_category"].value_counts())
    print("\nSample results:")
    print(output.head(8).to_string(index=False))

if __name__ == "__main__":
    main()