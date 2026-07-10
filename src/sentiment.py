"""
Sentiment scoring for support ticket text
-------------------------------------------
Reads events.csv, scores every support_ticket's text using VADER
(a lightweight, rule-based sentiment analyser well-suited to short text),
and writes a separate ticket_sentiment.csv with one row per ticket.

Output columns:
  event_id, customer_id, event_date, compound_score, sentiment_label

compound_score ranges from -1 (very negative) to +1 (very positive).
sentiment_label buckets this into Negative / Neutral / Positive for readability.
"""

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def score_text(text):
    scores = analyzer.polarity_scores(text)
    return scores["compound"]

def label_from_score(compound):
    if compound <= -0.05:
        return "Negative"
    elif compound >= 0.05:
        return "Positive"
    else:
        return "Neutral"

def main():
    events = pd.read_csv("data/events.csv")
    tickets = events[events["signal_type"] == "support_ticket"].copy()

    tickets["compound_score"] = tickets["detail"].apply(score_text)
    tickets["sentiment_label"] = tickets["compound_score"].apply(label_from_score)

    output = tickets[["event_id", "customer_id", "event_date", "compound_score", "sentiment_label"]]
    output.to_csv("data/ticket_sentiment.csv", index=False)

    print("Tickets scored:", len(output))
    print("\nSentiment label distribution:")
    print(output["sentiment_label"].value_counts())
    print("\nSample scored tickets:")
    sample = tickets[["customer_id", "detail", "compound_score", "sentiment_label"]].head(8)
    print(sample.to_string(index=False))

if __name__ == "__main__":
    main()