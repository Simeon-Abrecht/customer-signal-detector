"""
Intelligent Customer Signal Detector - Dashboard
----------------------------------------------------
Displays the pipeline's output for a customer retention team: a prioritised,
risk-ranked list of customers with plain-language rationale and suggested
actions. This file does NO analysis itself - it only reads and displays
what scoring.py and rationale_generator.py have already produced.
 
Run with: streamlit run dashboard/app.py
"""
 
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.graph_objects as go
 
st.set_page_config(page_title="Customer Signal Detector", layout="wide")
 
 
@st.cache_data
def load_data():
    risk = pd.read_csv("data/risk_scores.csv")
    rationale = pd.read_csv("data/rationale_and_actions.csv")
    customers = pd.read_csv("data/customers_pipeline_input.csv")
    events = pd.read_csv("data/events.csv")
    intent = pd.read_csv("data/ticket_intent.csv")
 
    combined = risk.merge(rationale, on="customer_id").merge(
        customers[["customer_id", "plan_type", "tenure_months"]], on="customer_id"
    )
    return combined, events, intent
 
 
def risk_tier(row):
    if row["data_coverage"] in ("Low", "No data"):
        return "Data gap"
    elif pd.isna(row["risk_score"]):
        return "Unknown"
    elif row["risk_score"] >= 60:
        return "High"
    elif row["risk_score"] >= 30:
        return "Medium"
    else:
        return "Low"
 
 
TIER_COLORS = {
    "High": "#e05252",
    "Medium": "#e0a952",
    "Low": "#52a852",
    "Data gap": "#8c8c8c",
    "Unknown": "#8c8c8c",
}
 
 
def style_tier(val):
    color = TIER_COLORS.get(val, "#8c8c8c")
    return f"background-color: {color}; color: white; font-weight: 600; text-align: center;"
 
 
def format_evidence_history(customer_id, events_df, intent_df):
    billing = events_df[
        (events_df.customer_id == customer_id) & (events_df.signal_type == "billing_event")
    ].sort_values("event_date", ascending=False)
    csat = events_df[
        (events_df.customer_id == customer_id) & (events_df.signal_type == "csat_survey")
    ].sort_values("event_date", ascending=False)
    tickets = intent_df[intent_df.customer_id == customer_id].sort_values(
        "event_date", ascending=False
    )
    return billing, csat, tickets
 
 
def main():
    combined, events, intent = load_data()
    combined["risk_tier"] = combined.apply(risk_tier, axis=1)
 
    st.title("Intelligent Customer Signal Detector")
    st.caption("Prioritised customer churn-risk signals for the retention team")
 
    # --- Summary panel ---
    total = len(combined)
    high = (combined.risk_tier == "High").sum()
    medium = (combined.risk_tier == "Medium").sum()
    data_gap = (combined.risk_tier == "Data gap").sum()
    avg_score = combined["risk_score"].mean()
 
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total customers", total)
    c2.metric("High risk", high)
    c3.metric("Medium risk", medium)
    c4.metric("Data gap (low evidence)", data_gap)
    c5.metric("Average risk score", f"{avg_score:.1f}")
 
    # --- Risk tier distribution chart (satisfies the brief's "visual risk heatmap" enhancement) ---
    st.write("")
    tier_order = ["Data gap", "Low", "Medium", "High"]  # ascending risk, left to right
    tier_counts = combined["risk_tier"].value_counts().reindex(tier_order, fill_value=0)
 
    fig = go.Figure(
        data=[
            go.Bar(
                x=tier_order,
                y=tier_counts.values,
                marker_color=[TIER_COLORS[t] for t in tier_order],
                text=tier_counts.values,
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        title={"text": "Risk Tier Distribution", "x": 0.5, "xanchor": "center"},
        xaxis_title=None,
        yaxis_title="Number of customers",
        height=320,
        margin=dict(t=60, b=20, l=20, r=20),
        showlegend=False,
    )
 
    with st.container(border=True):
        st.plotly_chart(fig, use_container_width=True)
 
    st.divider()
 
    # --- Filters ---
    st.subheader("Customer risk list")
    col_a, col_b = st.columns([1, 3])
    with col_a:
        tier_filter = st.multiselect(
            "Filter by risk tier",
            options=["High", "Medium", "Low", "Data gap"],
            default=["High", "Medium", "Data gap"],
        )
 
    filtered = combined[combined["risk_tier"].isin(tier_filter)].sort_values(
        "risk_score", ascending=False, na_position="last"
    )
 
    st.write(f"Showing {len(filtered)} of {total} customers")
 
    # --- Main table ---
    display_cols = [
        "customer_id", "risk_score", "risk_tier", "data_coverage",
        "plan_type", "tenure_months", "rationale", "suggested_action",
    ]
    styled = filtered[display_cols].style.map(style_tier, subset=["risk_tier"])
    st.dataframe(styled, use_container_width=True, hide_index=True)
 
    st.divider()
 
    # --- Per-customer detail view ---
    st.subheader("Customer detail")
    selected_id = st.selectbox(
        "Select a customer to view full evidence",
        options=filtered["customer_id"].tolist(),
    )
 
    if selected_id:
        row = combined[combined.customer_id == selected_id].iloc[0]
 
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Risk score", f"{row['risk_score']:.1f}" if pd.notna(row["risk_score"]) else "N/A")
            st.write(f"**Tier:** {row['risk_tier']}")
            st.write(f"**Data coverage:** {row['data_coverage']}")
            st.write(f"**Plan:** {row['plan_type']} ({row['tenure_months']} months)")
            st.write(f"**Categories present:** {row['categories_present']}")
            st.write(f"**Categories missing:** {row['categories_missing']}")
 
        with col2:
            st.write("**Rationale**")
            st.info(row["rationale"])
            st.write("**Suggested action**")
            st.success(row["suggested_action"])
 
        st.write("**Evidence history**")
        billing, csat, tickets = format_evidence_history(selected_id, events, intent)
 
        e1, e2, e3 = st.columns(3)
        with e1:
            st.write("Billing events")
            if len(billing):
                st.dataframe(billing[["event_date", "detail"]], hide_index=True, use_container_width=True)
            else:
                st.write("_No data_")
        with e2:
            st.write("CSAT scores")
            if len(csat):
                st.dataframe(csat[["event_date", "detail"]], hide_index=True, use_container_width=True)
            else:
                st.write("_No data_")
        with e3:
            st.write("Support interactions")
            if len(tickets):
                st.dataframe(
                    tickets[["event_date", "intent_category", "intent_reason"]],
                    hide_index=True, use_container_width=True
                )
            else:
                st.write("_No data_")
 
 
if __name__ == "__main__":
    main()