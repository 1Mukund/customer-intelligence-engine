import pandas as pd

def generate_kpis(df):
    kpi_data = {}

    if df is None or df.empty:
        return None

    # Total leads
    kpi_data["Total Leads"] = df["masterLeadId"].nunique()

    # Avg. Time Spent
    if "TotalTimeSpent" in df.columns:
        kpi_data["Avg. Time Spent (sec)"] = round(df["TotalTimeSpent"].mean(), 2)
    else:
        kpi_data["Avg. Time Spent (sec)"] = "N/A"

    # Avg. Clicks
    if "ClickCount" in df.columns:
        kpi_data["Avg. Clicks"] = round(df["ClickCount"].mean(), 2)
    else:
        kpi_data["Avg. Clicks"] = "N/A"

    # Engagement Rate (% of leads with time > 0)
    if "TotalTimeSpent" in df.columns:
        engaged = df[df["TotalTimeSpent"] > 0]["masterLeadId"].nunique()
        kpi_data["Engaged %"] = round((engaged / df["masterLeadId"].nunique()) * 100, 2)
    else:
        kpi_data["Engaged %"] = "N/A"

    # High intent leads (you can customize threshold)
    if "ClickCount" in df.columns:
        high_intent = df[df["ClickCount"] >= 5]["masterLeadId"].nunique()
        kpi_data["High Intent Leads (≥5 clicks)"] = high_intent
    else:
        kpi_data["High Intent Leads (≥5 clicks)"] = "N/A"

    return pd.DataFrame(kpi_data.items(), columns=["Metric", "Value"])
