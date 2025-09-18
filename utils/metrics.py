import pandas as pd
from datetime import datetime

# Define your journey weight logic (higher = deeper)
PAGE_WEIGHTS = {
    "HOME": 1,
    "GALLERY": 2,
    "PLAN": 3,
    "PRICING": 4,
    "LOCATION": 5,
    "AMENITIES": 6,
    "ABOUT": 7,
    "CONTACT": 8
}

def calculate_behavior_metrics(web_df: pd.DataFrame) -> pd.DataFrame:
    # Ensure proper types
    web_df["masterLeadId"] = web_df["masterLeadId"].astype(str)
    web_df["time_spent_seconds"] = pd.to_numeric(web_df["time_spent_seconds"], errors="coerce").fillna(0)
    web_df["timestamp"] = pd.to_datetime(web_df["timestamp"], errors="coerce")

    # Remove rows without masterLeadId
    web_df = web_df[web_df["masterLeadId"].notnull()]

    # Group by lead
    grouped = web_df.groupby("masterLeadId")

    # Calculate metrics
    summary = grouped.agg(
        PageDepth=('page_name', lambda x: x[web_df.loc[x.index, "time_spent_seconds"] > 0].nunique()),
        TotalTimeSpent=('time_spent_seconds', 'sum'),
        LastSeen=('timestamp', 'max')
    )

    summary["AvgTimePerPage"] = summary["TotalTimeSpent"] / summary["PageDepth"]

    # Recency Index: 1 / days since last seen
    today = pd.Timestamp(datetime.now().date())
    summary["RecencyIndex"] = (today - summary["LastSeen"]).dt.days
    summary["RecencyIndex"] = summary["RecencyIndex"].apply(lambda x: round(1 / x, 3) if x > 0 else 1.0)

    # PageIndex: weighted average
    web_df["page_score"] = web_df["page_name"].map(PAGE_WEIGHTS).fillna(0)
    page_index_df = grouped["page_score"].mean().rename("PageIndex")

    # Merge page index
    summary = summary.join(page_index_df)

    # Drop timestamp to clean
    summary = summary.drop(columns=["LastSeen"])
    
    return summary.reset_index()
