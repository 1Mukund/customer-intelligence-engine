import pandas as pd

def match_not_interested_to_closure_behavior(df):
    try:
        if 'Stage' not in df.columns:
            return pd.DataFrame()

        closure_df = df[df['Stage'].str.lower().isin(['flat blocked', 'sales closure'])]
        ni_df = df[df['Stage'].str.lower() == 'not interested']

        if closure_df.empty or ni_df.empty:
            return pd.DataFrame()

        behavior_cols = ['Inbound WA Count', 'Call Attempt', 'callDuration(secs)', 'TotalTimeSpent', 'ClickCount', 'PageDepth']
        behavior_cols = [col for col in behavior_cols if col in df.columns]

        closure_avg = closure_df[behavior_cols].mean()

        matched_ni = ni_df.copy()
        for col in behavior_cols:
            matched_ni[col + '_gap'] = (ni_df[col] - closure_avg[col]).abs()

        # Simple logic: sum of gaps should be less than a threshold
        matched_ni['total_gap'] = matched_ni[[col + '_gap' for col in behavior_cols]].sum(axis=1)
        threshold = matched_ni['total_gap'].quantile(0.25)  # pick lower quartile

        final_matches = matched_ni[matched_ni['total_gap'] <= threshold]
        return final_matches.drop(columns=[col + '_gap' for col in behavior_cols] + ['total_gap'])

    except Exception as e:
        print(f"Behavior match error: {e}")
        return pd.DataFrame()
