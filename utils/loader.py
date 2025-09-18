import pandas as pd
import requests
from io import StringIO

def load_sheet_csv(sheet_url):
    try:
        response = requests.get(sheet_url)
        response.raise_for_status()
        csv_data = StringIO(response.text)
        df = pd.read_csv(csv_data)
        return df
    except Exception as e:
        print(f"Error loading sheet: {e}")
        return pd.DataFrame()
