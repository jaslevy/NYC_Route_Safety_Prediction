import pandas as pd
from sodapy import Socrata
from dotenv import load_dotenv
import os

load_dotenv()
MyAppToken = os.getenv("MY_APP_TOKEN")
USERNAME = os.getenv("NYC_EMAIL")
PASSWORD = os.getenv("NYC_PASSWORD")

client = Socrata("data.cityofnewyork.us", MyAppToken, username=USERNAME, password=PASSWORD)

metadata = client.get_metadata("h9gi-nx95")

try:
    total_rows = int(metadata["columns"][0]["cachedContents"]["count"])  # type: ignore
    print(f"Total rows available: {total_rows}")
except KeyError:
    print("Warning: Could not find total row count in metadata. Fetching without limit.")
    total_rows = None 

results = client.get("h9gi-nx95", limit=total_rows)

results_df = pd.DataFrame.from_records(results)

results_df.to_csv("static_data/raw/traffic_data.csv", index=False)