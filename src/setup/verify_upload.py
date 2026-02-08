import pandas as pd
import hopsworks
import os
from dotenv import load_dotenv

load_dotenv()

project = hopsworks.login(
    host="eu-west.cloud.hopsworks.ai",
    api_key_value=os.getenv("HOPSWORKS_API_KEY"),
    project=os.getenv("HOPSWORKS_PROJECT_NAME")
)

fs = project.get_feature_store()
fg = fs.get_feature_group(name="aqi_multan_features", version=1)

df = fg.read()
df['datetime'] = pd.to_datetime(df['datetime'])

print(f"\nTotal records: {len(df)}")
print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
print(f"\nLatest 10 records:")
print(df[['datetime', 'aqi', 'pm2_5', 'temp']].tail(10))