# debug_upload.py
import pandas as pd
import numpy as np
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

# Check schema
print("Expected schema:")
for feat in fg.features:
    print(f"  {feat.name}: {feat.type}")