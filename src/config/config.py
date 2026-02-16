import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME")
HOPSWORKS_HOST = "eu-west.cloud.hopsworks.ai"

FEATURE_GROUP_NAME = "aqi_multan_features"
FEATURE_GROUP_VERSION = 1
FEATURE_VIEW_NAME = "aqi_prediction_fv"
FEATURE_VIEW_VERSION = 1

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
LAT = 30.1979793
LON = 71.4724978
TIMEZONE = "Asia/Karachi"

MODEL_NAMES = ["random_forest", "gradient_boosting", "lightgbm", "decision_tree", "sklearn_gradient_boosting"]
PREDICTION_HORIZON_DAYS = 3

for directory in [RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)