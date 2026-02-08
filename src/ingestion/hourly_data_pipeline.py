import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import pytz
import hopsworks
import sys
from pathlib import Path
import os
from dotenv import load_dotenv
import time

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config.config import *

load_dotenv()

class HourlyPipeline:
    def __init__(self):
        self.api_key = os.getenv("OPENWEATHER_API_KEY")
        self.hopsworks_key = os.getenv("HOPSWORKS_API_KEY")
        self.project_name = os.getenv("HOPSWORKS_PROJECT_NAME")
        self.lat = LAT
        self.lon = LON
        self.pkt = pytz.timezone(TIMEZONE)
        self.project = None
        self.fs = None
        self.fg = None
        
    def connect_hopsworks(self):
        self.project = hopsworks.login(
            host=HOPSWORKS_HOST,
            api_key_value=self.hopsworks_key,
            project=self.project_name
        )
        self.fs = self.project.get_feature_store()
        self.fg = self.fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
        print("Connected")
    
    def fetch_current(self):
        print("\nFetching current data...")
        
        pollution_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={self.lat}&lon={self.lon}&appid={self.api_key}"
        weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={self.lat}&lon={self.lon}&appid={self.api_key}&units=metric"
        
        p = requests.get(pollution_url).json()
        w = requests.get(weather_url).json()
        
        if "list" not in p or "main" not in w:
            return None
        
        item = p["list"][0]
        utc_time = datetime.utcfromtimestamp(item["dt"]).replace(tzinfo=timezone.utc)
        pkt_time = utc_time.astimezone(self.pkt)
        
        data = {
            "datetime": pkt_time,
            "aqi": item["main"]["aqi"],
            "co": item["components"]["co"],
            "no2": item["components"]["no2"],
            "pm2_5": item["components"]["pm2_5"],
            "pm10": item["components"]["pm10"],
            "temp": w["main"]["temp"],
            "feels_like": w["main"]["feels_like"],
            "pressure": w["main"]["pressure"],
            "dew_point": w["main"].get("dew_point", w["main"]["temp"] - ((100 - w["main"]["humidity"]) / 5)),
            "wind_speed": w["wind"]["speed"],
            "wind_deg": w["wind"]["deg"]
        }
        
        print(f"Fetched: {pkt_time}, AQI={data['aqi']}")
        return pd.DataFrame([data])
    
    def check_duplicate(self, new, recent_hist):
        print("\nChecking duplicates (last 6h)...")
        
        new_dt = pd.to_datetime(new['datetime'].iloc[0])
        six_h_ago = new_dt - timedelta(hours=6)
        
        # Filter the history we already pulled
        recent = recent_hist[recent_hist['datetime'] >= six_h_ago]
        
        if len(recent) == 0:
            return False
        
        cols = ['aqi', 'pm2_5', 'pm10', 'temp']
        new_vals = new[cols].iloc[0]
        
        for _, row in recent.iterrows():
            if all(abs(row[c] - new_vals[c]) < 0.01 for c in cols):
                print(f"Duplicate found (matches {row['datetime']})")
                return True
        return False
    
    def engineer(self, new, hist):
        print("\nEngineering features...")
        
        combined = pd.concat([hist, new], ignore_index=True).sort_values('datetime').reset_index(drop=True)
        idx = len(combined) - 1
        dt = combined.loc[idx, 'datetime']
        
        combined.loc[idx, 'hour_sin'] = np.sin(2 * np.pi * dt.hour / 24)
        combined.loc[idx, 'hour_cos'] = np.cos(2 * np.pi * dt.hour / 24)
        combined.loc[idx, 'day_of_week_sin'] = np.sin(2 * np.pi * dt.weekday() / 7)
        combined.loc[idx, 'day_of_week_cos'] = np.cos(2 * np.pi * dt.weekday() / 7)
        combined.loc[idx, 'month_sin'] = np.sin(2 * np.pi * dt.month / 12)
        combined.loc[idx, 'month_cos'] = np.cos(2 * np.pi * dt.month / 12)
        combined.loc[idx, 'is_weekend'] = 1 if dt.weekday() >= 5 else 0
        
        for lag in [1, 3, 6, 12, 24, 48]:
            if idx >= lag:
                combined.loc[idx, f'aqi_lag_{lag}h'] = combined.loc[idx - lag, 'aqi']
                if lag in [1, 3, 6, 24]:
                    combined.loc[idx, f'pm2_5_lag_{lag}h'] = combined.loc[idx - lag, 'pm2_5']
                if lag in [1, 3, 24]:
                    combined.loc[idx, f'pm10_lag_{lag}h'] = combined.loc[idx - lag, 'pm10']
        
        for w in [3, 6, 12, 24]:
            if idx >= w - 1:
                combined.loc[idx, f'aqi_rolling_mean_{w}h'] = combined.loc[max(0, idx-w+1):idx+1, 'aqi'].mean()
        
        if idx >= 5:
            combined.loc[idx, 'aqi_rolling_std_6h'] = combined.loc[idx-5:idx+1, 'aqi'].std()
            combined.loc[idx, 'pm2_5_rolling_mean_6h'] = combined.loc[idx-5:idx+1, 'pm2_5'].mean()
        
        if idx >= 23:
            combined.loc[idx, 'aqi_rolling_std_24h'] = combined.loc[idx-23:idx+1, 'aqi'].std()
            combined.loc[idx, 'aqi_rolling_min_24h'] = combined.loc[idx-23:idx+1, 'aqi'].min()
            combined.loc[idx, 'aqi_rolling_max_24h'] = combined.loc[idx-23:idx+1, 'aqi'].max()
            combined.loc[idx, 'pm2_5_rolling_mean_24h'] = combined.loc[idx-23:idx+1, 'pm2_5'].mean()
            combined.loc[idx, 'aqi_change_24h'] = combined.loc[idx, 'aqi'] - combined.loc[idx-24, 'aqi']
        
        combined.loc[idx, 'pm2_5_x_wind_speed'] = combined.loc[idx, 'pm2_5'] * combined.loc[idx, 'wind_speed']
        
        result = combined.iloc[idx:idx+1].copy()
        result['timestamp'] = int(dt.timestamp() * 1000)
        result['is_weekend'] = result['is_weekend'].astype(int)
        print(f"{len(result.columns)} features")
        return result
    
    def upload(self, data):
        print("\nUploading...")
        
        data = data.fillna(0)
        data['timestamp'] = data['datetime'].values.astype('int64') // 10**6
        data['is_weekend'] = data['is_weekend'].astype('int64')
        data['aqi'] = data['aqi'].astype('int64')
        
        try:
            self.fg.insert(data, write_options={"wait_for_job": False})
            print("Uploaded")
            time.sleep(2)
        except Exception as e:
            print(f"Upload failed: {e}")
            print("Retrying...")
            time.sleep(3)
            try:
                self.fg.insert(data, write_options={"wait_for_job": False})
                print("Uploaded (retry)")
            except Exception as e2:
                print(f"Upload failed again: {e2}")
    
    def run(self):
        print("\n" + "="*60)
        print("HOURLY PIPELINE")
        print("="*60)
        
        self.connect_hopsworks()
        
        new = self.fetch_current()
        if new is None:
            print("Fetch failed")
            return
        
        print("Fetching recent context from Online Store...")
        hist = self.fg.read(online=True)
        hist['datetime'] = pd.to_datetime(hist['datetime'])
        hist = hist.sort_values('datetime')
        
        cutoff = datetime.now(self.pkt) - timedelta(hours=50)
        hist_recent = hist[hist['datetime'] >= cutoff].copy()
        
        if self.check_duplicate(new, hist_recent):
            return
        
        raw_cols = ['datetime', 'aqi', 'co', 'no2', 'pm2_5', 'pm10', 'temp', 'feels_like', 'pressure', 'dew_point', 'wind_speed', 'wind_deg']
        engineered = self.engineer(new, hist_recent[raw_cols])
        
        print("\nUploading to Feature Store...")
        self.fg.insert(engineered, write_options={"wait_for_job": False})
        print("Upload complete. Materialization started in background.")

def main():
    HourlyPipeline().run()

if __name__ == "__main__":
    main()