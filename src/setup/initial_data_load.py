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

class InitialBackfill:
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
        
        # Get existing feature group (NOT create new)
        self.fg = self.fs.get_or_create_feature_group(
            name=FEATURE_GROUP_NAME,
            version=FEATURE_GROUP_VERSION,
            description="AQI prediction features for Multan, Pakistan",
            primary_key=['timestamp'],
            event_time='datetime',
            online_enabled=True,
            statistics_config=True
        )
        
        print("Connected to existing feature group")
    
    def get_latest_date(self):
        print("\n" + "="*60)
        print("CHECKING EXISTING DATA")
        print("="*60)
        
        existing = self.fg.read()
        existing['datetime'] = pd.to_datetime(existing['datetime'])
        latest = existing['datetime'].max()
        
        print(f"Existing: {len(existing)} records")
        print(f"Range: {existing['datetime'].min()} to {latest}")
        
        return existing, latest
    
    def extract_data(self, start_date, end_date):
        print("\n" + "="*60)
        print(f"EXTRACTING NEW DATA")
        print("="*60)
        print(f"From: {start_date}")
        print(f"To: {end_date}")
        
        all_data = []
        current_date = start_date
        
        while current_date < end_date:
            next_date = current_date + timedelta(days=1)
            start_ts = int(current_date.timestamp())
            end_ts = int(next_date.timestamp())
            
            pollution_url = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={self.lat}&lon={self.lon}&start={start_ts}&end={end_ts}&appid={self.api_key}"
            weather_url = f"https://api.openweathermap.org/data/3.0/onecall/timemachine?lat={self.lat}&lon={self.lon}&dt={start_ts}&appid={self.api_key}&units=metric"
            
            pollution_resp = requests.get(pollution_url).json()
            weather_resp = requests.get(weather_url).json()
            
            if "list" in pollution_resp:
                for i, item in enumerate(pollution_resp["list"]):
                    utc_time = datetime.utcfromtimestamp(item["dt"]).replace(tzinfo=timezone.utc)
                    pkt_time = utc_time.astimezone(self.pkt)
                    weather = weather_resp.get("data", [])[i] if i < len(weather_resp.get("data", [])) else {}
                    
                    all_data.append({
                        "datetime": pkt_time,
                        "aqi": item["main"]["aqi"],
                        "co": item["components"]["co"],
                        "no2": item["components"]["no2"],
                        "pm2_5": item["components"]["pm2_5"],
                        "pm10": item["components"]["pm10"],
                        "temp": weather.get("temp"),
                        "feels_like": weather.get("feels_like"),
                        "pressure": weather.get("pressure"),
                        "dew_point": weather.get("dew_point"),
                        "wind_speed": weather.get("wind_speed"),
                        "wind_deg": weather.get("wind_deg")
                    })
            
            print(f"  {current_date.date()}")
            current_date = next_date
            time.sleep(1)
        
        if len(all_data) == 0:
            return None
            
        df = pd.DataFrame(all_data).sort_values("datetime").reset_index(drop=True)
        print(f"\nExtracted {len(df)} new records")
        return df
    
    def engineer_features(self, new_data, historical_data):
        print("\n" + "="*60)
        print("ENGINEERING FEATURES")
        print("="*60)
        
        # Combine historical + new
        combined = pd.concat([historical_data, new_data], ignore_index=True)
        combined = combined.sort_values('datetime').reset_index(drop=True)
        
        start_idx = len(historical_data)
        
        for idx in range(start_idx, len(combined)):
            dt = combined.loc[idx, 'datetime']
            
            # Time features
            combined.loc[idx, 'hour_sin'] = np.sin(2 * np.pi * dt.hour / 24)
            combined.loc[idx, 'hour_cos'] = np.cos(2 * np.pi * dt.hour / 24)
            combined.loc[idx, 'day_of_week_sin'] = np.sin(2 * np.pi * dt.weekday() / 7)
            combined.loc[idx, 'day_of_week_cos'] = np.cos(2 * np.pi * dt.weekday() / 7)
            combined.loc[idx, 'month_sin'] = np.sin(2 * np.pi * dt.month / 12)
            combined.loc[idx, 'month_cos'] = np.cos(2 * np.pi * dt.month / 12)
            combined.loc[idx, 'is_weekend'] = 1 if dt.weekday() >= 5 else 0
            
            # Lags
            for lag in [1, 3, 6, 12, 24, 48]:
                if idx >= lag:
                    combined.loc[idx, f'aqi_lag_{lag}h'] = combined.loc[idx - lag, 'aqi']
                    if lag in [1, 3, 6, 24]:
                        combined.loc[idx, f'pm2_5_lag_{lag}h'] = combined.loc[idx - lag, 'pm2_5']
                    if lag in [1, 3, 24]:
                        combined.loc[idx, f'pm10_lag_{lag}h'] = combined.loc[idx - lag, 'pm10']
            
            # Rolling
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
            
            # Interaction
            combined.loc[idx, 'pm2_5_x_wind_speed'] = combined.loc[idx, 'pm2_5'] * combined.loc[idx, 'wind_speed']
        
        # Extract only NEW rows
        new_engineered = combined.iloc[start_idx:].copy()
        
        print(f"Engineered {len(new_engineered)} new records")
        return new_engineered
    
    def upload(self, data):
        print("\n" + "="*60)
        print("APPENDING TO FEATURE STORE")
        print("="*60)
        
        # Prepare data
        data = data.fillna(0)
        data['timestamp'] = data['datetime'].values.astype('int64') // 10**6
        data['is_weekend'] = data['is_weekend'].astype('int64')
        data['aqi'] = data['aqi'].astype('int64')
        
        print(f"Uploading {len(data)} records...")
        print("(This will APPEND to existing 2857 records)")
        
        # Insert - will append to existing data
        self.fg.insert(data, write_options={"start_offline_materialization": True})
        self.fg.commit_details(wallclock_time=None, limit=None)
        
        print(f"Appended {len(data)} records")
    
    def verify(self):
        print("\n" + "="*60)
        print("VERIFYING APPEND")
        print("="*60)
        
        df = self.fg.read(online=True)
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        print(f"Total records NOW: {len(df)}")
        print(f"Latest date: {df['datetime'].max()}")
        print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        
        return True
    
    def run(self):
        print("\n" + "="*80)
        print("BACKFILL - APPEND NEW DATA")
        print("="*80)
        
        # Connect
        self.connect_hopsworks()
        
        # Get existing data
        existing, latest = self.get_latest_date()
        
        # Determine date range for new data
        start = latest + timedelta(hours=1)
        end = datetime.now(timezone.utc)
        
        if start >= end:
            print(f"\nAlready up to date (latest: {latest})")
            print("\nProceeding to training & prediction...")
        else:
            # Extract only raw columns from existing data
            raw_cols = ['datetime', 'aqi', 'co', 'no2', 'pm2_5', 'pm10', 
                       'temp', 'feels_like', 'pressure', 'dew_point', 
                       'wind_speed', 'wind_deg']
            historical_raw = existing[raw_cols].copy()
            
            # Extract new raw data
            new_raw = self.extract_data(start, end)
            
            if new_raw is None or len(new_raw) == 0:
                print("\nNo new data available")
            else:
                # Engineer features using historical context
                new_featured = self.engineer_features(new_raw, historical_raw)
                
                # Append to existing feature group
                self.upload(new_featured)
                
                # Verify
                self.verify()
        
        print("\n" + "="*80)
        print("BACKFILL COMPLETE")
        print("="*80)

def main():
    InitialBackfill().run()

if __name__ == "__main__":
    main()