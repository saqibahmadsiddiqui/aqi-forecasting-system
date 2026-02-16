import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import pytz
import hopsworks
import sys
from pathlib import Path
import time

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config.config import *

class HistoricalBackfill:
    def __init__(self):
        self.lat = LAT
        self.lon = LON
        self.pkt = pytz.timezone(TIMEZONE)
        self.project = None
        self.fs = None
        self.fg = None
        
    def connect_hopsworks(self):
        self.project = hopsworks.login(
            host=HOPSWORKS_HOST,
            api_key_value=HOPSWORKS_API_KEY,
            project=HOPSWORKS_PROJECT_NAME
        )
        self.fs = self.project.get_feature_store()
        
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
    
    def get_existing_data(self):
        print("\n" + "="*60)
        print("FETCHING EXISTING DATA")
        print("="*60)
        
        existing = self.fg.read()
        existing['datetime'] = pd.to_datetime(existing['datetime'])
        
        print(f"Existing records: {len(existing)}")
        print(f"Date range: {existing['datetime'].min()} to {existing['datetime'].max()}")
        
        return existing
    
    def extract_data(self, start_date, end_date):
        print("\n" + "="*60)
        print(f"EXTRACTING HISTORICAL DATA")
        print("="*60)
        print(f"From: {start_date}")
        print(f"To: {end_date}")
        
        all_data = []
        current_date = start_date
        total_days = (end_date - start_date).days
        processed_days = 0
        
        while current_date < end_date:
            next_date = current_date + timedelta(days=1)
            start_ts = int(current_date.timestamp())
            end_ts = int(next_date.timestamp())
            
            try:
                pollution_url = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={self.lat}&lon={self.lon}&start={start_ts}&end={end_ts}&appid={OPENWEATHER_API_KEY}"
                weather_url = f"https://api.openweathermap.org/data/3.0/onecall/timemachine?lat={self.lat}&lon={self.lon}&dt={start_ts}&appid={OPENWEATHER_API_KEY}&units=metric"
                
                pollution_resp = requests.get(pollution_url, timeout=10).json()
                weather_resp = requests.get(weather_url, timeout=10).json()
                
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
                
                processed_days += 1
                progress = (processed_days / total_days) * 100
                print(f"  [{processed_days}/{total_days}] {current_date.date()} ({progress:.1f}%)")
                
            except requests.exceptions.RequestException as e:
                print(f"  ERROR on {current_date.date()}: {str(e)}")
                processed_days += 1
            
            current_date = next_date
            time.sleep(1)
        
        if len(all_data) == 0:
            print("\nWARNING: No data extracted!")
            return None
            
        df = pd.DataFrame(all_data).sort_values("datetime").reset_index(drop=True)
        print(f"\nExtracted {len(df)} records")
        return df
    
    def engineer_features(self, new_data, historical_data):
        print("\n" + "="*60)
        print("ENGINEERING FEATURES")
        print("="*60)
        
        raw_cols = ['datetime', 'aqi', 'co', 'no2', 'pm2_5', 'pm10', 
                    'temp', 'feels_like', 'pressure', 'dew_point', 
                    'wind_speed', 'wind_deg']
        historical_raw = historical_data[raw_cols].copy()
        
        combined = pd.concat([historical_raw, new_data], ignore_index=True)
        combined = combined.sort_values('datetime').reset_index(drop=True)
        
        start_idx = len(historical_raw)
        
        print(f"Processing {len(new_data)} new records for feature engineering...")
        
        for idx in range(start_idx, len(combined)):
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
        
        new_engineered = combined.iloc[start_idx:].copy()
        
        print(f"Engineered {len(new_engineered)} new records")
        return new_engineered
    
    def upload(self, data):
        print("\n" + "="*60)
        print("APPENDING TO FEATURE STORE")
        print("="*60)
        
        data = data.fillna(0)
        data['datetime'] = pd.to_datetime(data['datetime'], utc=True)
        data['timestamp'] = (data['datetime'].astype('int64') // 10**6).astype('int64')
        
        data['is_weekend'] = data['is_weekend'].astype('int64')
        data['aqi'] = data['aqi'].astype('int64')
        
        print(f"Uploading {len(data)} records...")
        print(f"Date range: {data['datetime'].min()} to {data['datetime'].max()}")
        print(f"Timestamp range: {data['timestamp'].min()} to {data['timestamp'].max()}")
        
        self.fg.insert(data, write_options={"start_offline_materialization": True})
        self.fg.commit_details(wallclock_time=None, limit=None)
        
        print(f"Successfully appended {len(data)} records")
    
    def verify(self):
        print("\n" + "="*60)
        print("VERIFYING APPEND")
        print("="*60)
        
        df = self.fg.read(online=True)
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        print(f"Total records now: {len(df)}")
        print(f"Latest date: {df['datetime'].max()}")
        print(f"Earliest date: {df['datetime'].min()}")
        print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        
        return True
    
    def run(self, start_date_str, end_date_str):
        print("\n" + "="*80)
        print("HISTORICAL BACKFILL - APPEND HISTORICAL DATA")
        print("="*80)
        
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        
        print(f"\nBackfill period: {start_date_str} to {end_date_str}")
        print(f"Total days: {(end_date - start_date).days}")
        
        self.connect_hopsworks()
        existing = self.get_existing_data()
            
        new_raw = self.extract_data(start_date, end_date)
        
        if new_raw is None or len(new_raw) == 0:
            print("\nNo data extracted. Exiting.")
            return
        
        new_featured = self.engineer_features(new_raw, existing)
        
        self.upload(new_featured)
        
        self.verify()
        
        print("\n" + "="*80)
        print("HISTORICAL BACKFILL COMPLETE")
        print("="*80)

def main():
    start_date = "2025-01-01"
    end_date = "2025-10-04"
    
    HistoricalBackfill().run(start_date, end_date)

if __name__ == "__main__":
    main()
