import pandas as pd
import numpy as np
import joblib
import hopsworks
from datetime import datetime, timedelta
import pytz
from pathlib import Path
import sys
import os
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config.config import *

load_dotenv()

class AQIPredictor:
    def __init__(self):
        self.project = None
        self.mr = None
        self.fs = None
        self.models = {}
        self.model_metrics = {}
        self.best_model_name = None
        
        self.api_key = os.getenv("HOPSWORKS_API_KEY")
        self.project_name = os.getenv("HOPSWORKS_PROJECT_NAME")
        
    def connect_hopsworks(self):
        self.project = hopsworks.login(
            host=HOPSWORKS_HOST,
            api_key_value=self.api_key,
            project=self.project_name
        )
        self.mr = self.project.get_model_registry()
        self.fs = self.project.get_feature_store()
        print("Connected")
        
    def load_all_models(self):
        print("\nLoading models...")
        
        registry_names = {
            "random_forest": "aqi_random_forest",
            "xgboost": "aqi_xgboost",
            "lightgbm": "aqi_lightgbm"
        }
        
        for local_name, registry_name in registry_names.items():
            try:
                model = self.mr.get_model(registry_name, version=1)
                model_dir = model.download()
                model_path = Path(model_dir) / "model.joblib"
                
                self.models[local_name] = joblib.load(model_path)
                
                metrics = model.training_metrics
                self.model_metrics[local_name] = {
                    'mae': metrics.get('mae', 0),
                    'rmse': metrics.get('rmse', 0),
                    'r2_score': metrics.get('r2_score', 0)
                }
                
                print(f"{local_name}: R² Score={self.model_metrics[local_name]['r2_score']:.3f}")
                
            except Exception as e:
                print(f"{registry_name}: {e}")
        
        self.best_model_name = max(self.model_metrics.items(), key=lambda x: x[1]['r2_score'])[0]
        print(f"\nBest Model Selected (R²): {self.best_model_name}")
        
    def get_latest_features(self):
        print("Fetching latest features from Online Store...")
        fg = self.fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
        df = fg.read(online=True) 
        
        df['datetime'] = pd.to_datetime(df['datetime'])
        return df.sort_values('datetime')
    
    def create_future_features(self, latest_data, prediction_date):
        last_row = latest_data.iloc[-1].copy()
        last_row['datetime'] = prediction_date
        
        last_row['hour_sin'] = np.sin(2 * np.pi * prediction_date.hour / 24)
        last_row['hour_cos'] = np.cos(2 * np.pi * prediction_date.hour / 24)
        last_row['day_of_week_sin'] = np.sin(2 * np.pi * prediction_date.weekday() / 7)
        last_row['day_of_week_cos'] = np.cos(2 * np.pi * prediction_date.weekday() / 7)
        last_row['month_sin'] = np.sin(2 * np.pi * prediction_date.month / 12)
        last_row['month_cos'] = np.cos(2 * np.pi * prediction_date.month / 12)
        last_row['is_weekend'] = 1 if prediction_date.weekday() >= 5 else 0
        
        return last_row
    
    def predict_next_3_days(self):
        print("\n" + "="*60)
        print("PREDICTING NEXT 3 DAYS")
        print("="*60)
        
        latest_data = self.get_latest_features()
        print(f"Using context data up to: {latest_data['datetime'].max()}") # Verify latest date
        
        pkt = pytz.timezone(TIMEZONE)
        now = datetime.now(pkt)
        
        predictions = []
        
        all_cols = latest_data.columns
        feature_cols = [col for col in all_cols if col not in ['datetime', 'timestamp', 'aqi']]
        
        for day_offset in range(1, 4):
            target_date = (now + timedelta(days=day_offset)).date()
            hourly_preds = []
            
            for hour in range(24):
                prediction_datetime = pkt.localize(
                    datetime.combine(target_date, datetime.min.time()) + timedelta(hours=hour)
                )
                
                features_row = self.create_future_features(latest_data, prediction_datetime)
                
                # Ensure input is a DataFrame with feature names to avoid XGBoost/LightGBM warnings
                X = pd.DataFrame([features_row[feature_cols]])
                
                pred_aqi = self.models[self.best_model_name].predict(X)[0]
                hourly_preds.append(pred_aqi)
            
            avg_aqi = np.mean(hourly_preds)
            category = self._get_category(avg_aqi)
            warning = self._get_warning(avg_aqi)
            
            predictions.append({
                'date': target_date.strftime('%Y-%m-%d'),
                'day_name': target_date.strftime('%A'),
                'average_aqi': round(avg_aqi, 2),
                'min_aqi': round(min(hourly_preds), 2),
                'max_aqi': round(max(hourly_preds), 2),
                'category': category,
                'warning': warning
            })
            
            print(f"\n{target_date.strftime('%A, %B %d, %Y')}:")
            print(f"  Avg AQI: {avg_aqi:.2f} ({category})")
            if warning:
                print(f" WARNING: {warning}")
        
        return predictions
    
    def _get_category(self, aqi):
        if aqi <= 1.5:
            return 'Good'
        elif aqi <= 2.5:
            return 'Fair'
        elif aqi <= 3.5:
            return 'Moderate'
        elif aqi <= 4.5:
            return 'Poor'
        else:
            return 'Very Poor'
    
    def _get_warning(self, aqi):
        if aqi >= 4.5:
            return "Hazardous air quality! Avoid outdoor activities."
        elif aqi >= 3.5:
            return "Poor air quality. Sensitive groups should limit outdoor exposure."
        return None
    
    def get_model_comparison(self):
        comparison = []
        for model_name, metrics in self.model_metrics.items():
            comparison.append({
                'model': model_name.replace('_', ' ').title(),
                'mae': metrics['mae'],
                'rmse': metrics['rmse'],
                'r2_score': metrics['r2_score'],
                'is_best': model_name == self.best_model_name
            })
        return sorted(comparison, key=lambda x: x['r2_score'], reverse=True)

def main():
    predictor = AQIPredictor()
    predictor.connect_hopsworks()
    predictor.load_all_models()
    predictions = predictor.predict_next_3_days()
    
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    pd.DataFrame(predictions).to_csv(PROCESSED_DATA_DIR / 'latest_predictions.csv', index=False)
    pd.DataFrame(predictor.get_model_comparison()).to_csv(PROCESSED_DATA_DIR / 'model_comparison.csv', index=False)
    
    print("\nPredictions saved")

if __name__ == "__main__":
    main()