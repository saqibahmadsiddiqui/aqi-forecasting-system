# import pandas as pd
# import numpy as np
# import joblib
# import hopsworks
# from datetime import datetime, timedelta
# import pytz
# from pathlib import Path
# import sys

# sys.path.append(str(Path(__file__).parent.parent.parent))
# from src.config.config import *

# class AQIPredictor:
#     def __init__(self):
#         self.project = None
#         self.mr = None
#         self.fs = None
#         self.models = {}
#         self.model_metrics = {}
#         self.best_model_name = None
        
#         self.api_key = HOPSWORKS_API_KEY
#         self.project_name = HOPSWORKS_PROJECT_NAME
        
#     def connect_hopsworks(self):
#         self.project = hopsworks.login(
#             host=HOPSWORKS_HOST,
#             api_key_value=self.api_key,
#             project=self.project_name
#         )
#         self.mr = self.project.get_model_registry()
#         self.fs = self.project.get_feature_store()
#         print("Connected")
        
#     def load_all_models(self):
#         print("\nLoading models...")
        
#         registry_names = {
#             "random_forest": "aqi_random_forest",
#             "xgboost": "aqi_xgboost",
#             "lightgbm": "aqi_lightgbm"
#         }
        
#         for local_name, registry_name in registry_names.items():
#             try:
#                 model = self.mr.get_model(registry_name, version=1)
#                 model_dir = model.download()
#                 model_path = Path(model_dir) / "model.joblib"
                
#                 self.models[local_name] = joblib.load(model_path)
                
#                 metrics = model.training_metrics
#                 self.model_metrics[local_name] = {
#                     'mae': metrics.get('mae', 0),
#                     'rmse': metrics.get('rmse', 0),
#                     'r2_score': metrics.get('r2_score', 0)
#                 }
                
#                 print(f"{local_name}: R² Score={self.model_metrics[local_name]['r2_score']:.3f}")
                
#             except Exception as e:
#                 print(f"{registry_name}: {e}")
        
#         self.best_model_name = max(self.model_metrics.items(), key=lambda x: x[1]['r2_score'])[0]
#         print(f"\nBest Model Selected (R²): {self.best_model_name}")
        
#     def get_latest_features(self):
#         print("Fetching latest features from Online Store...")
#         fg = self.fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
#         df = fg.read(online=True) 
        
#         df['datetime'] = pd.to_datetime(df['datetime'])
#         return df.sort_values('datetime')
    
#     def create_future_features(self, latest_data, prediction_date):
#         last_row = latest_data.iloc[-1].copy()
#         last_row['datetime'] = prediction_date
        
#         last_row['hour_sin'] = np.sin(2 * np.pi * prediction_date.hour / 24)
#         last_row['hour_cos'] = np.cos(2 * np.pi * prediction_date.hour / 24)
#         last_row['day_of_week_sin'] = np.sin(2 * np.pi * prediction_date.weekday() / 7)
#         last_row['day_of_week_cos'] = np.cos(2 * np.pi * prediction_date.weekday() / 7)
#         last_row['month_sin'] = np.sin(2 * np.pi * prediction_date.month / 12)
#         last_row['month_cos'] = np.cos(2 * np.pi * prediction_date.month / 12)
#         last_row['is_weekend'] = 1 if prediction_date.weekday() >= 5 else 0
        
#         return last_row
    
#     def predict_next_3_days(self):
#         print("\n" + "="*60)
#         print("PREDICTING NEXT 3 DAYS")
#         print("="*60)
        
#         latest_data = self.get_latest_features()
#         print(f"Using context data up to: {latest_data['datetime'].max()}") # Verify latest date
        
#         pkt = pytz.timezone(TIMEZONE)
#         today_pkt = datetime.now(pkt).date()
        
#         predictions = []
        
#         all_cols = latest_data.columns
#         feature_cols = [col for col in all_cols if col not in ['datetime', 'timestamp', 'aqi']]
        
#         for day_offset in range(1, 4):
#             target_date = today_pkt + timedelta(days=day_offset)
#             hourly_preds = []
            
#             for hour in range(24):
#                 prediction_datetime = pkt.localize(
#                     datetime.combine(target_date, datetime.min.time()) + timedelta(hours=hour)
#                 )
                
#                 features_row = self.create_future_features(latest_data, prediction_datetime)
                
#                 # Ensure input is a DataFrame with feature names to avoid XGBoost/LightGBM warnings
#                 X = pd.DataFrame([features_row[feature_cols]])
                
#                 pred_aqi = self.models[self.best_model_name].predict(X)[0]
#                 hourly_preds.append(pred_aqi)
            
#             avg_aqi = np.mean(hourly_preds)
#             category = self._get_category(avg_aqi)
#             warning = self._get_warning(avg_aqi)
            
#             predictions.append({
#                 'date': target_date.strftime('%Y-%m-%d'),
#                 'day_name': target_date.strftime('%A'),
#                 'average_aqi': round(avg_aqi, 2),
#                 'min_aqi': round(min(hourly_preds), 2),
#                 'max_aqi': round(max(hourly_preds), 2),
#                 'category': category,
#                 'warning': warning
#             })
            
#             print(f"\n{target_date.strftime('%A, %B %d, %Y')}:")
#             print(f"  Avg AQI: {avg_aqi:.2f} ({category})")
#             if warning:
#                 print(f" WARNING: {warning}")
        
#         return predictions
    
#     def _get_category(self, aqi):
#         if aqi <= 1.5:
#             return 'Good'
#         elif aqi <= 2.5:
#             return 'Fair'
#         elif aqi <= 3.5:
#             return 'Moderate'
#         elif aqi <= 4.5:
#             return 'Poor'
#         else:
#             return 'Very Poor'
    
#     def _get_warning(self, aqi):
#         if aqi >= 4.5:
#             return "Hazardous air quality! Avoid outdoor activities."
#         elif aqi >= 3.5:
#             return "Poor air quality. Sensitive groups should limit outdoor exposure."
#         return None
    
#     def get_model_comparison(self):
#         comparison = []
#         for model_name, metrics in self.model_metrics.items():
#             comparison.append({
#                 'model': model_name.replace('_', ' ').title(),
#                 'mae': metrics['mae'],
#                 'rmse': metrics['rmse'],
#                 'r2_score': metrics['r2_score'],
#                 'is_best': model_name == self.best_model_name
#             })
#         return sorted(comparison, key=lambda x: x['r2_score'], reverse=True)

# def main():
#     predictor = AQIPredictor()
#     predictor.connect_hopsworks()
#     predictor.load_all_models()
#     predictions = predictor.predict_next_3_days()
    
#     PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
#     pd.DataFrame(predictions).to_csv(PROCESSED_DATA_DIR / 'latest_predictions.csv', index=False)
#     pd.DataFrame(predictor.get_model_comparison()).to_csv(PROCESSED_DATA_DIR / 'model_comparison.csv', index=False)
    
#     print("\nPredictions saved")

# if __name__ == "__main__":
#     main()

import pandas as pd
import numpy as np
import joblib
import hopsworks
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Ensure local imports work
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config.config import *


class AQIPredictor:
    def __init__(self):
        self.project = None
        self.mr = None
        self.fs = None
        self.models = {}
        self.model_metrics = {}
        self.best_model_name = None
        
        self.api_key = HOPSWORKS_API_KEY
        self.project_name = HOPSWORKS_PROJECT_NAME
        
    def connect_hopsworks(self):
        self.project = hopsworks.login(
            host=HOPSWORKS_HOST,
            api_key_value=self.api_key,
            project=self.project_name
        )
        self.mr = self.project.get_model_registry()
        self.fs = self.project.get_feature_store()
        print("Connected to Hopsworks")
        
    def load_latest_models(self):
        print("\nLoading the LATEST Classification models...")
        registry_names = {
            "random_forest": "aqi_random_forest",
            "gradient_boosting": "aqi_gradient_boosting",
            "lightgbm": "aqi_lightgbm"
        }
        
        scores = {}
        for local_name, registry_name in registry_names.items():
            try:
                # Use get_models() to retrieve ALL versions as a list
                versions = self.mr.get_models(registry_name)
                
                # Sort versions by version number (descending) and pick the first one
                # This ensures we get v14 instead of v1
                model = sorted(versions, key=lambda m: m.version, reverse=True)[0]
                
                model_dir = model.download()
                model_path = Path(model_dir) / "model.joblib"
                
                self.models[local_name] = joblib.load(model_path)
                
                # Retrieve the classifier metrics
                f1 = model.training_metrics.get('f1_score', 0)
                scores[local_name] = f1
                self.model_metrics[local_name] = {
                    'f1_score': f1,
                    'accuracy': model.training_metrics.get('accuracy', 0),
                    'version': model.version
                }
                print(f"Successfully loaded {local_name} v{model.version} (F1: {f1:.3f})")
                
            except Exception as e:
                print(f"Error loading {registry_name}: {e}")
        
        # Select the best model name based on F1 Score
        self.best_model_name = max(scores, key=scores.get)
        print(f"Best Model Selected: {self.best_model_name}")



    
    def get_latest_features(self):
        print("Fetching context for recursive forecasting...")
        fg = self.fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
        df = fg.read(online=True).sort_values('datetime')
        df['datetime'] = pd.to_datetime(df['datetime'])
        return df

    def predict_next_3_days(self):
        print("\n" + "="*40)
        print("GENERATING 72-HOUR RECURSIVE FORECAST")
        print("="*40)
        
        # Initialize history with last 48 hours
        history = self.get_latest_features().tail(48)
        feature_cols = [c for c in history.columns if c not in ['datetime', 'timestamp', 'aqi']]
        
        last_dt = history['datetime'].max()
        
        # Recursive Prediction Loop
        for i in range(1, 73):
            current_dt = last_dt + timedelta(hours=i)
            new_row = history.iloc[-1].copy()
            new_row['datetime'] = current_dt
            
            # Update Cyclical Time Features
            new_row['hour_sin'] = np.sin(2 * np.pi * current_dt.hour / 24)
            new_row['hour_cos'] = np.cos(2 * np.pi * current_dt.hour / 24)
            new_row['day_of_week_sin'] = np.sin(2 * np.pi * current_dt.weekday() / 7)
            new_row['is_weekend'] = 1 if current_dt.weekday() >= 5 else 0
            
            # UPDATE LAGS (Matches your Engineering Pipeline)
            lags = [1, 3, 6, 12, 24, 48]
            for lag in lags:
                col = f'aqi_lag_{lag}h'
                if col in new_row:
                    new_row[col] = history.iloc[-lag]['aqi']

            # UPDATE ROLLING MEANS
            windows = [3, 6, 12, 24]
            for w in windows:
                col = f'aqi_rolling_mean_{w}h'
                if col in new_row:
                    new_row[col] = history['aqi'].tail(w).mean()

            # PREDICT (Now using 1-5 direct classification)
            X = pd.DataFrame([new_row[feature_cols]])
            pred_class = self.models[self.best_model_name].predict(X)[0]
            
            # Update local history for the next iteration's lags
            new_row['aqi'] = pred_class
            history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)

        # Aggregate 72 Hourly Predictions into 3 Daily Summaries
        final_predictions = []
        for day_offset in range(1, 4):
            target_date = (last_dt + timedelta(days=day_offset)).date()
            day_slice = history[history['datetime'].dt.date == target_date]
            
            # Categorical average (using the most frequent or rounded mean)
            avg_aqi = int(round(day_slice['aqi'].mean()))
            
            final_predictions.append({
                'date': target_date.strftime('%Y-%m-%d'),
                'day_name': target_date.strftime('%A'),
                'average_aqi': avg_aqi,
                'min_aqi': int(day_slice['aqi'].min()),
                'max_aqi': int(day_slice['aqi'].max()),
                'category': self._get_label(avg_aqi),
                'warning': self._get_warning(avg_aqi)
            })
            print(f"{target_date}: Predicted Category {avg_aqi}")
            
        return final_predictions

    def _get_label(self, aqi):
        mapping = {1: 'Good', 2: 'Fair', 3: 'Moderate', 4: 'Poor', 5: 'Very Poor'}
        return mapping.get(int(aqi), "Hazardous")

    def _get_warning(self, aqi):
        if aqi >= 5: return "Hazardous! Avoid all outdoor exertion."
        if aqi >= 4: return "Poor quality. Limit outdoor exposure."
        return None

    def get_model_comparison(self):
        comparison = []
        for model_name, metrics in self.model_metrics.items():
            comparison.append({
                'model': model_name.replace('_', ' ').title(),
                'f1_score': metrics['f1_score'],
                'accuracy': metrics['accuracy'],
                'version': metrics['version'],
                'is_best': model_name == self.best_model_name
            })
        return sorted(comparison, key=lambda x: x['f1_score'], reverse=True)

def main():
    predictor = AQIPredictor()
    predictor.connect_hopsworks()
    predictor.load_latest_models()
    predictions = predictor.predict_next_3_days()
    
    # Save CSV for UI and API
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(predictions).to_csv(PROCESSED_DATA_DIR / 'latest_predictions.csv', index=False)
    pd.DataFrame(predictor.get_model_comparison()).to_csv(PROCESSED_DATA_DIR / 'model_comparison.csv', index=False)
    print("\nRecursive Forecast CSV Saved.")

if __name__ == "__main__":
    main()
