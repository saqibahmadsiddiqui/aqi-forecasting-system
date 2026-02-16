import pandas as pd
import numpy as np
import joblib
import hopsworks
from datetime import datetime, timedelta
import pytz
from pathlib import Path
import sys

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
        print("Connecting to Hopsworks...")
        self.project = hopsworks.login(
            host=HOPSWORKS_HOST,
            api_key_value=self.api_key,
            project=self.project_name
        )
        self.mr = self.project.get_model_registry()
        self.fs = self.project.get_feature_store()
        print("Connected to Hopsworks")
        
    def load_latest_models(self):
        """Load the latest versions of all 5 models from registry"""
        print("\nLoading the LATEST 5 Classification Models...")
        
        registry_names = {
            "random_forest": "aqi_random_forest",
            "gradient_boosting": "aqi_gradient_boosting",
            "lightgbm": "aqi_lightgbm",
            "decision_tree": "aqi_decision_tree",
            "sklearn_gradient_boosting": "aqi_sklearn_gradient_boosting"
        }
        
        scores = {}
        
        for local_name, registry_name in registry_names.items():
            try:
                versions = self.mr.get_models(registry_name)
                
                if not versions:
                    print(f"    No versions found for {registry_name}")
                    continue
                
                latest_model = sorted(versions, key=lambda m: m.version, reverse=True)[0]
                
                model_dir = latest_model.download()
                model_path = Path(model_dir) / "model.joblib"
                
                self.models[local_name] = joblib.load(model_path)
                
                f1 = latest_model.training_metrics.get('f1_score', 0)
                accuracy = latest_model.training_metrics.get('accuracy', 0)
                
                scores[local_name] = f1
                self.model_metrics[local_name] = {
                    'f1_score': f1,
                    'accuracy': accuracy,
                    'precision': latest_model.training_metrics.get('precision', 0),
                    'recall': latest_model.training_metrics.get('recall', 0),
                    'version': latest_model.version
                }
                
                print(f"   {local_name:30} v{latest_model.version:3} | F1: {f1:.4f} | Acc: {accuracy:.4f}")
                
            except Exception as e:
                print(f"   Error loading {registry_name}: {str(e)}")
        
        if not scores:
            raise Exception("No models loaded successfully!")
        
        self.best_model_name = max(scores, key=scores.get)
        print(f"\nBest Model Selected: {self.best_model_name} (F1: {scores[self.best_model_name]:.4f})")
    
    def get_latest_features(self):
        """Fetch latest feature data from online feature store"""
        print("\nFetching feature context from Online Feature Store...")
        fg = self.fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
        df = fg.read(online=True).sort_values('datetime')
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        print(f"   Loaded {len(df)} records")
        print(f"   Latest datetime: {df['datetime'].max()}")
        
        return df
    
    def predict_next_3_days(self):
        """Generate 72-hour recursive forecast with 24-hour aggregation"""
        print("\n" + "="*70)
        print("GENERATING 72-HOUR RECURSIVE FORECAST")
        print("="*70)
        
        history = self.get_latest_features().tail(48).copy()
        
        feature_cols = [c for c in history.columns 
                       if c not in ['datetime', 'timestamp', 'aqi']]
        
        last_dt = history['datetime'].max()
        pkt = pytz.timezone(TIMEZONE)
        
        print(f"   Starting from: {last_dt}")
        print(f"   Prediction window: 72 hours (3 days)")
        print(f"   Features used: {len(feature_cols)}")
        
        hourly_predictions = []
        
        for hour_offset in range(1, 73):
            current_dt = last_dt + timedelta(hours=hour_offset)
            
            new_row = history.iloc[-1].copy()
            new_row['datetime'] = current_dt
            
            # Update cyclical time features
            new_row['hour_sin'] = np.sin(2 * np.pi * current_dt.hour / 24)
            new_row['hour_cos'] = np.cos(2 * np.pi * current_dt.hour / 24)
            new_row['day_of_week_sin'] = np.sin(2 * np.pi * current_dt.weekday() / 7)
            new_row['day_of_week_cos'] = np.cos(2 * np.pi * current_dt.weekday() / 7)
            new_row['month_sin'] = np.sin(2 * np.pi * current_dt.month / 12)
            new_row['month_cos'] = np.cos(2 * np.pi * current_dt.month / 12)
            new_row['is_weekend'] = 1 if current_dt.weekday() >= 5 else 0
            
            # Update lag features
            lags = [1, 3, 6, 12, 24, 48]
            for lag in lags:
                col = f'aqi_lag_{lag}h'
                if col in new_row.index and lag <= len(history):
                    new_row[col] = history.iloc[-lag]['aqi']
            
            # Update rolling features
            windows = [3, 6, 12, 24]
            for w in windows:
                col = f'aqi_rolling_mean_{w}h'
                if col in new_row.index:
                    new_row[col] = history['aqi'].tail(w).mean()
            
            col = 'aqi_rolling_std_6h'
            if col in new_row.index:
                new_row[col] = history['aqi'].tail(6).std()
            
            col = 'aqi_rolling_std_24h'
            if col in new_row.index:
                new_row[col] = history['aqi'].tail(24).std()
            
            col = 'pm2_5_rolling_mean_6h'
            if col in new_row.index:
                new_row[col] = history['pm2_5'].tail(6).mean()
            
            col = 'pm2_5_rolling_mean_24h'
            if col in new_row.index:
                new_row[col] = history['pm2_5'].tail(24).mean()
            
            col = 'aqi_change_24h'
            if col in new_row.index and len(history) >= 24:
                new_row[col] = history.iloc[-1]['aqi'] - history.iloc[-24]['aqi']
            
            col = 'pm2_5_x_wind_speed'
            if col in new_row.index:
                new_row[col] = history.iloc[-1]['pm2_5'] * history.iloc[-1]['wind_speed']
            
            # Predict
            X = pd.DataFrame([new_row[feature_cols]])
            pred_class = self.models[self.best_model_name].predict(X)[0]
            
            new_row['aqi'] = pred_class
            hourly_predictions.append({
                'datetime': current_dt,
                'aqi': pred_class
            })
            
            history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)
            
            if hour_offset % 24 == 0:
                print(f"   Predicted {hour_offset} hours")
        
        # Aggregate into 3 daily predictions
        df_hourly = pd.DataFrame(hourly_predictions)
        df_hourly['datetime'] = pd.to_datetime(df_hourly['datetime'])
        
        final_predictions = []
        
        for day_offset in range(1, 4):
            target_date = (last_dt + timedelta(days=day_offset)).date()
            
            day_slice = df_hourly[df_hourly['datetime'].dt.date == target_date]
            
            if len(day_slice) == 0:
                continue
            
            avg_aqi = int(round(day_slice['aqi'].mean()))
            avg_aqi = max(1, min(5, avg_aqi))
            
            final_predictions.append({
                'date': target_date.strftime('%Y-%m-%d'),
                'day_name': target_date.strftime('%A'),
                'average_aqi': avg_aqi,
                'min_aqi': int(day_slice['aqi'].min()),
                'max_aqi': int(day_slice['aqi'].max()),
                'category': self._get_label(avg_aqi),
                'warning': self._get_warning(avg_aqi),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            
            print(f"   {target_date.strftime('%A, %B %d')}: Category {avg_aqi} ({self._get_label(avg_aqi)})")
        
        return final_predictions
    
    def _get_label(self, aqi):
        """Convert AQI (1-5) to category label"""
        mapping = {
            1: '🟢 Good',
            2: '🟡 Fair',
            3: '🟠 Moderate',
            4: '🔴 Poor',
            5: '🔴 Very Poor'
        }
        return mapping.get(int(aqi), "❓ Unknown")
    
    def _get_warning(self, aqi):
        """Generate health warning based on AQI"""
        if aqi >= 5:
            return "⚠️  HAZARDOUS! Avoid all outdoor exertion."
        if aqi >= 4:
            return "⚠️  POOR! Limit outdoor exposure, especially for sensitive groups."
        if aqi >= 3:
            return "⚠️  MODERATE! Outdoor activities may affect sensitive individuals."
        return None
    
    def get_model_comparison(self):
        """Get performance metrics for all loaded models"""
        comparison = []
        for model_name, metrics in self.model_metrics.items():
            comparison.append({
                'model': model_name.replace('_', ' ').title(),
                'f1_score': round(metrics['f1_score'], 4),
                'accuracy': round(metrics['accuracy'], 4),
                'precision': round(metrics['precision'], 4),
                'recall': round(metrics['recall'], 4),
                'version': metrics['version'],
                'is_best': model_name == self.best_model_name
            })
        return sorted(comparison, key=lambda x: x['f1_score'], reverse=True)
    
    def run(self):
        """Execute prediction pipeline"""
        print("\n" + "="*80)
        print("AQI PREDICTION PIPELINE")
        print(f"Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print("="*80)
        
        try:
            self.connect_hopsworks()
            self.load_latest_models()
            predictions = self.predict_next_3_days()
            comparison = self.get_model_comparison()
            
            PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
            
            df_pred = pd.DataFrame(predictions)
            df_pred.to_csv(PROCESSED_DATA_DIR / 'latest_predictions.csv', index=False)
            print(f"\nSaved predictions to: {PROCESSED_DATA_DIR / 'latest_predictions.csv'}")
            
            df_comp = pd.DataFrame(comparison)
            df_comp.to_csv(PROCESSED_DATA_DIR / 'model_comparison.csv', index=False)
            print(f"Saved model comparison to: {PROCESSED_DATA_DIR / 'model_comparison.csv'}")
            
            print("\n" + "="*80)
            print("PREDICTION PIPELINE COMPLETE")
            print("="*80)
            print(f"\n3-Day Forecast Summary:")
            for pred in predictions:
                print(f"   {pred['date']} ({pred['day_name']}): {pred['category']}")
                if pred['warning']:
                    print(f"      {pred['warning']}")
            
            print(f"\nModel Rankings:")
            for i, comp in enumerate(comparison, 1):
                best_marker = " ⭐" if comp['is_best'] else ""
                print(f"   {i}. {comp['model']:30} | F1: {comp['f1_score']:.4f} | Acc: {comp['accuracy']:.4f}{best_marker}")
            
            print("\n" + "="*80)
            
        except Exception as e:
            print(f"\nERROR in prediction pipeline: {str(e)}")
            raise

def main():
    predictor = AQIPredictor()
    predictor.run()

if __name__ == "__main__":
    main()
