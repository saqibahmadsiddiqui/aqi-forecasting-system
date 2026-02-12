# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# import xgboost as xgb
# import lightgbm as lgb
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import hopsworks
# import joblib
# import tempfile
# from pathlib import Path
# import sys

# sys.path.append(str(Path(__file__).parent.parent.parent))
# from src.config.config import *

# load_dotenv()

# class DailyTraining:
#     def __init__(self):
#         self.hopsworks_key = HOPSWORKS_API_KEY
#         self.project_name = HOPSWORKS_PROJECT_NAME
#         self.project = None
#         self.fs = None
#         self.mr = None
        
#     def connect_hopsworks(self):
#         self.project = hopsworks.login(
#             host=HOPSWORKS_HOST,
#             api_key_value=self.hopsworks_key,
#             project=self.project_name
#         )
#         self.fs = self.project.get_feature_store()
#         self.mr = self.project.get_model_registry()
#         print("Connected")
    
#     def load_data(self):
#         print("\nLoading data from Online Store...")
#         fg = self.fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
        
#         df = fg.read(online=True)
        
#         df['datetime'] = pd.to_datetime(df['datetime'])
#         df = df.sort_values('datetime').reset_index(drop=True)
        
#         print(f"{len(df)} records retrieved from {df['datetime'].min()} to {df['datetime'].max()}")
#         return df
    
#     def prepare_data(self, df):
#         features = [col for col in df.columns if col not in ['datetime', 'timestamp', 'aqi']]
#         X = df[features]
#         y = df['aqi']
        
#         print(f"{len(features)} features")
#         return X, y
    
#     def train_models(self, X_train, X_test, y_train, y_test):
#         print("\nTraining models...")
        
#         models = {
#             "random_forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
#             "xgboost": xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1),
#             "lightgbm": lgb.LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)
#         }
        
#         results = {}
        
#         for name, model in models.items():
#             print(f"\n{name}...")
#             model.fit(X_train, y_train)
#             preds = model.predict(X_test)
            
#             mae = mean_absolute_error(y_test, preds)
#             rmse = np.sqrt(mean_squared_error(y_test, preds))
#             r2 = r2_score(y_test, preds)
            
#             results[name] = {
#                 'model': model,
#                 'mae': mae,
#                 'rmse': rmse,
#                 'r2_score': r2
#             }
            
#             print(f"  MAE: {mae:.3f}, RMSE: {rmse:.3f}, R²: {r2:.3f}")
        
#         return results
    
#     def register_models(self, results):
#         print("\nRegistering models...")
        
#         for name, data in results.items():
#             with tempfile.TemporaryDirectory() as model_dir:
#                 model_path = Path(model_dir) / "model.joblib"
#                 joblib.dump(data['model'], model_path)
                
#                 metrics = {
#                     "mae": float(data['mae']),
#                     "rmse": float(data['rmse']),
#                     "r2_score": float(data['r2_score'])
#                 }
                
#                 registered = self.mr.python.create_model(
#                     name=f"aqi_{name}",
#                     metrics=metrics,
#                     description=f"{name} for AQI prediction"
#                 )
                
#                 version = registered.save(model_dir)
#                 print(f"{name} v{version.version}")
                
#                 # Save locally
#                 local_path = MODELS_DIR / f"{name}.joblib"
#                 joblib.dump(data['model'], local_path)
        
#         best = max(results.items(), key=lambda x: x[1]['r2_score'])
#         print(f"\nBest: {best[0]} (R² Score={best[1]['r2_score']:.3f})")
    
#     def run(self):
#         print("\n" + "="*60)
#         print("DAILY TRAINING")
#         print("="*60)
        
#         self.connect_hopsworks()
#         df = self.load_data()
#         X, y = self.prepare_data(df)
        
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=0.2, random_state=42, shuffle=False
#         )
        
#         print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")
        
#         results = self.train_models(X_train, X_test, y_train, y_test)
#         self.register_models(results)
        
#         print("\nCOMPLETE")

# def main():
#     DailyTraining().run()

# if __name__ == "__main__":
#     main()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score
import hopsworks
import joblib
import tempfile
from pathlib import Path
import sys
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config.config import *

load_dotenv()

class DailyTraining:
    def __init__(self):
        self.project = None
        self.fs = None
        self.mr = None
        
    def connect_hopsworks(self):
        self.project = hopsworks.login(
            host=HOPSWORKS_HOST, 
            api_key_value=HOPSWORKS_API_KEY, 
            project=HOPSWORKS_PROJECT_NAME
        )
        self.fs = self.project.get_feature_store()
        self.mr = self.project.get_model_registry()
        print("Connected to Hopsworks")
    
    def load_data(self):
        fg = self.fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
        df = fg.read(online=True)
        df['datetime'] = pd.to_datetime(df['datetime'])
        return df.sort_values('datetime').reset_index(drop=True)
    
    def train_models(self, X_train, X_test, y_train, y_test):
        from sklearn.ensemble import HistGradientBoostingClassifier
        print("\nTraining Classifiers...")
        
        # We use the original 1-5 labels. Scikit-Learn models handle this perfectly.
        y_train_clean = y_train.values.flatten().astype(int)
        y_test_clean = y_test.values.flatten().astype(int)

        models_to_train = {
            "random_forest": RandomForestClassifier(
                n_estimators=100, 
                class_weight='balanced', 
                random_state=42,
                n_jobs=-1
            ),
            "gradient_boosting": HistGradientBoostingClassifier(
                max_iter=100,
                random_state=42,
                class_weight='balanced' # Great for your 95% imbalance
            ),
            "lightgbm": lgb.LGBMClassifier(
                n_estimators=100, 
                class_weight='balanced',
                random_state=42, 
                n_jobs=-1, 
                verbose=-1
            )
        }
        
        results = {}
        for name, model in models_to_train.items():
            print(f"Training {name}...")
            
            # Simple fit using original 1-5 categories
            model.fit(X_train, y_train_clean)
            preds = model.predict(X_test)
            
            acc = accuracy_score(y_test_clean, preds)
            f1 = f1_score(y_test_clean, preds, average='weighted')
            
            results[name] = {
                'model': model,
                'accuracy': acc,
                'f1_score': f1
            }
            print(f"  {name} Accuracy: {acc:.3f}, F1: {f1:.3f}")
            
        return results



    def register_models(self, results):
        for name, data in results.items():
            with tempfile.TemporaryDirectory() as model_dir:
                joblib.dump(data['model'], Path(model_dir) / "model.joblib")
                
                registered = self.mr.python.create_model(
                    name=f"aqi_{name}",
                    metrics={"accuracy": float(data['accuracy']), "f1_score": float(data['f1_score'])},
                    description=f"{name} Classifier (Category 1-5)"
                )
                registered.save(model_dir)
                
                MODELS_DIR.mkdir(parents=True, exist_ok=True)
                joblib.dump(data['model'], MODELS_DIR / f"{name}.joblib")

    def run(self):
        self.connect_hopsworks()
        df = self.load_data()
        X = df.drop(columns=['datetime', 'timestamp', 'aqi'])
        y = df['aqi'].astype(int)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
        results = self.train_models(X_train, X_test, y_train, y_test)
        self.register_models(results)
        print("Training complete. New versions registered.")

if __name__ == "__main__":
    DailyTraining().run()