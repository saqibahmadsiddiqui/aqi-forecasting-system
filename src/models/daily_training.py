import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import hopsworks
import joblib
import tempfile
from pathlib import Path
import sys
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config.config import *

class DailyTraining:
    def __init__(self):
        self.project = None
        self.fs = None
        self.mr = None
        self.imputer = None
        
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
        print("\nLoading data from Feature Store...")
        fg = self.fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
        df = fg.read(online=True)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df_sorted = df.sort_values('datetime').reset_index(drop=True)
        
        print(f"  Total records: {len(df_sorted)}")
        print(f"  Date range: {df_sorted['datetime'].min()} to {df_sorted['datetime'].max()}")
        print(f"  Missing values: {df_sorted.isnull().sum().sum()} total")
        print(f"  AQI distribution:")
        
        # Print distribution
        for aqi_class in sorted(df_sorted['aqi'].unique()):
            count = (df_sorted['aqi'] == aqi_class).sum()
            pct = (count / len(df_sorted)) * 100
            bar = "█" * int(pct / 2)
            print(f"    Class {aqi_class}: {count:5d} samples ({pct:5.2f}%) {bar}")
        
        return df_sorted
    
    def prepare_data(self, df):
        """Prepare features and target with NaN handling"""
        X = df.drop(columns=['datetime', 'timestamp', 'aqi'])
        y = df['aqi'].astype(int)
        
        print(f"\nHandling missing values...")
        print(f"  Features with NaN:")
        nan_cols = X.columns[X.isnull().any()].tolist()
        
        if nan_cols:
            for col in nan_cols:
                nan_count = X[col].isnull().sum()
                pct = (nan_count / len(X)) * 100
                print(f"    {col}: {nan_count} missing ({pct:.2f}%)")
        else:
            print(f"    No missing values found!")
        
        # Impute missing values using median strategy
        self.imputer = SimpleImputer(strategy='median')
        X_imputed = self.imputer.fit_transform(X)
        X = pd.DataFrame(X_imputed, columns=X.columns)
        
        print(f"\nMissing values imputed")
        
        print(f"\nClass distribution before split:")
        for aqi_class in sorted(y.unique()):
            count = (y == aqi_class).sum()
            pct = (count / len(y)) * 100
            print(f"  Class {aqi_class}: {count:5d} samples ({pct:5.2f}%)")
        
        min_class_count = y.value_counts().min()
        
        if min_class_count < 2:
            print(f"\nWARNING: Smallest class has only {min_class_count} sample(s)")
            print(f"   Using random split (stratification not possible)")
            stratify_option = None
        else:
            print(f"\nAll classes have ≥2 samples - using stratified split")
            stratify_option = y
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=42, 
            shuffle=True,
            stratify=stratify_option
        )
        
        print(f"\nData split completed:")
        print(f" Training samples: {len(X_train)}")
        print(f" Testing samples:  {len(X_test)}")
        print(f" Train/Test ratio: {len(X_train)/len(X_test):.2f}:1")
        
        return X_train, X_test, y_train, y_test
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """Train 5 different classification models"""
        print("\n" + "="*60)
        print("TRAINING 5 CLASSIFICATION MODELS")
        print("="*60)
        
        # Prepare clean labels
        y_train_clean = y_train.values.flatten().astype(int)
        y_test_clean = y_test.values.flatten().astype(int)
        
        # Define 5 models with optimal hyperparameters
        models_config = {
            "random_forest": {
                "model": RandomForestClassifier(
                    n_estimators=200,
                    max_depth=20,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    class_weight='balanced',
                    random_state=42,
                    n_jobs=-1
                ),
                "description": "Random Forest with 200 trees"
            },
            "gradient_boosting": {
                "model": HistGradientBoostingClassifier(
                    max_iter=150,
                    max_depth=8,
                    learning_rate=0.1,
                    class_weight='balanced',
                    random_state=42,
                    loss='log_loss'
                ),
                "description": "Histogram Gradient Boosting Classifier"
            },
            "lightgbm": {
                "model": lgb.LGBMClassifier(
                    n_estimators=150,
                    max_depth=10,
                    learning_rate=0.05,
                    num_leaves=31,
                    class_weight='balanced',
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                ),
                "description": "LightGBM Gradient Boosting"
            },
            "decision_tree": {
                "model": DecisionTreeClassifier(
                    max_depth=15,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    class_weight='balanced',
                    random_state=42
                ),
                "description": "Decision Tree Classifier"
            },
            "sklearn_gradient_boosting": {
                "model": GradientBoostingClassifier(
                    n_estimators=150,
                    max_depth=5,
                    learning_rate=0.1,
                    subsample=0.8,
                    random_state=42,
                    verbose=0
                ),
                "description": "Scikit-Learn Gradient Boosting (with NaN handling)"
            }
        }
        
        results = {}
        
        for name, config in models_config.items():
            print(f"\nTraining: {name}")
            print(f" Description: {config['description']}")
            
            try:
                model = config['model']
                model.fit(X_train, y_train_clean)
                preds = model.predict(X_test)
                
                accuracy = accuracy_score(y_test_clean, preds)
                f1 = f1_score(y_test_clean, preds, average='weighted', zero_division=0)
                precision = precision_score(y_test_clean, preds, average='weighted', zero_division=0)
                recall = recall_score(y_test_clean, preds, average='weighted', zero_division=0)
                
                results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'precision': precision,
                    'recall': recall,
                    'description': config['description']
                }
                
                print(f"   Accuracy:  {accuracy:.4f}")
                print(f"   F1 Score:  {f1:.4f}")
                print(f"   Precision: {precision:.4f}")
                print(f"   Recall:    {recall:.4f}")
                
            except Exception as e:
                print(f"   Error training {name}: {str(e)}")
        
        return results
    
    def register_models(self, results):
        """Register all trained models in Hopsworks Model Registry"""
        print("\n" + "="*60)
        print("REGISTERING MODELS IN HOPSWORKS")
        print("="*60)
        
        registered_models = {}
        
        for name, data in results.items():
            try:
                print(f"\nRegistering: {name}")
                
                with tempfile.TemporaryDirectory() as model_dir:
                    model_path = Path(model_dir) / "model.joblib"
                    joblib.dump(data['model'], model_path)
                    
                    metrics = {
                        "accuracy": float(data['accuracy']),
                        "f1_score": float(data['f1_score']),
                        "precision": float(data['precision']),
                        "recall": float(data['recall'])
                    }
                    
                    registered = self.mr.python.create_model(
                        name=f"aqi_{name}",
                        metrics=metrics,
                        description=f"{data['description']} - Trained on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    )
                    registered.save(model_dir)
                    
                    MODELS_DIR.mkdir(parents=True, exist_ok=True)
                    joblib.dump(data['model'], MODELS_DIR / f"{name}.joblib")
                    
                    registered_models[name] = {
                        'version': registered.version,
                        'metrics': metrics
                    }
                    
                    print(f"  Registered: aqi_{name} (v{registered.version})")
                    print(f"   F1: {metrics['f1_score']:.4f} | Acc: {metrics['accuracy']:.4f}")
                    
            except Exception as e:
                print(f"   Error registering {name}: {str(e)}")
        
        return registered_models
    
    def run(self):
        """Execute the full training pipeline"""
        print("\n" + "="*80)
        print("DAILY MODEL TRAINING PIPELINE")
        print(f"Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print("="*80)
        
        try:
            self.connect_hopsworks()
            df = self.load_data()
            X_train, X_test, y_train, y_test = self.prepare_data(df)
            results = self.train_models(X_train, X_test, y_train, y_test)
            registered_models = self.register_models(results)
            
            print("\n" + "="*80)
            print("TRAINING PIPELINE COMPLETE")
            print("="*80)
            print(f"\nSummary:")
            print(f" Models trained: {len(results)}")
            print(f" Models registered: {len(registered_models)}")
            
            if results:
                best_model = max(results.items(), key=lambda x: x[1]['f1_score'])
                print(f"\nBest Performing Model:")
                print(f"   Name: {best_model[0]}")
                print(f"   F1 Score: {best_model[1]['f1_score']:.4f}")
                print(f"   Accuracy: {best_model[1]['accuracy']:.4f}")
                print(f"   Precision: {best_model[1]['precision']:.4f}")
                print(f"   Recall: {best_model[1]['recall']:.4f}")
            
            print("\n" + "="*80)
            print("Ready for predictions!")
            print("="*80)
            
        except Exception as e:
            print(f"\nERROR in training pipeline: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

if __name__ == "__main__":
    DailyTraining().run()
