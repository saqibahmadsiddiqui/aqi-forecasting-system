import hopsworks
import pandas as pd
import numpy as np
import joblib
import os
import json
from typing import Dict, Tuple
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

class HopsworksClient:
    def __init__(self):
        """Initialize Hopsworks connection"""
        self.api_key = os.getenv('HOPSWORKS_API_KEY')
        self.project_name = os.getenv('HOPSWORKS_PROJECT_NAME')
        self.project = None
        self.fs = None
        self.mr = None
        
        # Get project root directory
        self.root_dir = Path(__file__).parent.parent.parent
        self.models_dir = self.root_dir / 'models'
        
    def connect(self):
        """Connect to Hopsworks"""
        try:
            self.project = hopsworks.login(
                api_key_value=self.api_key,
                project=self.project_name
            )
            self.fs = self.project.get_feature_store()
            self.mr = self.project.get_model_registry()
            return True
        except Exception as e:
            print(f"Connection error: {e}")
            return False
    
    def get_latest_features(self, n_hours: int = 72) -> pd.DataFrame:
        """
        Get latest features from feature store
        Args:
            n_hours: Number of hours of historical data needed
        Returns:
            DataFrame with latest features
        """
        try:
            # Get feature view
            feature_view = self.fs.get_feature_view(
                name="aqi_prediction_fv",
                version=1
            )
            
            # Get batch data
            X, _ = feature_view.get_batch_data()
            
            # Get latest n_hours of data
            latest_data = X.tail(n_hours)
            
            return latest_data
            
        except Exception as e:
            print(f"Error fetching features: {e}")
            return None
    
    def load_production_model(self) -> Tuple[object, Dict]:
        """
        Load production model from registry
        Returns:
            (model, model_info)
        """
        try:
            # Load production config with absolute path
            config_path = self.models_dir / 'production_model_config.json'
            
            print(f"Looking for config at: {config_path}")
            
            if not config_path.exists():
                print(f"Config file not found at: {config_path}")
                return None, None
            
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            model_key = config['model_key']
            version = config['version']
            
            print(f"Loading model: {model_key} (version {version})")
            
            # Get model from registry
            model = self.mr.get_model(name=model_key, version=version)
            model_dir = model.download()
            
            # Load the actual model
            model_file = os.path.join(model_dir, f"{model_key}_model.pkl")
            
            if not os.path.exists(model_file):
                print(f"Model file not found: {model_file}")
                return None, None
            
            loaded_model = joblib.load(model_file)
            
            return loaded_model, config
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return None, None
    
    def get_all_models_info(self) -> Dict:
        """Get information about all registered models"""
        try:
            # Use absolute path
            ui_json_path = self.models_dir / 'ui_model_comparison.json'
            
            if not ui_json_path.exists():
                print(f"UI comparison file not found at: {ui_json_path}")
                return None
            
            with open(ui_json_path, 'r') as f:
                models_data = json.load(f)
            
            return models_data
            
        except Exception as e:
            print(f"Error loading models info: {e}")
            return None


# Global client instance
hops_client = HopsworksClient()