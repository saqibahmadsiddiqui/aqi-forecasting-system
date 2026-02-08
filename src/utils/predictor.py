import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict

class AQIPredictor:
    def __init__(self, model, feature_columns: List[str]):
        """
        Initialize predictor
        Args:
            model: Trained model
            feature_columns: List of feature names
        """
        self.model = model
        self.feature_columns = feature_columns
    
    def create_prediction_features(self, latest_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create features for next 3 days prediction
        Uses latest data to engineer features for future predictions
        """
        # Get the most recent row
        latest_row = latest_data.iloc[-1].copy()
        
        # Create features for 3 predictions (next 3 days, 24h apart)
        predictions_data = []
        
        for day in range(1, 4):  # Day 1, Day 2, Day 3
            pred_features = latest_row.copy()
            
            # Adjust time features for future day
            # (In production, you'd fetch actual forecasted weather)
            # For now, we use latest available features
            
            predictions_data.append(pred_features)
        
        pred_df = pd.DataFrame(predictions_data)
        
        # Ensure correct column order
        pred_df = pred_df[self.feature_columns]
        
        return pred_df
    
    def predict_next_3_days(self, latest_data: pd.DataFrame) -> List[Dict]:
        """
        Predict AQI for next 3 days
        Returns:
            List of predictions with metadata
        """
        # Create prediction features
        pred_features = self.create_prediction_features(latest_data)
        
        # Make predictions
        predictions = self.model.predict(pred_features)
        
        # Format results
        results = []
        base_date = datetime.now()
        
        for day, aqi_value in enumerate(predictions, 1):
            pred_date = base_date + timedelta(days=day)
            
            # Determine AQI category
            category, color = self.get_aqi_category(aqi_value)
            
            results.append({
                'day': day,
                'date': pred_date.strftime('%Y-%m-%d'),
                'aqi': round(float(aqi_value), 2),
                'category': category,
                'color': color,
                'health_message': self.get_health_message(category)
            })
        
        return results
    
    @staticmethod
    def get_aqi_category(aqi: float) -> tuple:
        """
        Get AQI category and color
        Based on standard AQI index:
        1 = Good, 2 = Fair, 3 = Moderate, 4 = Poor, 5 = Very Poor
        """
        if aqi <= 1.5:
            return "Good", "#00e400"
        elif aqi <= 2.5:
            return "Fair", "#ffff00"
        elif aqi <= 3.5:
            return "Moderate", "#ff7e00"
        elif aqi <= 4.5:
            return "Poor", "#ff0000"
        else:
            return "Very Poor", "#8f3f97"
    
    @staticmethod
    def get_health_message(category: str) -> str:
        """Get health advisory message"""
        messages = {
            "Good": "Air quality is satisfactory. Enjoy outdoor activities!",
            "Fair": "Air quality is acceptable. Unusually sensitive people should consider reducing prolonged outdoor exertion.",
            "Moderate": "Members of sensitive groups may experience health effects. General public less likely to be affected.",
            "Poor": "Everyone may begin to experience health effects. Members of sensitive groups may experience more serious effects.",
            "Very Poor": "Health alert! Everyone may experience serious health effects. Avoid outdoor activities."
        }
        return messages.get(category, "No data available")