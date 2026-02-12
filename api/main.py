# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import List
# import pandas as pd
# from datetime import datetime
# import sys
# from pathlib import Path
# import os
# from dotenv import load_dotenv

# load_dotenv()

# sys.path.append(str(Path(__file__).parent.parent))
# from src.prediction.predictor import AQIPredictor
# from src.config.config import PROCESSED_DATA_DIR, TIMEZONE

# app = FastAPI(
#     title="AQI Prediction API",
#     description="3-day AQI predictions for Multan, Pakistan",
#     version="1.0.0"
# )

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# class PredictionResponse(BaseModel):
#     date: str
#     day_name: str
#     average_aqi: float
#     min_aqi: float
#     max_aqi: float
#     category: str
#     warning: str = None

# class ModelMetrics(BaseModel):
#     model: str
#     mae: float
#     rmse: float
#     r2_score: float
#     is_best: bool

# @app.get("/")
# def root():
#     return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# @app.get("/predict", response_model=List[PredictionResponse])
# def get_predictions():
#     try:
#         predictions_path = PROCESSED_DATA_DIR / 'latest_predictions.csv'
        
#         if predictions_path.exists():
#             df = pd.read_csv(predictions_path).fillna("")
#             return df.to_dict('records')
#         else:
#             predictor = AQIPredictor()
#             predictor.connect_hopsworks()
#             predictor.load_all_models()
#             predictions = predictor.predict_next_3_days()
            
#             pd.DataFrame(predictions).to_csv(predictions_path, index=False)
#             return predictions
            
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/models", response_model=List[ModelMetrics])
# def get_models():
#     try:
#         comparison_path = PROCESSED_DATA_DIR / 'model_comparison.csv'
        
#         if comparison_path.exists():
#             df = pd.read_csv(comparison_path)
#             max_r2 = df['r2_score'].max()
#             df['is_best'] = df['r2_score'] == max_r2
#             return df.to_dict('records')
#         else:
#             predictor = AQIPredictor()
#             predictor.connect_hopsworks()
#             predictor.load_all_models()
#             comparison = predictor.get_model_comparison()
            
#             pd.DataFrame(comparison).to_csv(comparison_path, index=False)
#             return comparison
            
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

sys.path.append(str(Path(__file__).parent.parent))
from src.prediction.predictor import AQIPredictor
from src.config.config import PROCESSED_DATA_DIR, TIMEZONE

app = FastAPI(
    title="AQI Prediction API",
    description="3-day AQI Category predictions for Multan, Pakistan",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Updated Pydantic Models ---
class PredictionResponse(BaseModel):
    date: str
    day_name: str
    average_aqi: int  # Changed from float to int
    min_aqi: int      # Changed from float to int
    max_aqi: int      # Changed from float to int
    category: str
    warning: Optional[str] = None

class ModelMetrics(BaseModel):
    model: str
    accuracy: float   # New
    f1_score: float   # New
    is_best: bool
    # mae, rmse, r2_score removed as they don't apply to classifiers

@app.get("/")
def root():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/predict", response_model=List[PredictionResponse])
def get_predictions():
    try:
        predictions_path = PROCESSED_DATA_DIR / 'latest_predictions.csv'
        
        if predictions_path.exists():
            df = pd.read_csv(predictions_path).fillna("")
            # Ensure AQI values are integers to match schema
            df['average_aqi'] = df['average_aqi'].astype(int)
            return df.to_dict('records')
        else:
            predictor = AQIPredictor()
            predictor.connect_hopsworks()
            predictor.load_latest_models() # Updated method name
            predictions = predictor.predict_next_3_days()
            
            pd.DataFrame(predictions).to_csv(predictions_path, index=False)
            return predictions
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models", response_model=List[ModelMetrics])
def get_models():
    try:
        comparison_path = PROCESSED_DATA_DIR / 'model_comparison.csv'
        
        if comparison_path.exists():
            df = pd.read_csv(comparison_path)
            # Find best model based on F1 Score
            max_f1 = df['f1_score'].max()
            df['is_best'] = df['f1_score'] == max_f1
            return df.to_dict('records')
        else:
            predictor = AQIPredictor()
            predictor.connect_hopsworks()
            predictor.load_latest_models() # Updated method name
            comparison = predictor.get_model_comparison()
            
            pd.DataFrame(comparison).to_csv(comparison_path, index=False)
            return comparison
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
