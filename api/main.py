from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.config.config import PROCESSED_DATA_DIR

app = FastAPI(
    title="AQI Prediction API",
    description="3-day AQI Category predictions for Multan, Pakistan",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionResponse(BaseModel):
    date: str
    day_name: str
    average_aqi: int
    min_aqi: int
    max_aqi: int
    category: str
    warning: Optional[str] = None
    timestamp: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "date": "2026-02-17",
                "day_name": "Tuesday",
                "average_aqi": 4,
                "min_aqi": 3,
                "max_aqi": 5,
                "category": "🔴 Poor",
                "warning": "⚠️ Poor! Limit outdoor exposure.",
                "timestamp": "2026-02-16 12:15:30"
            }
        }

class ModelMetrics(BaseModel):
    model: str
    accuracy: float
    f1_score: float
    precision: float
    recall: float
    version: int
    is_best: bool
    
    class Config:
        json_schema_extra = {
            "example": {
                "model": "LightGBM",
                "accuracy": 0.9985,
                "f1_score": 0.9985,
                "precision": 0.9985,
                "recall": 0.9985,
                "version": 21,
                "is_best": True
            }
        }

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2026-02-16T12:15:30"
            }
        }

# ==================== ENDPOINTS ====================

@app.get("/", response_model=HealthResponse)
def root():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health", response_model=HealthResponse)
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/predict", response_model=List[PredictionResponse])
def get_predictions():
    try:
        predictions_path = PROCESSED_DATA_DIR / 'latest_predictions.csv'
        
        if predictions_path.exists():
            df = pd.read_csv(predictions_path)
            
            df['average_aqi'] = df['average_aqi'].astype(int)
            df['min_aqi'] = df['min_aqi'].astype(int)
            df['max_aqi'] = df['max_aqi'].astype(int)
            
            df['warning'] = df['warning'].where(pd.notna(df['warning']), None)
            df['timestamp'] = df['timestamp'].where(pd.notna(df['timestamp']), None)
            
            records = df.to_dict('records')
            
            return [PredictionResponse(**record) for record in records]
        else:
            raise HTTPException(
                status_code=404,
                detail="Predictions not found. Please run the training pipeline first."
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving predictions: {str(e)}"
        )

@app.get("/models", response_model=List[ModelMetrics])
def get_models():
    try:
        comparison_path = PROCESSED_DATA_DIR / 'model_comparison.csv'
        
        if comparison_path.exists():
            df = pd.read_csv(comparison_path)
            
            for col in ['accuracy', 'f1_score', 'precision', 'recall']:
                if col in df.columns:
                    df[col] = df[col].astype(float)
            
            if 'f1_score' in df.columns:
                max_f1 = df['f1_score'].max()
                df['is_best'] = df['f1_score'] == max_f1
            else:
                df['is_best'] = False
            
            df['version'] = df['version'].astype(int)
            
            df = df.sort_values('f1_score', ascending=False)
            
            records = df.to_dict('records')
            
            return [ModelMetrics(**record) for record in records]
        else:
            raise HTTPException(
                status_code=404,
                detail="Model comparison not found. Please run the training pipeline first."
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving model metrics: {str(e)}"
        )

@app.get("/models/best")
def get_best_model():
    try:
        comparison_path = PROCESSED_DATA_DIR / 'model_comparison.csv'
        
        if comparison_path.exists():
            df = pd.read_csv(comparison_path)
            
            if 'f1_score' in df.columns:
                best_idx = df['f1_score'].idxmax()
                best_model = df.iloc[best_idx]
                
                return {
                    "model": best_model['model'],
                    "accuracy": float(best_model['accuracy']),
                    "f1_score": float(best_model['f1_score']),
                    "precision": float(best_model['precision']),
                    "recall": float(best_model['recall']),
                    "version": int(best_model['version'])
                }
            else:
                raise HTTPException(
                    status_code=400,
                    detail="F1 score not found in model metrics"
                )
        else:
            raise HTTPException(
                status_code=404,
                detail="Model comparison not found"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving best model: {str(e)}"
        )

@app.get("/status")
def get_status():
    predictions_path = PROCESSED_DATA_DIR / 'latest_predictions.csv'
    comparison_path = PROCESSED_DATA_DIR / 'model_comparison.csv'
    
    return {
        "predictions_available": predictions_path.exists(),
        "models_available": comparison_path.exists(),
        "location": "Multan, Pakistan",
        "forecast_horizon": "72 hours (3 days)",
        "model_type": "Classification (1-5 categories)",
        "data_source": "OpenWeather API",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/info")
def get_info():
    return {
        "api_version": "1.0.0",
        "title": "AQI Prediction API",
        "description": "3-day AQI Category predictions for Multan, Pakistan",
        "models_count": 5,
        "model_names": [
            "Random Forest",
            "Histogram Gradient Boosting",
            "LightGBM",
            "Decision Tree",
            "Sklearn Gradient Boosting"
        ],
        "aqi_categories": {
            1: "Good",
            2: "Fair",
            3: "Moderate",
            4: "Poor",
            5: "Very Poor"
        },
        "endpoints": {
            "health": "GET /health",
            "predictions": "GET /predict",
            "models": "GET /models",
            "best_model": "GET /models/best",
            "status": "GET /status",
            "docs": "GET /docs"
        }
    }

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return {
        "error": exc.detail,
        "status_code": exc.status_code,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
