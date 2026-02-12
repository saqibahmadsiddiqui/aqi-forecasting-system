# 🌫️ AQI Forecasting System - Multan, Pakistan

**100% Serverless 3-Day Air Quality Prediction System**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://multan-aqi.streamlit.app)
[![GitHub](https://img.shields.io/badge/GitHub-saqibahmadsiddiqui-blue?logo=github)](https://github.com/saqibahmadsiddiqui)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/)

## 🚀 Live Demo

- **📊 Dashboard**: [https://multan-aqi.streamlit.app](https://multan-aqi.streamlit.app)
- **📍 Location**: Multan, Punjab, Pakistan (30.1979°N, 71.4724°E)
- **⚡ Update Frequency**: Hourly data collection, Daily model retraining at 12 AM UTC (5 AM PKT)
- **🎯 Forecast Horizon**: 3 days ahead with daily averages

## 🎯 Project Overview

End-to-end machine learning system that predicts Air Quality Index (AQI) for the next 3 days in Multan, Pakistan. Built with a completely serverless architecture using Hopsworks Feature Store, GitHub Actions for CI/CD, and deployed on Streamlit Cloud.

The system continuously:
- 📥 Collects real-time air quality data hourly from OpenWeather API
- 🔧 Engineers 40+ advanced features (time, lag, rolling statistics, interactions)
- 🤖 Trains 3 ML models daily (Random Forest, XGBoost, LightGBM)
- 📈 Generates accurate 3-day AQI forecasts
- 📊 Provides interactive visualizations and alerts
- 🌐 Serves predictions via web dashboard

## ✨ Key Features

✅ **Hourly Data Pipeline**: Automated data collection from OpenWeather API every hour  
✅ **Smart Duplicate Detection**: Compares with last 6 hours to avoid redundant storage  
✅ **40+ Engineered Features**: Time-based, lag features, rolling statistics, interactions  
✅ **3 ML Models**: Random Forest, XGBoost, LightGBM with auto best model selection  
✅ **Daily Retraining**: Models retrain daily with cumulative historical data  
✅ **3-Day Forecasts**: Hourly predictions aggregated to daily averages with ranges  
✅ **Beautiful Dashboard**: Dark theme, animated sidebar, real-time visualizations  
✅ **REST API**: FastAPI endpoints for programmatic access  
✅ **100% Serverless**: No server management, fully automated CI/CD pipelines  
✅ **Production-Ready**: Error handling, monitoring, logging, and alerts  

---

## 📊 Model Performance and Selection Example

| Model | RMSE | MAE | R² Score | Status |
|-------|------|-----|----------|--------|
| **Random Forest** | 0.026 | 0.004 | 0.999 | ⭐ SELECTED |
| XGBoost | 0.035 | 0.002 | 0.997 | Good |
| LightGBM | 0.049 | 0.012 | 0.995 | Good |

*Performance metrics updated daily. Random Forest selected for best RMSE performance.*

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Cloud Dashboard                │
│  https://multan-aqi.streamlit.app                           │
│  ├─ 3-Day Forecast Cards    (AQI values & alerts)           │
│  ├─ Model Comparison        (R², MAE, RMSE metrics)         │
│  └─ Project Information     (Architecture & models)         │
└──────────────────────┬──────────────────────────────────────┘
                       │ (Reads predictions)
                       ↓
┌──────────────────────────────────────────────────────────────┐
│              Hopsworks Feature Store                         │
│  ├─ Raw Features (19): CO, NO2, O3, PM2.5, PM10, Temp, etc   │
│  ├─ Engineered (40+): Time, lags, rolling stats, interact    │
│  ├─ Model Registry: Trained models stored                    │
│  └─ Predictions: 3-day forecast results                      │
└──────────────┬──────────────────────────────────────────────-┘
               │ (Store & fetch)
        ┌──────┴──────┐
        │             │
        ↓             ↓
┌──────────────┐  ┌──────────────────────┐
│ Hourly       │  │ Daily Training       │
│ Pipeline     │  │ Pipeline             │
│              │  │                      │
│ (every hour) │  │ (2 AM UTC/7 AM PKT)  │
│              │  │                      │
│ • Collect    │  │ • Train 3 models     │
│   data       │  │ • Compare & select   │
│ • Check      │  │   best (RMSE)        │
│   duplicates │  │ • Register models    │
│ • Engineer   │  │ • Generate forecasts │
│   features   │  │ • Push to Hub        │
│ • Upload     │  │                      │
│              │  │                      │
└──────────────┘  └──────────────────────┘
        ↑                    ↑
        └────────────────────┘
            GitHub Actions
           (Fully Automated)
            ↑
            │ (OpenWeather API)
            │
    ┌───────────────┐
    │ OpenWeather   │
    │ API           │
    │ (Real-time)   │
    └───────────────┘
```

---

## 🛠️ Tech Stack

### **Machine Learning**
- **scikit-learn**: Random Forest, preprocessing
- **XGBoost**: Gradient boosting model
- **LightGBM**: Fast gradient boosting
- **pandas** & **numpy**: Data processing
- **joblib**: Model serialization

### **Feature Store & MLOps**
- **Hopsworks**: Feature store, model registry, data versioning
- **GitHub Actions**: CI/CD automation (hourly & daily)

### **Web & API**
- **Streamlit**: Interactive dashboard (dark theme, animations)
- **FastAPI**: REST API for predictions
- **Uvicorn**: ASGI server

### **Data Source**
- **OpenWeather API**: Real-time weather & air quality data
- **Python-dotenv**: Secure environment variable management

### **Infrastructure**
- **Streamlit Cloud**: Dashboard deployment
- **GitHub**: Code repository & CI/CD
- **Hopsworks Cloud**: Feature store & model registry

---

## 📦 Installation & Setup

### Prerequisites
- Python 3.10 or higher
- GitHub account (for CI/CD)
- Hopsworks account (free tier available)
- OpenWeather API key (free tier available)

### Local Development

```bash
# Clone repository
git clone https://github.com/saqibahmadsiddiqui/aqi-forecasting-system.git
cd aqi-forecasting-system

# Create virtual environment
python -3.11 -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env
# Edit .env with your API keys:
# - HOPSWORKS_API_KEY
# - OPENWEATHER_API_KEY
# - HOPSWORKS_PROJECT_NAME
```

### Environment Variables

Create `.env` file in project root:

```
# Hopsworks Configuration
HOPSWORKS_API_KEY=your_hopsworks_api_key
HOPSWORKS_PROJECT_NAME=aqi_forecasting

# OpenWeather API
OPENWEATHER_API_KEY=your_openweather_api_key

# Location (Multan, Pakistan)
LOCATION_LAT=30.1979793
LOCATION_LON=71.4724978
LOCATION_NAME=Multan

# Timezone
TIMEZONE=Asia/Karachi
```

---

## 🎯 Usage Guide

### 1. Initial Data Load (Run Once)

```bash
python src/setup/initial_data_load.py
```

This will:
- Extract historical data from October 2025
- Engineer 40+ features
- Upload to Hopsworks Feature Store
- Create initial feature view

### 2. Train Models

```bash
python src/models/daily_training.py
```

Trains all 3 models and selects the best one based on RMSE.

### 3. Generate Predictions

```bash
python src/prediction/predictor.py
```

Generates 3-day ahead forecasts using the best model.

### 4. Run Dashboard

```bash
streamlit run dashboard/app.py
```

Visit: http://localhost:8501

**Dashboard Pages:**
- **3-Day Forecast**: AQI predictions with categories and alerts
- **Model Comparison**: Performance metrics for all models
- **About**: Project information and feature details

### 5. Run API (Optional)

```bash
python api/main.py
```

Starts FastAPI server at http://localhost:8000
- API Docs: http://localhost:8000/docs
- Predictions: GET `/predict`

---

## 🔄 Automated Pipelines

### Hourly Pipeline (GitHub Actions)

Runs **every hour at :00** UTC:

```yaml
Schedule: 0 * * * *

Tasks:
  1. Fetch current air quality data from OpenWeather
  2. Check for duplicates (last 6 hours)
  3. Engineer features (42+ calculations)
  4. Upload to Hopsworks Feature Store
  5. Commit results to GitHub
```

**Purpose**: Keep feature store updated with latest data for daily training.

### Daily Pipeline (GitHub Actions)

Runs **daily at 12 AM UTC** (5 AM PKT):

```yaml
Schedule: 0 0 * * *

Tasks:
  1. Load all data from Feature Store
  2. Train 3 models (RF, XGBoost, LightGBM)
  3. Evaluate performance (RMSE, MAE, R²)
  4. Select best model (lowest RMSE)
  5. Register in Hopsworks Model Registry
  6. Generate 3-day predictions
  7. Save predictions to CSV
  8. Commit to GitHub
  9. Push to Feature Store
```

**Purpose**: Retrain models with new data and generate forecasts.

---

## 📈 Features Engineering

### Raw Features (19)
- **Pollutants**: CO, NO, NO2, O3, SO2, PM2.5, PM10, NH3 (8 features)
- **Weather**: Temperature, Humidity, Pressure, Wind Speed, Cloud Cover (5 features)
- **Target**: AQI (1-5 scale) (1 feature)
- **Temporal**: Date, Time (2 features)
- **Metadata**: Latitude, Longitude, Elevation (3 features)

### Engineered Features (40+)

**Time-based Features** (8):
- Hour of day (cyclical: sin/cos)
- Day of week (cyclical: sin/cos)
- Month (cyclical: sin/cos)
- Is weekend

**Lag Features** (24):
- AQI lags: t-1, t-3, t-6, t-12, t-24, t-48, t-72 hours
- PM2.5 lags: t-1, t-3, t-6, t-12, t-24, t-48, t-72
- Temperature lag: t-1
- Humidity lag: t-1

**Rolling Statistics** (12):
- AQI mean/std over 3h, 6h, 12h, 24h windows
- PM2.5 mean/std over 3h, 6h, 12h, 24h windows

**Rate of Change** (4):
- AQI change over 1h, 3h, 24h
- PM2.5 change over 1h

**Min/Max** (2):
- 24-hour rolling min/max

**Interactions** (3):
- PM2.5 × Humidity
- PM2.5 × Temperature
- PM2.5 × Wind Speed

**Total**: 42+ features for training

---

## 🎨 Dashboard Features

### 3-Day Forecast Cards
- Daily average AQI value
- Category: Good, Fair, Moderate, Poor, Very Poor
- Color-coded backgrounds
- Min-Max range display
- Status alerts (SAFE/ALERT)

### Trend Visualization
- Interactive line chart
- AQI trend over 3 days
- Min/Max confidence bands
- Hover details

### Model Comparison
- Performance table for all 3 models
- R² Score, MAE, RMSE metrics
- Best model highlighted
- Bar chart comparison

### Navigation
- Animated sidebar with smooth transitions
- Dark theme with cyan accents
- Responsive design (mobile-friendly)
- Real-time updates (2-minute cache)

### Footer
- Created by Saqib Ahmad Siddiqui
- Last updated timestamp
- System status indicator

---

## 📡 API Endpoints

### Health & Status
```
GET /health
Response: {"status": "ok", "timestamp": "2025-02-11T10:30:00Z"}
```

### Predictions
```
GET /predict
Response: {
  "date": "2025-02-11",
  "aqi": 2.5,
  "category": "Fair",
  "min_aqi": 2.1,
  "max_aqi": 2.9
}
```

### Model Comparison
```
GET /models
Response: [
  {"model": "Random Forest", "r2": 0.999, "mae": 0.004, "rmse": 0.026},
  {"model": "XGBoost", "r2": 0.997, "mae": 0.002, "rmse": 0.035},
  {"model": "LightGBM", "r2": 0.995, "mae": 0.012, "rmse": 0.049}
]
```

### API Documentation
```
GET /docs
Interactive API documentation (Swagger UI)
```

---

## 📊 Project Structure

```
aqi-forecasting-system/
├── .github/
│   └── workflows/                    # CI/CD pipelines
│       ├── hourly_data_pipeline.yml  # Hourly data collection
│       └── daily_training_pipeline.yml # Daily model training
│
├── api/                              # FastAPI application
│   └── main.py                       # API server
│
├── dashboard/                        # Streamlit application
│   └── app.py                        # Main dashboard
│
├── data/                             # Data storage
│   ├── raw/                          # Raw data from APIs
│   ├── interim/                      # Intermediate processing
│   └── processed/                    # Final predictions & CSVs
│
├── models/                           # Saved ML models
│   ├── random_forest_model.joblib
│   ├── xgboost_model.joblib
│   └── lightgbm_model.joblib
│
├── notebooks/                        # Jupyter notebooks (EDA)
│   └── exploratory_data_analysis.ipynb
│
├── src/                              # Source code
│   ├── config/
│   │   └── config.py                 # Configuration & constants
│   ├── ingestion/
│   │   └── hourly_data_pipeline.py         # OpenWeather API integration
│   ├── models/
│   │   └── daily_training.py               # Model training
│   ├── prediction/
│   │   └── predictor.py              # Prediction pipeline
│   ├── features/
│   └── setup/
│       |── initial_data_load.py      # Initial data load
|       |── debug_upload.py      # Check data features
|       └── verify_upload.py      # Check data
│
├── .env                              # Environment variables (local)
├── .gitignore                        # Git ignore rules
├── requirements.txt                  # Python dependencies (15+)
└── README.md                         # This file
```

---

## 🚀 Deployment Guide

### Deploy to Streamlit Cloud

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Deploy to Streamlit Cloud"
   git push origin main
   ```

2. **Create Streamlit Cloud App**
   - Go to https://share.streamlit.io
   - Click "New app"
   - Select repository: `aqi-forecasting-system`
   - Select branch: `main`
   - Set file: `dashboard/app.py`

3. **Add Secrets**
   - In Streamlit Cloud: Settings → Secrets
   - Add:
     ```toml
     hopsworks_api_key = "..."
     hopsworks_project_name = "..."
     openweather_api_key = "..."
     ```

4. **Wait for Deployment**
   - First build: 8-12 minutes
   - Subsequent: 3-5 minutes

Your dashboard will be live at: https://multan-aqi.streamlit.app

---

## 🔄 CI/CD Pipeline Configuration

### GitHub Secrets Required

1. **HOPSWORKS_API_KEY**
   - Get from: Hopsworks → Settings → API Keys

2. **OPENWEATHER_API_KEY**
   - Get from: OpenWeather → API Keys

### Workflow Files

Both workflows automatically configured to:
- Use secrets from GitHub
- Run on schedule (hourly & daily)
- Commit results to repository
- Handle errors gracefully

No additional setup needed!

---

## 📈 Performance Metrics

### Model Training Time
- First training: ~2-3 minutes
- Daily retraining: ~3-5 minutes
- GitHub Actions execution: 2-5 minutes total

### Prediction Latency
- Cold start: ~2 seconds
- Subsequent: <500ms
- API response: <1 second

### Data Pipeline
- Hourly collection: ~30 seconds
- Feature engineering: ~20 seconds
- Upload to Feature Store: ~10 seconds
- Total: <2 minutes per hour

### Dashboard Performance
- Page load: 2-5 seconds
- Chart rendering: <1 second
- Data refresh: Every 2 minutes (cache)

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Areas for contribution:
- Additional ML models (LSTM, Logistic Regression etc)
- More engineered features
- Improved API endpoints
- Dashboard enhancements
- Documentation improvements

---

## 📝 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 👨‍💻 Author

**Saqib Ahmad Siddiqui**
- GitHub: [@saqibahmadsiddiqui](https://github.com/saqibahmadsiddiqui)
- Email: saqibahmad2004@gmail.com
- LinkedIn: [Profile](https://linkedin.com/in/saqib-ahmad-siddiqui)

---

## 🙏 Acknowledgments

- **OpenWeather API**: Real-time weather and air quality data
- **Hopsworks**: Feature store and ML ops platform
- **Streamlit**: Interactive dashboard framework
- **GitHub Actions**: CI/CD automation
- **scikit-learn, XGBoost, LightGBM**: ML libraries

---

## 📚 Documentation

For detailed documentation, see:

- [Setup Guide](docs/Report)

---

## 🎯 Roadmap

### Current (v1.0)
- ✅ 3-day AQI forecasting
- ✅ 3 ML models with auto-selection
- ✅ Streamlit dashboard
- ✅ REST API
- ✅ Hourly & daily pipelines

---

## 📞 Support

For issues and questions:
- **GitHub Issues**: [Report bugs](https://github.com/saqibahmadsiddiqui/aqi-forecasting-system/issues)
- **Email**: saqibahmad2004@gmail.com
- **Dashboard**: https://multan-aqi.streamlit.app

---

**Last Updated**: February 2025  
**Version**: 1.0  
**Status**: Active & Maintained
