# 🌫️ AQI Forecasting System - Multan, Pakistan

**100% Serverless 3-Day Air Quality Prediction System**


## 🎯 Project Overview

End-to-end machine learning system that predicts Air Quality Index (AQI) for the next 3 days in Multan, Pakistan. Built with a completely serverless architecture using Hopsworks Feature Store, GitHub Actions for CI/CD, and deployed on Streamlit Cloud.

## ✨ Features

- ✅ **Hourly Data Collection**: Automated data fetching from OpenWeather API
- ✅ **Smart Duplicate Detection**: Compares with last 6 hours to avoid redundant storage
- ✅ **42 Engineered Features**: Time-based, lag, rolling statistics, and interaction features
- ✅ **3 ML Models**: Random Forest, XGBoost, LightGBM with automatic best model selection
- ✅ **Daily Retraining**: Models retrain daily with all historical + new data
- ✅ **3-Day Predictions**: Hourly forecasts aggregated to daily averages
- ✅ **Interactive Dashboard**: Real-time visualizations with Streamlit
- ✅ **REST API**: FastAPI endpoints for programmatic access
- ✅ **100% Serverless**: No servers to manage, fully automated pipelines

## 🏗️ Architecture
```
┌─────────────────┐
│  OpenWeather API│
└────────┬────────┘
         │ Hourly (GitHub Actions)
         ▼
┌─────────────────┐
│ Feature Pipeline│ → Duplicate Check (6h)
└────────┬────────┘   → Feature Engineering (42 features)
         │
         ▼
┌─────────────────┐
│ Hopsworks        │
│ Feature Store   │
└────────┬────────┘
         │ Daily (GitHub Actions)
         ▼
┌─────────────────┐
│Training Pipeline│ → Train 3 models
└────────┬────────┘   → Select best (RMSE)
         │            → Register in Model Registry
         ▼
┌─────────────────┐
│   Predictions   │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐ ┌────────┐
│Streamlit│ │FastAPI │
│Dashboard│ │  API   │
└─────────┘ └────────┘
```

## 📊 Model Performance

| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| **Random Forest** ⭐ | 0.026 | 0.004 | 0.999 |
| XGBoost | 0.035 | 0.002 | 0.997 |
| LightGBM | 0.049 | 0.012 | 0.995 |

## 🚀 Live Demo

- **Dashboard**: [Your Streamlit URL]
- **API**: [Your API URL]
- **API Docs**: [Your API URL]/docs

## 🛠️ Tech Stack

- **ML**: Scikit-learn, XGBoost, LightGBM
- **Feature Store**: Hopsworks
- **Dashboard**: Streamlit
- **API**: FastAPI
- **CI/CD**: GitHub Actions
- **Data Source**: OpenWeather API

## 📦 Installation

### Prerequisites

- Python 3.10+
- OpenWeather API Key
- Hopsworks Account

### Local Setup
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
cp .env.example .env
# Edit .env with your API keys
```

### Environment Variables

Create `.env` file:
```env
HOPSWORKS_API_KEY=your_key
HOPSWORKS_PROJECT_NAME=your_project
OPENWEATHER_API_KEY=your_key
```

## 🎯 Usage

### 1. Initial Data Load (Run Once)
```bash
python src/setup/initial_data_load.py
```

This will:
- Extract data from latest_data_date to current date
- Engineer 42 features
- Upload to Hopsworks Feature Store

### 2. Train Models
```bash
python src/models/daily_training.py
```

### 3. Generate Predictions
```bash
python src/prediction/predictor.py
```

### 4. Run Dashboard
```bash
streamlit run dashboard/app.py
```

Visit: http://localhost:8501

### 5. Run API
```bash
python api/main.py
```

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | GET | Get 3-day predictions |
| `/predictions/{date}` | GET | Get prediction for specific date |
| `/models` | GET | Get all models comparison |
| `/models/best` | GET | Get best model info |
| `/health` | GET | Health check |

## 🤖 Automated Pipelines

### Hourly Pipeline (GitHub Actions)

Runs every hour:
1. Fetches current air quality data
2. Checks for duplicates (last 6 hours)
3. Engineers features
4. Uploads to Feature Store

### Daily Pipeline (GitHub Actions)

Runs at 2 AM UTC:
1. Loads all data from Feature Store
2. Trains 3 models
3. Evaluates performance
4. Selects best model (R² Score)
5. Registers in Model Registry
6. Generates 3-day predictions

## 📈 Features

### Raw Features (19)
- Pollutants: CO, NO, NO2, O3, SO2, PM2.5, PM10, NH3
- Weather: Temperature, Humidity, Pressure, Wind Speed, etc.
- Target: AQI (1-5)

### Engineered Features (40+ features)
- **Time Features**: Hour, day of week, month (cyclical encoding)
- **Lag Features**: AQI, PM2.5, PM10 at t-1, t-3, t-6, t-12, t-24, t-48, t-72
- **Rolling Statistics**: Mean and std over 3h, 6h, 12h, 24h windows
- **Rate of Change**: AQI change over 1h, 3h, 24h
- **Min/Max**: 24-hour rolling min/max
- **Interactions**: PM2.5 × Humidity, PM2.5 × Temperature, PM2.5 × Wind Speed

## 🎨 Dashboard Features

- **3-Day Forecast Cards**: Daily average AQI with category colors
- **Trend Visualization**: Interactive line charts with min/max ranges
- **Model Comparison**: Performance metrics for all models
- **Best Model Indicator**: Highlights currently selected model
- **AQI Categories**: Color-coded reference guide

## 📝 Project Structure
```
aqi-forecasting-system/
├── .github/workflows/         # CI/CD pipelines
├── api/                      # FastAPI application
├── dashboard/                # Streamlit dashboard
├── data/                     # Data storage
│   ├── raw/                  # Raw data
│   ├── interim/              # Intermediate data
│   └── processed/            # Final predictions
├── models/                   # Saved models
├── notebooks/                # Jupyter notebooks (EDA)
├── src/                      # Source code
│   ├── config/               # Configuration
│   ├── ingestion/            # Data collection
│   ├── models/               # Training pipeline
│   ├── prediction/           # Prediction pipeline
│   └── setup/                # Initial setup
├── .env                      # Environment variables
├── .gitignore               # Git ignore rules
├── README.md                # This file
└── requirements.txt         # Dependencies
```

## 🔄 CI/CD Pipeline

GitHub Actions automatically:
- **Hourly**: Collects new data and updates Feature Store
- **Daily**: Retrains models and generates predictions
- **On Push**: Runs tests and validations

## 📊 Monitoring

- Feature Store: Hopsworks UI
- Model Performance: Dashboard comparison page
- API Health: `/health` endpoint
- Logs: GitHub Actions logs

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 👨‍💻 Author

- Name: Saqib Ahmad Siddiqui
- GitHub: [saqibahmadsiddiqui](https://github.com/saqibahmadsiddiqui)
- Email: saqibahmad2004@gmail.com

## 🙏 Acknowledgments

- OpenWeather API for data
- Hopsworks for Feature Store
- Streamlit for dashboard framework
- GitHub Actions for CI/CD

## 📚 Documentation

For detailed documentation, see:
- [Setup Guide](docs/setup.md)
- [API Documentation](docs/api.md)
- [Feature Engineering](docs/features.md)
- [Model Training](docs/training.md)

---

**Made by Saqib for air quality check in Multan**