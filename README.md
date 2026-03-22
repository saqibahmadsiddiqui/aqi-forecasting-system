# 🌫️ AQI Forecasting System - Multan, Pakistan

**Production-Ready 3-Day Air Quality Prediction with 5 ML Models & Complete MLOps Pipeline**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://multan-aqi.streamlit.app)
[![GitHub](https://img.shields.io/badge/GitHub-saqibahmadsiddiqui-blue?logo=github)](https://github.com/saqibahmadsiddiqui/aqi-forecasting-system)

## 🚀 Live Demo

- **📊 Dashboard**: https://multan-aqi.streamlit.app
- **📍 Location**: Multan, Punjab, Pakistan (30.1979°N, 71.4724°E)
- **⚡ Update Frequency**: Hourly data collection, Daily model training at 01:00 AM UTC (6:00 AM PKT)
- **🎯 Forecast Horizon**: 72-hour ahead with daily aggregates

## 🎯 Project Overview

An end-to-end production-ready machine learning system that predicts average Air Quality Index (AQI) for the next 3 days in Multan, Pakistan. Built with a **completely serverless architecture** using:

- **5 ML Classification Models** (Random Forest, Gradient Boosting, LightGBM, Decision Tree, Sklearn GB)
- **Hopsworks Feature Store** for centralized data management
- **GitHub Actions** for fully automated CI/CD pipelines
- **Streamlit Cloud** for interactive dashboard deployment
- **FastAPI** for REST API endpoints
- **OpenWeather API** for real-time air quality data

The system **continuously and automatically**:
- 📥 Collects hourly air quality data via OpenWeather API
- 🔧 Engineers 40+ intelligent features (temporal, lags, rolling stats, interactions)
- 🤖 Trains 5 classification models daily with NaN handling & imbalance management
- 📈 Generates accurate 72-hour recursive forecasts aggregated to 3-day summaries
- 🏆 Auto-selects best model based on F1 score
- 📊 Provides real-time interactive visualizations with health alerts
- 🔌 Serves predictions via REST API for integration

---

## ✨ Key Features

### 🤖 Machine Learning (5 Models)
- ✅ **Random Forest Classifier**: 200 trees, balanced class weights
- ✅ **Histogram Gradient Boosting**: 150 iterations with NaN handling
- ✅ **LightGBM Classifier**: Fast gradient boosting, optimized for performance
- ✅ **Decision Tree Classifier**: Interpretable baseline model
- ✅ **Sklearn Gradient Boosting**: 150 iterations with SimpleImputer
- ✅ **Auto Model Selection**: Selects best model by F1 score daily
- ✅ **Class Imbalance Handling**: Stratified split + class_weight='balanced'
- ✅ **NaN Value Handling**: SimpleImputer with median strategy

### 📊 Data Pipeline
- ✅ **Hourly Data Collection**: Real-time OpenWeather API integration
- ✅ **Smart Duplicate Detection**: Compares with last 6 hours
- ✅ **42+ Feature Engineering**: Temporal, lag, rolling stats, interactions
- ✅ **Historical Backfill**: Loads 6,650+ records (Jan-Oct 2025)
- ✅ **Hopsworks Feature Store**: Centralized data with versioning
- ✅ **Daily Retraining**: Automatic with cumulative historical data

### 🎨 User Interface
- ✅ **Beautiful Dashboard**: Dark theme with cyan accents, smooth animations
- ✅ **3-Day Forecast Cards**: Color-coded AQI (Good→Very Poor) with alerts
- ✅ **Interactive Charts**: Plotly-powered trend visualization
- ✅ **Model Comparison**: Real-time F1, Accuracy, Precision, Recall metrics
- ✅ **System Information**: Complete architecture details
- ✅ **Responsive Design**: Desktop, tablet, and mobile support
- ✅ **Smart Caching**: 1-hour cache for performance optimization

### 🔌 API & Integration
- ✅ **REST API**: FastAPI with Swagger documentation
- ✅ **JSON Responses**: Structured prediction and metrics data
- ✅ **CORS Enabled**: Cross-origin requests supported
- ✅ **Multiple Endpoints**: `/predict`, `/models`, `/models/best`, `/status`, `/info`
- ✅ **Error Handling**: Meaningful error messages and graceful failures
- ✅ **Auto Documentation**: Interactive API explorer at `/docs`

### 🚀 DevOps & Automation
- ✅ **GitHub Actions CI/CD**: Hourly data + daily training fully automated
- ✅ **100% Serverless**: No server management required
- ✅ **Production-Ready**: Comprehensive error handling, logging, monitoring
- ✅ **Git Integration**: Automatic commits after each pipeline run
- ✅ **Model Versioning**: Full version control in Hopsworks
- ✅ **Scheduled Execution**: Reliable cron-based automation

---

## 📈 Current Model Performance

| Model | F1 Score | Accuracy | Precision | Recall | Status |
|-------|----------|----------|-----------|--------|--------|
| **Gradient Boosting** | **0.9985** | **0.9985** | **0.9985** | **0.9985** | 🥇 Best |
| **LightGBM** | **0.9985** | **0.9985** | **0.9985** | **0.9985** | 🥇 Tied |
| **Random Forest** | 0.9897 | 0.9897 | 0.9898 | 0.9897 | ✅ Excellent |
| **Decision Tree** | 0.9904 | 0.9904 | 0.9905 | 0.9904 | ✅ Excellent |
| **Sklearn GB** | 0.9871 | 0.9871 | 0.9872 | 0.9871 | ✅ Good |

*Metrics updated daily. All 5 models trained on 6,790+ records with AQI categories (1-5).*

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Presentation Layer                          │
├──────────────────────┬──────────────┬───────────────────────────┤
│  Streamlit Cloud     │  FastAPI     │  Streamlit Local/Docker   │
│  Dashboard           │  REST API    │  Development              │
│  (Production)        │  (Optional)  │  (Testing)                │
└──────────┬───────────┴──────┬───────┴──────────┬────────────────┘
           │                  │                  │
           └──────────────────┼──────────────────┘
                              ↓
        ┌──────────────────────────────────────────┐
        │   Hopsworks Feature Store & Registry     │
        │                                          │
        │  ├─ Raw Features (19)                    │
        │  │  - Pollutants, Weather, Temporal      │
        │  │                                       │
        │  ├─ Engineered Features (42+)            │
        │  │  - Lags, Rolling Stats, Interactions  │
        │  │                                       │
        │  ├─ Model Registry (5 models)            │
        │  │  - All versions with metrics          │
        │  │                                       │
        │  └─ Predictions (3-day forecast)         │
        │     - Latest forecast results            │
        └───────────┬──────────────────────────────┘
                    ↑
        ┌───────────┴────────────────┐
        │                            │
        ↓                            ↓
    ┌──────────────┐         ┌──────────────────────┐
    │  Hourly      │         │  Daily Training      │
    │  Pipeline    │         │  Pipeline            │
    │              │         │                      │
    │  (Every Hour)│         │  (12:00 AM UTC)      │
    │              │         │  (5:00 AM PKT)       │
    │  • Collect   │         │                      │
    │    data      │         │  • Load historical   │
    │  • Check     │         │  • Handle NaN values │
    │    duplicates│         │  • Train 5 models    │
    │  • Engineer  │         │  • Evaluate metrics  │
    │    42+       │         │  • Select best model │
    │    features  │         │  • Register models   │
    │  • Upload    │         │  • Generate forecasts│
    │              │         │  • Save predictions  │
    │  Duration:   │         │  • Push to GitHub    │
    │  < 2 min     │         │                      │
    │              │         │  Duration: 3-5 min   │
    └──────────────┘         └──────────────────────┘
        ↑                            ↑
        └────────────────────────────┘
             GitHub Actions
           (Fully Automated)
             ↑
             │
        ┌─────────────────┐
        │ OpenWeather API │
        │  (Real-time)    │
        │                 │
        │ • Air Quality   │
        │ • Weather Data  │
        └─────────────────┘
```

---

## 🛠️ Tech Stack

### Machine Learning & Data Science
- **scikit-learn** (1.3+): Random Forest, preprocessing, metrics
- **LightGBM** (4.0+): Fast gradient boosting classifier
- **pandas** (2.0+): Data manipulation and analysis
- **numpy** (1.24+): Numerical computations
- **joblib**: Model persistence and serialization

### MLOps & Feature Store
- **Hopsworks** (4.7+): Feature store, model registry, versioning
- **GitHub Actions**: CI/CD automation (hourly & daily)

### Web & API
- **Streamlit** (1.28+): Interactive dashboard with animations
- **FastAPI** (0.104+): Modern REST API framework
- **Uvicorn** (0.24+): ASGI server
- **Plotly** (5.17+): Interactive data visualizations
- **python-dotenv** (1.0+): Environment management

### Data Collection
- **OpenWeather API**: Real-time weather & air quality data
- **requests**: HTTP client for API calls

### Infrastructure & Deployment
- **Streamlit Cloud**: Dashboard hosting (free tier)
- **GitHub**: Repository & CI/CD platform
- **Hopsworks Cloud**: Feature store & model registry

---

## 📦 Installation & Setup

### Prerequisites
- Python 3.11 or higher
- GitHub account (for CI/CD)
- Hopsworks account ([free tier](https://www.hopsworks.ai/))
- OpenWeather API key ([free tier](https://openweathermap.org/api))

### Step 1: Clone Repository

```bash
git clone https://github.com/saqibahmadsiddiqui/aqi-forecasting-system.git
cd aqi-forecasting-system
```

### Step 2: Create Virtual Environment

```bash
# Create
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables

Create `.env` file in project root:

```
# Hopsworks Configuration
HOPSWORKS_HOST=eu-west.cloud.hopsworks.ai
HOPSWORKS_API_KEY=your_hopsworks_api_key_here
HOPSWORKS_PROJECT_NAME=aqi_multan

# Feature Group Settings
FEATURE_GROUP_NAME=aqi_multan_features
FEATURE_GROUP_VERSION=1

# OpenWeather API
OPENWEATHER_API_KEY=your_openweather_api_key_here

# Location (Multan, Pakistan)
LAT=30.1979793
LON=71.4724978
TIMEZONE=Asia/Karachi
```

### Step 5: Verify Installation

```bash
python -c "import hopsworks; import streamlit; import lightgbm; print('All imports successful!')"
```

---

## 🎯 Usage Guide

### Option 1: Run Dashboard (Recommended for Users)

```bash
streamlit run dashboard/app.py
```

**Opens at**: http://localhost:8501

**Dashboard Sections**:
- 🔮 **3-Day Forecast**: AQI predictions with color-coded alerts
- 📊 **Model Comparison**: Performance metrics for all 5 models
- ℹ️ **About**: System architecture and feature details

### Option 2: Run REST API (For Developers)

```bash
uvicorn api.main:app --reload
```

**API Server at**: http://localhost:8000

**Available Endpoints**:
- `GET /` → Health check
- `GET /health` → System status
- `GET /predict` → Get 3-day predictions
- `GET /models` → All models with metrics
- `GET /models/best` → Best performing model
- `GET /status` → System status details
- `GET /info` → API information
- `GET /docs` → Interactive API documentation

### Option 3: Run Training Pipeline (Manual)

```bash
# Train all 5 models
python src/models/daily_training.py

# Generate forecasts
python src/prediction/predictor.py
```
---

## 🔄 Automated Pipelines

### Hourly Data Collection Pipeline

**Trigger**: Every hour at :00 UTC (7:00, 8:00, 9:00 AM PKT, etc.)

```yaml
Workflow: .github/workflows/hourly_data_pipeline.yml

Tasks:
  1. Fetch real-time air quality data from OpenWeather API
  2. Engineer 40+ features (temporal, lags, rolling statistics, interactions)
  3. Check for duplicates in last 6 hours to avoid redundancy
  4. Upload to Hopsworks Feature Store
  5. Commit changes to GitHub
  
Duration: < 2 minutes per hour
```

### Daily Training & Prediction Pipeline

**Trigger**: Daily at 12:00 AM UTC (5:00 AM PKT)

```yaml
Workflow: .github/workflows/daily_training_pipeline.yml

Tasks:
  1. Load 6,500+ historical records from Feature Store
  2. Handle missing values (NaN imputation with SimpleImputer)
  3. Check class distribution and handle imbalance
  4. Split data (80% train, 20% test with stratification)
  5. Train 5 classification models:
     ├─ Random Forest Classifier (200 trees)
     ├─ Histogram Gradient Boosting Classifier (150 iterations)
     ├─ LightGBM Classifier (150 trees, max_depth=10)
     ├─ Decision Tree Classifier (max_depth=15)
     └─ Sklearn Gradient Boosting Classifier (150 iterations)
  6. Evaluate all models (F1, Accuracy, Precision, Recall)
  7. Select best model by F1 score
  8. Register all models in Hopsworks Model Registry
  9. Generate 72-hour recursive forecasts with feature updates
  10. Aggregate to 3-day summaries with min/max ranges
  11. Save predictions to CSV files
  12. Commit results and push to GitHub
  
Duration: 3-5 minutes
```

---

## 📈 Feature Engineering Pipeline

### Raw Features (19)
- **Pollutants** (8): CO, NO, NO2, O3, SO2, PM2.5, PM10, NH3
- **Weather** (5): Temperature, Humidity, Pressure, Wind Speed, Cloud Cover
- **Target** (1): AQI (1-5 scale)
- **Temporal** (2): Date, Time
- **Metadata** (3): Latitude, Longitude, Elevation

### Engineered Features (40+)

| Category | Features | Count |
|----------|----------|-------|
| **Temporal** | hour_sin/cos, day_of_week_sin/cos, month_sin/cos, is_weekend | 7 |
| **Lag Features** | aqi_lag_{1,3,6,12,24,48}h, pm2_5_lag_{1,3,6,24}h, pm10_lag_{1,3,24}h | 14 |
| **Rolling Mean** | aqi_rolling_mean_{3,6,12,24}h, pm2_5_rolling_mean_{6,24}h | 6 |
| **Rolling Std** | aqi_rolling_std_{6,24}h | 2 |
| **Min/Max** | aqi_rolling_min/max_24h | 2 |
| **Change** | aqi_change_24h, pm2_5_change_24h | 2 |
| **Interactions** | pm2_5_x_wind_speed | 1 |
| **Raw** | All original features | 19 |
| | **TOTAL** | **42+** |

---

## 🎨 Dashboard Interface

### 3-Day Forecast Page
- Interactive forecast cards with AQI values (1-5)
- Color-coded categories: Good 🟢 → Very Poor 🔴
- Min-Max ranges for each day
- Health alerts and warnings
- Animated trend chart with Plotly
- Best model information

### Model Comparison Page
- Best model performance highlighted
- Individual metrics for all 5 models (F1, Accuracy, Precision, Recall)
- Interactive comparison bar chart
- Detailed metrics table
- Model version information

### About Page
- System architecture explanation
- Data pipeline overview
- ML models description
- Feature engineering details
- AQI categories guide
- Location and update frequency information

### UI/UX Features
- ✅ Dark theme with cyan accents
- ✅ Smooth sidebar animations
- ✅ Real-time PKT timezone display
- ✅ Responsive mobile design
- ✅ System status indicator
- ✅ 1-hour auto-cache
- ✅ Manual refresh capability

---

## 🔌 API Usage Examples

### Python Client

```python
import requests

# Get predictions
response = requests.get("http://localhost:8000/predict")
predictions = response.json()

for pred in predictions:
    print(f"{pred['date']}: {pred['category']} (AQI: {pred['average_aqi']})")

# Get best model
response = requests.get("http://localhost:8000/models/best")
best_model = response.json()
print(f"Best Model: {best_model['model']} (F1: {best_model['f1_score']:.4f})")
```

### JavaScript/React

```javascript
// Fetch predictions
fetch('http://localhost:8000/predict')
  .then(res => res.json())
  .then(predictions => {
    console.log(predictions);
  });
```

### cURL

```bash
# Get predictions
curl http://localhost:8000/predict | jq

# Get best model
curl http://localhost:8000/models/best | jq

# Check system status
curl http://localhost:8000/status | jq
```

---

## 📊 Project Structure

```
aqi-forecasting-system/
├── .github/
│   └── workflows/
│       ├── daily_training_pipeline.yml
│       └── hourly_data_pipeline.yml
├── api/
│   └── main.py
├── dashboard/
│   └── app.py
├── src/
│   ├── config/
│   │   └── config.py                        # Configuration & constants
│   ├── models/
│   │   └── daily_training.py                # 5 model training script
│   ├── prediction/
│   │   └── predictor.py                     # 72-hour forecasting
│   └── setup/
│       ├── historical_data.py               # Historical data backfill
│       ├── initial_data_load.py             # Initial data backfill
│       ├── debug_upload.py                  # Get feature store schema
│       └── verify_upload.py                 # Get data information
│
├── data/
│   └── processed/                           # Final predictions
│       ├── latest_predictions.csv           # 3-day forecast
│       └── model_comparison.csv             # Model metrics
│
├── .env                                     # Environment variables
├── .gitignore                               # Git ignore rules
├── requirements.txt                         # Python dependencies
└── README.md                                # This file
```

---

## 🚀 Deployment Guide

### Deploy Dashboard to Streamlit Cloud

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Deploy to Streamlit Cloud"
   git push origin main
   ```

2. **Create Streamlit Cloud App**
   - Visit https://share.streamlit.io
   - Sign in with GitHub
   - Click "New app"
   - Select repository: `aqi-forecasting-system`
   - Select main file: `dashboard/app.py`

3. **Add Secrets**
   - Click Settings (gear icon)
   - Go to Secrets
   - Add environment variables:
     ```toml
     HOPSWORKS_HOST = "eu-west.cloud.hopsworks.ai"
     HOPSWORKS_API_KEY = "your_key_here"
     HOPSWORKS_PROJECT_NAME = "AQI_MULTAN"
     FEATURE_GROUP_NAME = "aqi_multan_features"
     FEATURE_GROUP_VERSION = 1
     OPENWEATHER_API_KEY = "your_key_here"
     LAT = "30.1979793"
     LON = "71.4724978"
     TIMEZONE = "Asia/Karachi"
     ```

4. **Deploy**
   - First deploy: 8-12 minutes
   - Subsequent: 3-5 minutes
   - Your dashboard will be live!

### Deploy API to Cloud Run

```bash
# Create Dockerfile
cat > Dockerfile << 'EOF'
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src/ ./src/
ENV PYTHONUNBUFFERED=1
CMD ["uvicorn", "api.main:app", "--reload"]
EOF

# Deploy to Google Cloud Run
gcloud run deploy aqi-api \
  --source . \
  --platform managed \
  --region us-central1 \
  --set-env-vars HOPSWORKS_API_KEY=your_key
```

---

## 🔄 GitHub Actions Configuration

### Required Secrets

Navigate to: **Settings → Secrets and variables → Actions**

1. **HOPSWORKS_API_KEY**
   - From: Hopsworks → Account Settings → API Keys

2. **HOPSWORKS_HOST**
   - Value: `eu-west.cloud.hopsworks.ai`

3. **HOPSWORKS_PROJECT_NAME**
   - Your Hopsworks project name

4. **OPENWEATHER_API_KEY**
   - From: OpenWeather → API Keys

### Workflows Included

Both workflows are fully automated:
- ✅ Run on schedule (hourly & daily)
- ✅ Use GitHub Secrets securely
- ✅ Handle errors gracefully
- ✅ Commit results automatically
- ✅ Send failure notifications

---

## 📈 Performance & Benchmarks

### Training Performance
- **Initial Training**: 5-10 minutes (5 models)
- **Daily Retraining**: 3-5 minutes
- **Feature Engineering**: 1-2 minutes per 1000 records
- **Prediction Generation**: 30 seconds

### API Performance
- **Prediction Endpoint**: <500ms
- **Models Endpoint**: <200ms
- **Health Check**: <100ms
- **Cold Start**: ~2 seconds

### Data Pipeline
- **Hourly Collection**: ~2 minutes total
- **Feature Engineering**: ~20 seconds
- **Upload to Hopsworks**: ~10 seconds
- **Total per hour**: <2 minutes

### Dashboard Performance
- **Page Load**: 2-3 seconds
- **Chart Rendering**: <1 second
- **Data Refresh**: Every hour (cached)
- **Mobile Load**: 3-5 seconds

---

## 🐛 Troubleshooting

### Dashboard Shows "No Predictions"
**Solution**: Run the prediction pipeline first
```bash
python src/models/daily_training.py
python src/prediction/predictor.py
```

### Hopsworks Connection Error
**Check**:
1. API key is valid
2. Project name is correct
3. Internet connection is active

**Debug**:
```bash
python -c "from src.prediction.predictor import AQIPredictor; \
           p = AQIPredictor(); p.connect_hopsworks(); print('✅ Connected!')"
```

### OpenWeather API Fails
**Check**:
1. API key is valid
2. Rate limit (free: 1000 calls/day)
3. Coordinates are correct

**Debug**:
```bash
curl "https://api.openweathermap.org/data/2.5/air_pollution?lat=30.1979&lon=71.4724&appid=YOUR_KEY"
```

### NaN Values in Training
**Solution**: Already handled with SimpleImputer (median strategy)

### Class Imbalance Warning
**Solution**: Handled with stratified split + class_weight='balanced'

---

## 📚 Documentation Files

Comprehensive guides available:

- **README.md** - Complete project overview
- **Project Report.pdf** - Detailed documentation

---

## 🤝 Contributing

Contributions are welcome! Areas for enhancement:

- Additional ML models (LSTM, XGBoost, Logistic Regression)
- More engineered features
- Real-time notifications
- Mobile app version
- Multi-city support
- Additional visualizations
- Documentation improvements

**Steps**:
1. Fork the repository
2. Create feature branch: `git checkout -b feature/YourFeature`
3. Commit changes: `git commit -m 'Add YourFeature'`
4. Push to branch: `git push origin feature/YourFeature`
5. Open Pull Request

---

## 👨‍💻 Author

**Saqib Ahmad Siddiqui**
- 🔗 **GitHub**: [@saqibahmadsiddiqui](https://github.com/saqibahmadsiddiqui)
- 📧 **Email**: saqibahmad2004@gmail.com
- 💼 **LinkedIn**: [saqib-ahmad-siddiqui](https://linkedin.com/in/saqib-ahmad-siddiqui)

---

## 🙏 Acknowledgments

- **OpenWeather API** - Real-time weather and air quality data
- **Hopsworks** - Feature store and MLOps platform
- **Streamlit** - Interactive dashboard framework
- **FastAPI** - Modern Python web framework
- **GitHub Actions** - CI/CD automation
- **scikit-learn, LightGBM** - Machine learning libraries

---

## 📞 Support

For issues, questions, or suggestions:

- **GitHub Issues**: [Report bugs](https://github.com/saqibahmadsiddiqui/aqi-forecasting-system/issues)
- **Email**: saqibahmad2004@gmail.com
- **Dashboard**: https://multan-aqi.streamlit.app
- **Documentation**: See docs folder

---

## 🎯 Roadmap

### ✅ Completed (v1.0)
- 5 ML classification models with auto-selection
- 40+ engineered features
- 72-hour recursive forecasting
- Streamlit interactive dashboard
- FastAPI REST endpoints
- GitHub Actions CI/CD
- Hopsworks feature store integration
- Historical data backfill (6,650+ records)
- NaN handling with SimpleImputer
- Class imbalance handling

### 📋 Planned (v2.0)
- LSTM/RNN models for temporal patterns
- Real-time mobile notifications
- Multi-city support
- Advanced visualization dashboards
- Model explainability (SHAP)
- A/B testing framework
- Ensemble predictions

---

**Status**: ✅ **Production Ready**  
**Last Updated**: February 2026  
**Version**: 1.0  
**Maintained**: Yes