import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import pytz
import sys
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

sys.path.append(str(Path(__file__).parent.parent))
from src.prediction.predictor import AQIPredictor
from src.config.config import *

st.set_page_config(
    page_title="AQI Forecast - Multan",
    page_icon="🌫️",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {font-size: 3rem; font-weight: bold; color: #1f77b4; text-align: center; margin-bottom: 2rem;}
    .warning-box {background-color: #ff4444; color: white; padding: 1rem; border-radius: 10px; margin: 1rem 0; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

AQI_COLORS = {
    'Good': '#00e400',
    'Fair': '#ffff00',
    'Moderate': '#ff7e00',
    'Poor': '#ff0000',
    'Very Poor': '#8f3f97'
}

@st.cache_data(ttl=3600)
def load_predictions():
    predictions_path = PROCESSED_DATA_DIR / 'latest_predictions.csv'
    comparison_path = PROCESSED_DATA_DIR / 'model_comparison.csv'
    
    if predictions_path.exists():
        return pd.read_csv(predictions_path).fillna(""), pd.read_csv(comparison_path).fillna("")
    else:
        st.info("Generating predictions...")
        predictor = AQIPredictor()
        predictor.connect_hopsworks()
        predictor.load_all_models()
        predictions = predictor.predict_next_3_days()
        comparison = predictor.get_model_comparison()
        
        pred_df = pd.DataFrame(predictions)
        comp_df = pd.DataFrame(comparison)
        
        pred_df.to_csv(predictions_path, index=False)
        comp_df.to_csv(comparison_path, index=False)
        
        return pred_df, comp_df

def main():
    st.markdown('<div class="main-header">🌫️ AQI Forecast Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align:center; font-size:1.5rem; color:#555; margin-bottom:2rem;">3-Day Air Quality Predictions for Multan, Pakistan</div>', unsafe_allow_html=True)
    
    with st.spinner("Loading predictions..."):
        predictions_df, comparison_df = load_predictions()
    
    pkt = pytz.timezone(TIMEZONE)
    current_time = datetime.now(pkt)
    st.markdown(f"**Last Updated:** {current_time.strftime('%A, %B %d, %Y at %I:%M %p PKT')}")
    
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.radio("Go to", ["3-Day Forecast", "Model Comparison", "About"])
    
    if page == "3-Day Forecast":
        show_forecast(predictions_df, comparison_df)
    elif page == "Model Comparison":
        show_model_comparison(comparison_df)
    else:
        show_about()

def show_forecast(predictions_df, comparison_df):
    st.markdown("---")
    st.header("📅 Next 3 Days AQI Forecast")
    
    best_model = comparison_df.loc[comparison_df['r2_score'].idxmax()]
    st.info(f"**Predictions powered by:** {best_model['model']} (R² Score: {best_model['r2_score']:.3f})")
    
    # Check for warnings
    warnings = predictions_df[predictions_df['warning'].notna()]
    if len(warnings) > 0:
        st.markdown('<div class="warning-box">⚠️ AIR QUALITY WARNINGS</div>', unsafe_allow_html=True)
        for _, row in warnings.iterrows():
            st.error(f"**{row['day_name']}, {row['date']}**: {row['warning']}")
    
    cols = st.columns(3)
    
    for idx, row in predictions_df.iterrows():
        with cols[idx]:
            color = AQI_COLORS.get(row['category'], '#gray')
            
            st.markdown(f"### {row['day_name']}")
            st.markdown(f"**{row['date']}**")
            
            st.markdown(f"""
            <div style='text-align: center; padding: 1rem; margin: 1rem 0;
                        background-color: {color}; border-radius: 10px;'>
                <h1 style='color: white; margin: 0;'>{row['average_aqi']}</h1>
                <h3 style='color: white; margin: 0;'>{row['category']}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"**Range:** {row['min_aqi']} - {row['max_aqi']}")
            
            if pd.notna(row['warning']):
                st.warning(row['warning'])
    
    st.markdown("---")
    st.subheader("📈 AQI Trend")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=predictions_df['date'],
        y=predictions_df['average_aqi'],
        mode='lines+markers',
        name='Average AQI',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=10)
    ))
    
    fig.update_layout(
        title="3-Day AQI Forecast",
        xaxis_title="Date",
        yaxis_title="AQI",
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_model_comparison(comparison_df):
    st.markdown("---")
    st.header("🤖 Model Performance Comparison")
    
    best_model = comparison_df[comparison_df['is_best'] == True].iloc[0]
    
    st.markdown("---")
    st.subheader("🏆 Current Best Model")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Model", best_model['model'])
    with col2:
        st.metric("MAE", f"{best_model['mae']:.3f}")
    with col3:
        st.metric("RMSE", f"{best_model['rmse']:.3f}")
    with col4:
        st.metric("R² Score", f"{best_model['r2_score']:.3f}")
    
    st.markdown("---")
    st.subheader("📊 All Models")
    
    display_df = comparison_df.copy()
    display_df['Status'] = display_df['is_best'].apply(lambda x: '🏆 SELECTED' if x else '')
    
    st.dataframe(
        display_df[['model', 'mae', 'rmse', 'r2_score', 'Status']],
        use_container_width=True,
        hide_index=True
    )

def show_about():
    st.markdown("---")
    st.header("ℹ️ About")
    
    st.markdown("""
    ### 🎯 Project Overview
    
    3-day Air Quality Index (AQI) forecasting system for Multan, Pakistan using ML models.
    
    ### 🔍 How It Works
    
    1. **Hourly**: Collects air quality data from OpenWeather API
    2. **Daily**: Trains 3 models on all historical data
    3. **Prediction**: Selects best model and predicts next 3 days
    4. **Warnings**: Alerts for Poor and Hazardous air quality
    
    ### 📊 Models
    
    - Random Forest
    - XGBoost
    - LightGBM
    
    ### 🏙️ Location
    
    **Multan, Pakistan**
    - Latitude: 30.1979793
    - Longitude: 71.4724978
    
    ### 📈 Data
    
    Historical data from October 2025 to present, updated hourly.
    """)

if __name__ == "__main__":
    main()