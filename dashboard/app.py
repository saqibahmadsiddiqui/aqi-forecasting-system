# import streamlit as st
# import pandas as pd
# import plotly.graph_objects as go
# import plotly.express as px
# from datetime import datetime
# import pytz
# import sys
# from pathlib import Path
# import os
# from dotenv import load_dotenv

# load_dotenv()

# sys.path.append(str(Path(__file__).parent.parent))
# from src.prediction.predictor import AQIPredictor
# from src.config.config import *

# st.set_page_config(
#     page_title="AQI Forecast - Multan",
#     page_icon="🌫️",
#     layout="wide"
# )

# st.markdown("""
# <style>
#     .main-header {font-size: 3rem; font-weight: bold; color: #1f77b4; text-align: center; margin-bottom: 2rem;}
#     .warning-box {background-color: #ff4444; color: white; padding: 1rem; border-radius: 10px; margin: 1rem 0; font-weight: bold;}
# </style>
# """, unsafe_allow_html=True)

# AQI_COLORS = {
#     'Good': '#00e400',
#     'Fair': '#ffff00',
#     'Moderate': '#ff7e00',
#     'Poor': '#ff0000',
#     'Very Poor': '#8f3f97'
# }

# @st.cache_data(ttl=3600)
# def load_predictions():
#     predictions_path = PROCESSED_DATA_DIR / 'latest_predictions.csv'
#     comparison_path = PROCESSED_DATA_DIR / 'model_comparison.csv'
    
#     if predictions_path.exists():
#         return pd.read_csv(predictions_path).fillna(""), pd.read_csv(comparison_path).fillna("")
#     else:
#         st.info("Generating predictions...")
#         predictor = AQIPredictor()
#         predictor.connect_hopsworks()
#         predictor.load_all_models()
#         predictions = predictor.predict_next_3_days()
#         comparison = predictor.get_model_comparison()
        
#         pred_df = pd.DataFrame(predictions)
#         comp_df = pd.DataFrame(comparison)
        
#         pred_df.to_csv(predictions_path, index=False)
#         comp_df.to_csv(comparison_path, index=False)
        
#         return pred_df, comp_df

# def main():
#     st.markdown('<div class="main-header">🌫️ AQI Forecast Dashboard</div>', unsafe_allow_html=True)
#     st.markdown('<div style="text-align:center; font-size:1.5rem; color:#555; margin-bottom:2rem;">3-Day Air Quality Predictions for Multan, Pakistan</div>', unsafe_allow_html=True)
    
#     with st.spinner("Loading predictions..."):
#         predictions_df, comparison_df = load_predictions()
    
#     pkt = pytz.timezone(TIMEZONE)
#     current_time = datetime.now(pkt)
#     st.markdown(f"**Last Updated:** {current_time.strftime('%A, %B %d, %Y at %I:%M %p PKT')}")
    
#     st.sidebar.title("📊 Navigation")
#     page = st.sidebar.radio("Go to", ["3-Day Forecast", "Model Comparison", "About"])
    
#     if page == "3-Day Forecast":
#         show_forecast(predictions_df, comparison_df)
#     elif page == "Model Comparison":
#         show_model_comparison(comparison_df)
#     else:
#         show_about()

# def show_forecast(predictions_df, comparison_df):
#     st.markdown("---")
#     st.header("📅 Next 3 Days AQI Forecast")
    
#     best_model = comparison_df.loc[comparison_df['r2_score'].idxmax()]
#     st.info(f"**Predictions powered by:** {best_model['model']} (R² Score: {best_model['r2_score']:.3f})")
    
#     # Check for warnings
#     warnings = predictions_df[predictions_df['warning'].notna()]
#     if len(warnings) > 0:
#         st.markdown('<div class="warning-box">⚠️ AIR QUALITY WARNINGS</div>', unsafe_allow_html=True)
#         for _, row in warnings.iterrows():
#             st.error(f"**{row['day_name']}, {row['date']}**: {row['warning']}")
    
#     cols = st.columns(3)
    
#     for idx, row in predictions_df.iterrows():
#         with cols[idx]:
#             color = AQI_COLORS.get(row['category'], '#gray')
            
#             st.markdown(f"### {row['day_name']}")
#             st.markdown(f"**{row['date']}**")
            
#             st.markdown(f"""
#             <div style='text-align: center; padding: 1rem; margin: 1rem 0;
#                         background-color: {color}; border-radius: 10px;'>
#                 <h1 style='color: white; margin: 0;'>{row['average_aqi']}</h1>
#                 <h3 style='color: white; margin: 0;'>{row['category']}</h3>
#             </div>
#             """, unsafe_allow_html=True)
            
#             st.markdown(f"**Range:** {row['min_aqi']} - {row['max_aqi']}")
            
#             if pd.notna(row['warning']):
#                 st.warning(row['warning'])
    
#     st.markdown("---")
#     st.subheader("📈 AQI Trend")
    
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(
#         x=predictions_df['date'],
#         y=predictions_df['average_aqi'],
#         mode='lines+markers',
#         name='Average AQI',
#         line=dict(color='#1f77b4', width=3),
#         marker=dict(size=10)
#     ))
    
#     fig.update_layout(
#         title="3-Day AQI Forecast",
#         xaxis_title="Date",
#         yaxis_title="AQI",
#         hovermode='x unified',
#         height=400
#     )
    
#     st.plotly_chart(fig, use_container_width=True)

# def show_model_comparison(comparison_df):
#     st.markdown("---")
#     st.header("🤖 Model Performance Comparison")
    
#     best_model = comparison_df[comparison_df['is_best'] == True].iloc[0]
    
#     st.markdown("---")
#     st.subheader("🏆 Current Best Model")
    
#     col1, col2, col3, col4 = st.columns(4)
#     with col1:
#         st.metric("Model", best_model['model'])
#     with col2:
#         st.metric("MAE", f"{best_model['mae']:.3f}")
#     with col3:
#         st.metric("RMSE", f"{best_model['rmse']:.3f}")
#     with col4:
#         st.metric("R² Score", f"{best_model['r2_score']:.3f}")
    
#     st.markdown("---")
#     st.subheader("📊 All Models")
    
#     display_df = comparison_df.copy()
#     display_df['Status'] = display_df['is_best'].apply(lambda x: '🏆 SELECTED' if x else '')
    
#     st.dataframe(
#         display_df[['model', 'mae', 'rmse', 'r2_score', 'Status']],
#         use_container_width=True,
#         hide_index=True
#     )

# def show_about():
#     st.markdown("---")
#     st.header("ℹ️ About")
    
#     st.markdown("""
#     ### 🎯 Project Overview
    
#     3-day Air Quality Index (AQI) forecasting system for Multan, Pakistan using ML models.
    
#     ### 🔍 How It Works
    
#     1. **Hourly**: Collects air quality data from OpenWeather API
#     2. **Daily**: Trains 3 models on all historical data
#     3. **Prediction**: Selects best model and predicts next 3 days
#     4. **Warnings**: Alerts for Poor and Hazardous air quality
    
#     ### 📊 Models
    
#     - Random Forest
#     - XGBoost
#     - LightGBM
    
#     ### 🏙️ Location
    
#     **Multan, Pakistan**
#     - Latitude: 30.1979793
#     - Longitude: 71.4724978
    
#     ### 📈 Data
    
#     Historical data from October 2025 to present, updated hourly.
#     """)

# if __name__ == "__main__":
#     main()

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
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM STYLING ====================
st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
    }
    
    /* Main background */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    [data-testid="stMain"] {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Header styling */
    .main-header {
        font-size: 3.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }
    
    .subtitle {
        text-align: center;
        font-size: 1.3rem;
        color: #555;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    
    /* Card styling */
    .forecast-card {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.5);
    }
    
    .forecast-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.15);
    }
    
    /* AQI Circle */
    .aqi-circle {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 140px;
        height: 140px;
        border-radius: 50%;
        margin: 1.5rem auto;
        font-size: 3rem;
        font-weight: bold;
        color: white;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15); }
        50% { box-shadow: 0 8px 35px rgba(0, 0, 0, 0.25); }
    }
    
    /* Warning box */
    .warning-box {
        background: linear-gradient(135deg, #ff6b6b 0%, #ff5252 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        font-weight: bold;
        border-left: 5px solid #ff3838;
        box-shadow: 0 8px 20px rgba(255, 107, 107, 0.3);
    }
    
    /* Info box */
    .info-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        font-weight: 500;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
    }
    
    /* Status badge */
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    .status-badge.best {
        background: linear-gradient(135deg, #ffd89b 0%, #ff9d3c 100%);
        color: white;
    }
    
    /* Divider */
    .divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 2rem 0;
    }
    
    /* Section header */
    .section-header {
        font-size: 2rem;
        font-weight: 800;
        color: #333;
        margin: 2rem 0 1.5rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Metric card */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(102, 126, 234, 0.1);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] > div > div > div {
        color: white !important;
    }
    
    /* Last updated */
    .last-updated {
        text-align: center;
        color: #999;
        font-size: 0.95rem;
        margin: 1rem 0;
        font-weight: 500;
    }
    
    /* Day name styling */
    .day-name {
        font-size: 1.5rem;
        font-weight: 800;
        color: #333;
        margin: 1rem 0 0.5rem 0;
    }
    
    .date-text {
        color: #999;
        font-size: 1rem;
        margin-bottom: 1rem;
    }
    
    .aqi-range {
        color: #666;
        font-size: 1.1rem;
        margin: 1rem 0;
        font-weight: 600;
    }
    
    /* About section */
    .about-box {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
        border-left: 4px solid #667eea;
    }
    
    .about-box h3 {
        color: #667eea;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }
    
    .about-box h3:first-child {
        margin-top: 0;
    }
</style>
""", unsafe_allow_html=True)

# ==================== COLOR MAPPING ====================
AQI_COLORS = {
    'Good': '#00e400',
    'Fair': '#ffff00',
    'Moderate': '#ff7e00',
    'Poor': '#ff0000',
    'Very Poor': '#8f3f97'
}

AQI_COLOR_HEX = {
    'Good': '#00E400',
    'Fair': '#FFFF00',
    'Moderate': '#FF7E00',
    'Poor': '#FF0000',
    'Very Poor': '#8F3F97'
}

# ==================== CACHE & DATA LOADING ====================
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

# ==================== HELPER FUNCTIONS ====================
def get_aqi_emoji(category):
    """Return emoji based on AQI category"""
    emojis = {
        'Good': '😊',
        'Fair': '🙂',
        'Moderate': '😐',
        'Poor': '😷',
        'Very Poor': '😵'
    }
    return emojis.get(category, '❓')

def get_aqi_gradient(category):
    """Return gradient CSS for AQI circles"""
    gradients = {
        'Good': 'linear-gradient(135deg, #00e400 0%, #00c800 100%)',
        'Fair': 'linear-gradient(135deg, #ffff00 0%, #ffcc00 100%)',
        'Moderate': 'linear-gradient(135deg, #ff7e00 0%, #ff6600 100%)',
        'Poor': 'linear-gradient(135deg, #ff0000 0%, #cc0000 100%)',
        'Very Poor': 'linear-gradient(135deg, #8f3f97 0%, #6d2d7a 100%)'
    }
    return gradients.get(category, 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)')

def create_trend_chart(predictions_df):
    """Create an interactive trend chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=predictions_df['date'],
        y=predictions_df['average_aqi'],
        mode='lines+markers',
        name='AQI',
        line=dict(
            color='#667eea',
            width=4,
            shape='spline'
        ),
        marker=dict(
            size=12,
            color='#667eea',
            symbol='circle',
            line=dict(color='white', width=2)
        ),
        fill='tozeroy',
        fillcolor='rgba(102, 126, 234, 0.1)',
        hovertemplate='<b>%{x}</b><br>AQI: %{y:.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': '📈 3-Day AQI Forecast Trend',
            'font': {'size': 20, 'color': '#333', 'family': 'Arial Black'}
        },
        xaxis_title='Date',
        yaxis_title='AQI Value',
        hovermode='x unified',
        height=400,
        template='plotly_white',
        plot_bgcolor='rgba(245, 247, 250, 0.5)',
        paper_bgcolor='white',
        font=dict(family='Arial, sans-serif', size=12),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig

def create_model_comparison_chart(comparison_df):
    """Create model comparison visualization"""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=comparison_df['model'],
        y=comparison_df['r2_score'],
        name='R² Score',
        marker=dict(
            color=comparison_df['r2_score'],
            colorscale='Viridis',
            showscale=False,
            line=dict(color='white', width=2)
        ),
        text=[f"{v:.3f}" for v in comparison_df['r2_score']],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>R² Score: %{y:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': '🤖 Model Performance (R² Score)',
            'font': {'size': 20, 'color': '#333', 'family': 'Arial Black'}
        },
        xaxis_title='Model',
        yaxis_title='R² Score',
        height=400,
        template='plotly_white',
        plot_bgcolor='rgba(245, 247, 250, 0.5)',
        paper_bgcolor='white',
        font=dict(family='Arial, sans-serif', size=12),
        showlegend=False,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig

# ==================== PAGE FUNCTIONS ====================
def show_forecast(predictions_df, comparison_df):
    """Display 3-day forecast with enhanced styling"""
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">📅 Next 3 Days Forecast</div>', unsafe_allow_html=True)
    
    best_model = comparison_df.loc[comparison_df['r2_score'].idxmax()]
    st.markdown(f'''
    <div class="info-box">
        ✨ <b>Predictions powered by:</b> {best_model['model']} 
        <span style="float: right;">R² Score: <b>{best_model['r2_score']:.3f}</b></span>
    </div>
    ''', unsafe_allow_html=True)
    
    # Check for warnings
    warnings = predictions_df[predictions_df['warning'].notna()]
    if len(warnings) > 0:
        st.markdown('<div class="warning-box">⚠️ AIR QUALITY WARNINGS</div>', unsafe_allow_html=True)
        for _, row in warnings.iterrows():
            st.error(f"🚨 **{row['day_name']}, {row['date']}**: {row['warning']}")
    
    # Display 3-day forecast cards
    cols = st.columns(3, gap="large")
    
    for idx, (_, row) in enumerate(predictions_df.iterrows()):
        with cols[idx]:
            st.markdown(f'''
            <div class="forecast-card">
                <div class="day-name">{row['day_name']} {get_aqi_emoji(row['category'])}</div>
                <div class="date-text">{row['date']}</div>
            </div>
            ''', unsafe_allow_html=True)
            
            gradient = get_aqi_gradient(row['category'])
            st.markdown(f'''
            <div style="text-align: center;">
                <div class="aqi-circle" style="background: {gradient};">
                    {int(row['average_aqi'])}
                </div>
            </div>
            ''', unsafe_allow_html=True)
            
            st.markdown(f'''
            <div class="forecast-card">
                <div style="text-align: center;">
                    <h3 style="color: #333; margin: 0 0 1rem 0;">{row['category']}</h3>
                    <div class="aqi-range">Range: <b>{int(row['min_aqi'])} - {int(row['max_aqi'])}</b></div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
            
            if pd.notna(row['warning']):
                st.warning(f"⚠️ {row['warning']}")
    
    # Trend chart
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    fig = create_trend_chart(predictions_df)
    st.plotly_chart(fig, use_container_width=True)

def show_model_comparison(comparison_df):
    """Display model comparison with enhanced styling"""
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">🤖 Model Performance Comparison</div>', unsafe_allow_html=True)
    
    # Best model section
    best_model = comparison_df[comparison_df['is_best'] == True].iloc[0]
    
    st.markdown('<div class="section-header">🏆 Current Best Model</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4, gap="large")
    
    with col1:
        st.markdown(f'''
        <div class="metric-card">
            <div style="color: #667eea; font-size: 0.9rem; font-weight: bold; margin-bottom: 0.5rem;">MODEL</div>
            <div style="font-size: 1.5rem; font-weight: bold; color: #333;">{best_model['model']}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'''
        <div class="metric-card">
            <div style="color: #667eea; font-size: 0.9rem; font-weight: bold; margin-bottom: 0.5rem;">MAE</div>
            <div style="font-size: 1.5rem; font-weight: bold; color: #333;">{best_model['mae']:.3f}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'''
        <div class="metric-card">
            <div style="color: #667eea; font-size: 0.9rem; font-weight: bold; margin-bottom: 0.5rem;">RMSE</div>
            <div style="font-size: 1.5rem; font-weight: bold; color: #333;">{best_model['rmse']:.3f}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col4:
        st.markdown(f'''
        <div class="metric-card">
            <div style="color: #667eea; font-size: 0.9rem; font-weight: bold; margin-bottom: 0.5rem;">R² SCORE</div>
            <div style="font-size: 1.5rem; font-weight: bold; color: #667eea;">{best_model['r2_score']:.3f}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # All models comparison chart
    fig = create_model_comparison_chart(comparison_df)
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed table
    st.markdown('<div class="section-header">📊 Detailed Metrics</div>', unsafe_allow_html=True)
    
    display_df = comparison_df.copy()
    display_df['Status'] = display_df['is_best'].apply(
        lambda x: '<span class="status-badge best">🏆 SELECTED</span>' if x else ''
    )
    
    # Format metrics for display
    display_df['MAE'] = display_df['mae'].apply(lambda x: f"{x:.4f}")
    display_df['RMSE'] = display_df['rmse'].apply(lambda x: f"{x:.4f}")
    display_df['R² Score'] = display_df['r2_score'].apply(lambda x: f"{x:.4f}")
    
    st.dataframe(
        display_df[['model', 'MAE', 'RMSE', 'R² Score']],
        use_container_width=True,
        hide_index=True
    )

def show_about():
    """Display about page with enhanced styling"""
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">ℹ️ About This Project</div>', unsafe_allow_html=True)
    
    st.markdown('''
    <div class="about-box">
        <h3>🎯 Project Overview</h3>
        <p>
            An advanced machine learning-powered Air Quality Index (AQI) forecasting system 
            designed to predict air quality for Multan, Pakistan over the next 3 days. 
            The system combines real-time data collection with state-of-the-art ML models 
            to provide accurate and timely air quality predictions.
        </p>
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown('''
    <div class="about-box">
        <h3>🔍 How It Works</h3>
        <ol>
            <li><b>Hourly Data Collection:</b> Continuously fetches air quality data from OpenWeather API</li>
            <li><b>Daily Model Training:</b> Trains 3 different ML models on accumulated historical data</li>
            <li><b>Model Selection:</b> Automatically selects the best-performing model based on R² score</li>
            <li><b>3-Day Prediction:</b> Generates accurate forecasts for the next 3 days</li>
            <li><b>Alert System:</b> Triggers alerts for Poor and Very Poor air quality conditions</li>
        </ol>
    </div>
    ''', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('''
        <div class="about-box">
            <h3>🤖 Machine Learning Models</h3>
            <ul>
                <li><b>Random Forest:</b> Ensemble method for robust predictions</li>
                <li><b>XGBoost:</b> Gradient boosting for high accuracy</li>
                <li><b>LightGBM:</b> Fast and efficient gradient boosting</li>
            </ul>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown('''
        <div class="about-box">
            <h3>📍 Location Information</h3>
            <p><b>City:</b> Multan, Pakistan</p>
            <p><b>Latitude:</b> 30.1979793</p>
            <p><b>Longitude:</b> 71.4724978</p>
            <p><b>Data Range:</b> Oct 2025 - Present</p>
            <p><b>Update Frequency:</b> Hourly</p>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown('''
    <div class="about-box">
        <h3>📊 AQI Categories</h3>
        <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 1rem; margin-top: 1rem;">
            <div style="text-align: center; padding: 1rem; background: #00e400; border-radius: 10px; color: white; font-weight: bold;">
                Good<br>(0-50)
            </div>
            <div style="text-align: center; padding: 1rem; background: #ffff00; border-radius: 10px; color: black; font-weight: bold;">
                Fair<br>(51-100)
            </div>
            <div style="text-align: center; padding: 1rem; background: #ff7e00; border-radius: 10px; color: white; font-weight: bold;">
                Moderate<br>(101-200)
            </div>
            <div style="text-align: center; padding: 1rem; background: #ff0000; border-radius: 10px; color: white; font-weight: bold;">
                Poor<br>(201-300)
            </div>
            <div style="text-align: center; padding: 1rem; background: #8f3f97; border-radius: 10px; color: white; font-weight: bold;">
                Very Poor<br>(301+)
            </div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

# ==================== MAIN APPLICATION ====================
def main():
    # Header
    st.markdown('<div class="main-header">🌫️ AQI Forecast Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">3-Day Air Quality Predictions for Multan, Pakistan</div>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner("⏳ Loading predictions..."):
        predictions_df, comparison_df = load_predictions()
    
    # Last updated
    pkt = pytz.timezone(TIMEZONE)
    current_time = datetime.now(pkt)
    st.markdown(
        f'<div class="last-updated">🕐 Last Updated: {current_time.strftime("%A, %B %d, %Y at %I:%M %p PKT")}</div>',
        unsafe_allow_html=True
    )
    
    # Sidebar navigation
    st.sidebar.markdown('''
    <div style="text-align: center; color: white; margin-bottom: 2rem;">
        <h1 style="font-size: 1.5rem; margin: 0;">🌍 Navigation</h1>
    </div>
    ''', unsafe_allow_html=True)
    
    page = st.sidebar.radio(
        "Select a page:",
        ["📊 3-Day Forecast", "🤖 Model Comparison", "ℹ️ About"],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown('''
    <div style="color: white; font-size: 0.9rem; text-align: center; margin-top: 2rem;">
        <p><b>System Status:</b> ✅ Active</p>
        <p><b>Data Source:</b> OpenWeather API</p>
        <p><b>Update Cycle:</b> Hourly</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Page routing
    if "📊" in page:
        show_forecast(predictions_df, comparison_df)
    elif "🤖" in page:
        show_model_comparison(comparison_df)
    else:
        show_about()

if __name__ == "__main__":
    main()