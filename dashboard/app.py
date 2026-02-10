import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import pytz
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

sys.path.append(str(Path(__file__).parent.parent))
from src.prediction.predictor import AQIPredictor
from src.config.config import *

st.set_page_config(
    page_title="AQI Forecast - Multan Pakistan",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
    }
    
    /* Main background */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 100%);
    }
    
    [data-testid="stMain"] {
        background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 100%);
    }
    
    /* header bar */
    header[data-testid="stHeader"] {
        background-color: #00ffff !important;   /* Black header */
    }

    /* Sidebar toggle + menu icons */
    header[data-testid="stHeader"] button {
        color: #ffffff !important;              /* White icons */
    }

    /* "Deploy", menu text, etc. */
    header[data-testid="stHeader"] * {
        color: #ffffff !important;              /* White text */
    }
    
    /* Sidebar styling with animation */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1b2a 0%, #1b263b 100%);
        animation: slideIn 0.5s ease-out;
    }
    
    @keyframes slideIn {
        from {
            transform: translateX(-100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        padding-top: 1rem;
    }
    
    /* Text colors for sidebar */
    [data-testid="stSidebar"] .css-1dp5vir {
        color: #ffffff !important;
    }
    
    /* sidebar text white */
    section[data-testid="stSidebar"] * {
        color: #ffffff !important;
    }

    /* Sidebar radio labels (3-Day Forecast, etc.) */
    section[data-testid="stSidebar"] [role="radiogroup"] label {
        color: #ffffff !important;
        font-weight: 500;
    }

    /* Hover effect for sidebar options */
    section[data-testid="stSidebar"] [role="radio"]:hover {
        background: rgba(255, 255, 255, 0.08) !important;
        border-radius: 10px;
    }

    /* Selected radio button indicator */
    section[data-testid="stSidebar"] [role="radio"][aria-checked="true"] div:first-child {
        background-color: #00d4ff !important;
        border-color: #00d4ff !important;
    }

    /* Sidebar open/close arrow icons */
    button[title="Close sidebar"] svg,
    button[title="Open sidebar"] svg {
        fill: #ffffff !important;
        color: #ffffff !important;
    }

    
    /* Header styling */
    .main-title {
        font-size: 3.5rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
        animation: fadeInDown 0.8s ease-out;
    }
    
    .sub-title {
        text-align: center;
        font-size: 1.4rem;
        color: #80d4ff;
        margin-bottom: 2rem;
        font-weight: 500;
        animation: fadeInUp 0.8s ease-out 0.2s backwards;
    }
    
    .tagline {
        text-align: center;
        font-size: 1rem;
        color: #555;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* DateTime display */
    .datetime-display {
        text-align: center;
        font-size: 1.1rem;
        color: #00d4ff;
        margin-bottom: 2rem;
        padding: 1rem;
        border: 2px solid #00d4ff;
        border-radius: 12px;
        background: rgba(0, 212, 255, 0.05);
        font-weight: 600;
        font-family: 'Courier New', monospace;
        animation: fadeInUp 0.8s ease-out 0.4s backwards;
    }
    
    /* Forecast card styling */
    .forecast-card {
        background: linear-gradient(135deg, #1a3a4a 0%, #0f2a3a 100%);
        border: 2px solid #00d4ff;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 15px 50px rgba(0, 212, 255, 0.15);
        transition: all 0.3s ease;
        animation: slideInCard 0.6s ease-out backwards;
    }
    
    .forecast-card:nth-child(1) {
        animation-delay: 0.2s;
    }
    
    .forecast-card:nth-child(2) {
        animation-delay: 0.4s;
    }
    
    .forecast-card:nth-child(3) {
        animation-delay: 0.6s;
    }
    
    @keyframes slideInCard {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .forecast-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 25px 80px rgba(0, 212, 255, 0.25);
        border-color: #00ffff;
    }
    
    .day-name {
        font-size: 1.8rem;
        font-weight: bold;
        color: #00d4ff;
        margin-bottom: 0.5rem;
    }
    
    .date-text {
        color: #888;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    
    .aqi-value {
        font-size: 4.5rem;
        font-weight: 900;
        color: #00ffff;
        margin: 1rem 0;
        text-shadow: 0 0 20px rgba(0, 212, 255, 0.5);
    }
    
    .aqi-category {
        font-size: 1.5rem;
        font-weight: bold;
        color: #00d4ff;
        margin-bottom: 1rem;
    }
    
    .aqi-range {
        color: #aaa;
        font-size: 1.1rem;
        margin-bottom: 1rem;
    }
    
    /* Status badge */
    .status-good {
        background: linear-gradient(135deg, rgba(0, 255, 136, 0.2) 0%, rgba(0, 200, 100, 0.2) 100%);
        border: 2px solid #00ff88;
        color: #00ff88;
        padding: 0.75rem 1.5rem;
        border-radius: 25px;
        font-weight: bold;
        font-size: 1rem;
    }
    
    .status-warning {
        background: linear-gradient(135deg, rgba(255, 165, 0, 0.2) 0%, rgba(255, 100, 0, 0.2) 100%);
        border: 2px solid #ff9500;
        color: #ff9500;
        padding: 0.75rem 1.5rem;
        border-radius: 25px;
        font-weight: bold;
        font-size: 1rem;
    }
    
    .status-critical {
        background: linear-gradient(135deg, rgba(255, 50, 50, 0.2) 0%, rgba(200, 0, 0, 0.2) 100%);
        border: 2px solid #ff3232;
        color: #ff3232;
        padding: 0.75rem 1.5rem;
        border-radius: 25px;
        font-weight: bold;
        font-size: 1rem;
    }
    
    /* Section header */
    .section-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #00d4ff;
        margin: 2rem 0 1.5rem 0;
        border-bottom: 3px solid #00d4ff;
        padding-bottom: 1rem;
    }
    
    /* Divider */
    .divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #00d4ff, #00ffff, transparent);
        margin: 2.5rem 0;
        border: none;
    }
    
    /* Model comparison card */
    .model-card {
        background: linear-gradient(135deg, #1a3a4a 0%, #0f2a3a 100%);
        border: 2px solid #00d4ff;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0, 212, 255, 0.15);
        transition: all 0.3s ease;
    }
    
    .model-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 50px rgba(0, 212, 255, 0.25);
        border-color: #00ffff;
    }
    
    .model-label {
        color: #888;
        font-size: 0.9rem;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }
    
    .model-value {
        font-size: 2rem;
        font-weight: bold;
        color: #00d4ff;
    }
    
    /* About section */
    .about-section {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.05) 0%, rgba(0, 100, 150, 0.05) 100%);
        border: 2px solid rgba(0, 212, 255, 0.3);
        border-radius: 15px;
        padding: 2rem;
        margin: 1.5rem 0;
        color: #ddd;
    }
    
    .about-title {
        font-size: 1.8rem;
        font-weight: bold;
        color: #00d4ff;
        margin-bottom: 1rem;
    }
    
    .about-text {
        line-height: 1.8;
        color: #bbb;
    }
    
    .model-info {
        background: linear-gradient(135deg, #1a3a4a 0%, #0f2a3a 100%);
        border-left: 4px solid #00d4ff;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: #ddd;
    }
    
    .model-name {
        font-size: 1.3rem;
        font-weight: bold;
        color: #00d4ff;
        margin-bottom: 0.5rem;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 2rem;
        color: #00d4ff;
        font-size: 1rem;
        border-top: 2px solid rgba(0, 212, 255, 0.2);
        font-weight: 500;
    }
    
    /* Chart styling */
    .chart-container {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.05) 0%, rgba(0, 100, 150, 0.05) 100%);
        border: 2px solid rgba(0, 212, 255, 0.2);
        border-radius: 15px;
        padding: 2rem;
        margin: 2rem 0;
    }
    
    /* Metric boxes */
    .metric-box {
        background: linear-gradient(135deg, #1a3a4a 0%, #0f2a3a 100%);
        border: 2px solid #00d4ff;
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        color: #00d4ff;
        font-weight: bold;
    }
    
    .metric-label {
        color: #888;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 2.2rem;
        color: #00ffff;
    }
    
    /* Sidebar menu title */
    .sidebar-menu-title {
        color: #00d4ff;
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid rgba(0, 212, 255, 0.3);
    }
    
    /* Navigation radio buttons */
    [data-testid="stSidebar"] [role="radiogroup"] {
        gap: 1rem;
    }
    
    [data-testid="stSidebar"] [role="radio"] {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(0, 100, 150, 0.1) 100%);
        border: 2px solid rgba(0, 212, 255, 0.3);
        border-radius: 12px;
        padding: 1rem;
        transition: all 0.3s ease;
    }
    
    [data-testid="stSidebar"] [role="radio"]:hover {
        border-color: #00d4ff;
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.2) 0%, rgba(0, 100, 150, 0.2) 100%);
    }
</style>
""", unsafe_allow_html=True)

# ==================== COLOR MAPPING ====================
AQI_CATEGORIES = {
    'Good': {
        'color': '#00ff88',
        'bg': 'rgba(0, 255, 136, 0.15)',
        'border': '#00ff88'
    },
    'Fair': {
        'color': '#ffff00',
        'bg': 'rgba(255, 255, 0, 0.15)',
        'border': '#ffff00'
    },
    'Moderate': {
        'color': '#ff9500',
        'bg': 'rgba(255, 149, 0, 0.15)',
        'border': '#ff9500'
    },
    'Poor': {
        'color': '#ff3232',
        'bg': 'rgba(255, 50, 50, 0.15)',
        'border': '#ff3232'
    },
    'Very Poor': {
        'color': '#ff0080',
        'bg': 'rgba(255, 0, 128, 0.15)',
        'border': '#ff0080'
    }
}

# ==================== CACHE & DATA LOADING ====================
@st.cache_data(ttl=3600)
def load_predictions():
    predictions_path = PROCESSED_DATA_DIR / 'latest_predictions.csv'
    comparison_path = PROCESSED_DATA_DIR / 'model_comparison.csv'
    
    if predictions_path.exists():
        return pd.read_csv(predictions_path).fillna(""), pd.read_csv(comparison_path).fillna("")
    else:
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
def get_status_class(category):
    """Get status badge class based on category"""
    if category == 'Good':
        return 'status-good'
    elif category in ['Fair']:
        return 'status-warning'
    else:
        return 'status-critical'

def create_trend_chart(predictions_df):
    """Create an interactive trend chart with proper AQI scaling (0-5)"""
    fig = go.Figure()
    
    # Add filled area
    fig.add_trace(go.Scatter(
        x=predictions_df['date'],
        y=predictions_df['average_aqi'],
        fill='tozeroy',
        fillcolor='rgba(0, 212, 255, 0.2)',
        line=dict(color='#00ffff', width=4),
        mode='lines',
        hovertemplate='<b>%{x}</b><br>AQI: %{y:.2f}<extra></extra>'
    ))
    
    # Add markers
    fig.add_trace(go.Scatter(
        x=predictions_df['date'],
        y=predictions_df['average_aqi'],
        mode='markers',
        marker=dict(
            size=14,
            color='#00d4ff',
            symbol='circle',
            line=dict(color='#00ffff', width=3)
        ),
        hovertemplate='<b>%{x}</b><br>AQI: %{y:.2f}<extra></extra>',
        showlegend=False
    ))
    
    fig.update_layout(
        title=dict(
            text='Air Quality Trend',
            font=dict(size=18, color='#00d4ff', family='Arial Black')
        ),
        xaxis=dict(
            title='Date',
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(0, 212, 255, 0.1)',
            color='#888'
        ),
        yaxis=dict(
            title='AQI Index',
            range=[0, 5],
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(0, 212, 255, 0.1)',
            color='#888'
        ),
        hovermode='x unified',
        height=450,
        template='plotly_dark',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        plot_bgcolor='rgba(15, 42, 58, 0.5)',
        font=dict(family='Arial', size=12, color='#ddd'),
        margin=dict(l=50, r=50, t=80, b=50),
        showlegend=False
    )
    
    return fig

def create_model_comparison_chart(comparison_df):
    """Create an appealing model comparison chart"""
    fig = go.Figure()
    
    colors = ['#00ffff' if x else '#00d4ff' for x in comparison_df['is_best']]
    
    fig.add_trace(go.Bar(
        x=comparison_df['model'],
        y=comparison_df['r2_score'],
        marker=dict(
            color=colors,
            line=dict(color='white', width=2)
        ),
        text=[f"{v:.4f}" for v in comparison_df['r2_score']],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>R² Score: %{y:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text='Model Performance Comparison',
            font=dict(size=18, color='#00d4ff', family='Arial Black')
        ),
        xaxis=dict(title='Model', color='#888'),
        yaxis=dict(
            title='R² Score',
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(0, 212, 255, 0.1)',
            color='#888'
        ),
        height=450,
        template='plotly_dark',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        plot_bgcolor='rgba(15, 42, 58, 0.5)',
        font=dict(family='Arial', size=12, color='#ddd'),
        margin=dict(l=50, r=50, t=80, b=50),
        showlegend=False
    )
    
    return fig

# ==================== PAGE FUNCTIONS ====================
def show_forecast(predictions_df, comparison_df):
    """Display 3-day forecast page"""
    
    # Section header
    st.markdown('<div class="section-header">Next 3 Days Forecast</div>', unsafe_allow_html=True)
    
    best_model = comparison_df.loc[comparison_df['r2_score'].idxmax()]
    st.info(f"Predictions powered by **{best_model['model']}** (R² Score: {best_model['r2_score']:.4f})")
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Display 3-day forecast cards
    cols = st.columns(3, gap="large")
    
    for idx, (_, row) in enumerate(predictions_df.iterrows()):
        category = row['category']
        color_info = AQI_CATEGORIES.get(category, AQI_CATEGORIES['Good'])
        status_class = get_status_class(category)
        
        with cols[idx]:
            st.markdown(f"""
            <div class="forecast-card">
                <div class="day-name">{row['day_name']}</div>
                <div class="date-text">{row['date']}</div>
                <div class="aqi-value" style="color: {color_info['color']};">{row['average_aqi']:.2f}</div>
                <div class="aqi-category" style="color: {color_info['color']};">{category}</div>
                <div class="aqi-range">Range: {row['min_aqi']:.2f} - {row['max_aqi']:.2f}</div>
                <div style="margin-top: 1rem;">
                    <div class="{status_class}">
                        {'SAFE' if category == 'Good' else 'ALERT'}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if pd.notna(row['warning']):
                st.warning(row['warning'])
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Trend chart
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    fig = create_trend_chart(predictions_df)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

def show_model_comparison(comparison_df):
    """Display model comparison page"""
    
    st.markdown('<div class="section-header">Model Performance Analysis</div>', unsafe_allow_html=True)
    
    # Best model highlight
    best_model = comparison_df[comparison_df['is_best'] == True].iloc[0]
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<h3 style="color: #00d4ff; text-align: center; margin: 2rem 0 1rem 0;">Current Best Model</h3>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4, gap="large")
    
    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Model</div>
            <div class="metric-value" style="font-size: 1.5rem;">{best_model['model']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">MAE</div>
            <div class="metric-value">{best_model['mae']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">RMSE</div>
            <div class="metric-value">{best_model['rmse']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">R² Score</div>
            <div class="metric-value">{best_model['r2_score']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Model comparison chart
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    fig = create_model_comparison_chart(comparison_df)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Detailed metrics table
    st.markdown('<h3 style="color: #00d4ff; margin: 2rem 0 1rem 0;">All Models Metrics</h3>', unsafe_allow_html=True)
    
    display_df = comparison_df[['model', 'mae', 'rmse', 'r2_score']].copy()
    display_df.columns = ['Model', 'MAE', 'RMSE', 'R² Score']
    display_df['MAE'] = display_df['MAE'].apply(lambda x: f"{x:.4f}")
    display_df['RMSE'] = display_df['RMSE'].apply(lambda x: f"{x:.4f}")
    display_df['R² Score'] = display_df['R² Score'].apply(lambda x: f"{x:.4f}")
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)

def show_about():
    """Display about page"""
    
    st.markdown('<div class="section-header">About This Project</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="about-section">
        <div class="about-title">Project Overview</div>
        <div class="about-text">
            This AQI Forecasting System is an advanced machine learning solution designed to predict 
            air quality conditions for the next 3 days in Multan, Pakistan. The system continuously 
            collects real-time air quality data, processes it through multiple machine learning models, 
            and selects the best-performing model to deliver accurate forecasts to help citizens make 
            informed decisions about outdoor activities and health precautions.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    st.markdown('<h3 style="color: #00d4ff; margin: 2rem 0 1rem 0;">How It Works</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="about-section">
            <div style="color: #00d4ff; font-weight: bold; margin-bottom: 1rem;">Data Collection & Processing</div>
            <ul style="color: #bbb; line-height: 2;">
                <li>Hourly data ingestion from OpenWeather API</li>
                <li>Real-time data validation and cleaning</li>
                <li>Feature engineering and storage in Hopsworks</li>
                <li>Historical data spanning Oct 2025 to present</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="about-section">
            <div style="color: #00d4ff; font-weight: bold; margin-bottom: 1rem;">Model Training & Prediction</div>
            <ul style="color: #bbb; line-height: 2;">
                <li>Daily model retraining on all historical data</li>
                <li>Automatic best model selection via R² score</li>
                <li>3-day AQI forecasting with confidence ranges</li>
                <li>Alert system for Poor and Very Poor conditions</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    st.markdown('<h3 style="color: #00d4ff; margin: 2rem 0 1rem 0;">Machine Learning Models</h3>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="model-info">
        <div class="model-name">Random Forest</div>
        <p style="color: #aaa;">
            An ensemble learning method that constructs multiple decision trees and merges their predictions.
            Random Forest is known for its robustness, ability to handle non-linear relationships, and resistance
            to overfitting. It provides reliable baseline predictions and feature importance rankings.
        </p>
    </div>
    
    <div class="model-info">
        <div class="model-name">XGBoost</div>
        <p style="color: #aaa;">
            Extreme Gradient Boosting is a scalable and optimized gradient boosting framework that builds
            sequential decision trees to correct previous errors. XGBoost excels at capturing complex patterns
            in data and typically provides high-accuracy predictions. It's widely recognized as a top-performing
            model in machine learning competitions.
        </p>
    </div>
    
    <div class="model-info">
        <div class="model-name">LightGBM</div>
        <p style="color: #aaa;">
            Light Gradient Boosting Machine is a fast, distributed gradient boosting framework. LightGBM uses
            leaf-wise tree growth strategy and supports categorical features natively. It's optimized for speed
            and memory efficiency while maintaining high accuracy, making it ideal for real-time applications
            like AQI forecasting.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    st.markdown('<h3 style="color: #00d4ff; margin: 2rem 0 1rem 0;">Location & Data</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="about-section">
            <div style="color: #00d4ff; font-weight: bold;">Multan, Pakistan</div>
            <ul style="color: #bbb; margin-top: 1rem;">
                <li>Latitude: 30.1979793</li>
                <li>Longitude: 71.4724978</li>
                <li>Data Source: OpenWeather API</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="about-section">
            <div style="color: #00d4ff; font-weight: bold;">System Information</div>
            <ul style="color: #bbb; margin-top: 1rem;">
                <li>Data Range: Oct 2025 - Present</li>
                <li>Update Frequency: Hourly</li>
                <li>Feature Storage: Hopsworks</li>
                <li>Update Frequency: Real-time</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ==================== MAIN APPLICATION ====================
def main():
    # Header
    st.markdown('<div class="main-title">AQI Forecasting System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Next 3-days Air Quality Index Prediction for Multan</div>', unsafe_allow_html=True)
    st.markdown('<div class="tagline">Stay informed. Stay safe.</div>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading predictions..."):
        predictions_df, comparison_df = load_predictions()
    
    # Current date and time
    pkt = pytz.timezone(TIMEZONE)
    current_time = datetime.now(pkt)
    st.markdown(f'<div class="datetime-display">{current_time.strftime("%A, %B %d, %Y | %I:%M %p PKT")}</div>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <div class="sidebar-menu-title">MENU</div>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.sidebar.radio(
        "Navigate to:",
        ["3-Day Forecast", "Model Comparison", "About"],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("<div style='margin-top: 3rem;'></div>", unsafe_allow_html=True)
    st.sidebar.markdown("""
    <div style="color: #00d4ff; font-size: 0.85rem; text-align: center; margin-top: 2rem; padding-top: 1rem; border-top: 1px solid rgba(0, 212, 255, 0.2);">
        <p style="margin: 0.5rem 0;">System Status: Active</p>
        <p style="margin: 0.5rem 0;">Data Source: OpenWeather API</p>
        <p style="margin: 0.5rem 0;">Update: Every Hour</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Page routing
    if page == "3-Day Forecast":
        show_forecast(predictions_df, comparison_df)
    elif page == "Model Comparison":
        show_model_comparison(comparison_df)
    else:
        show_about()
    
    # Footer
    st.markdown('<div class="footer">Created by Saqib Ahmad Siddiqui</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()