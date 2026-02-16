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
from src.config.config import *

st.set_page_config(
    page_title="AQI Forecast - Multan Pakistan",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1b2a 0%, #1b263b 100%);
    }
    
    [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        padding-top: 1rem;
    }
    
    [data-testid="stSidebar"] {
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] span {
        color: #ffffff !important;
    }
    
    /* Main title */
    .main-title {
        font-size: 3.5rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }
    
    .sub-title {
        text-align: center;
        font-size: 1.4rem;
        color: #80d4ff;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    
    .tagline {
        text-align: center;
        font-size: 1rem;
        color: #555;
        margin-bottom: 2rem;
    }
    
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
    }
    
    .forecast-card {
        background: linear-gradient(135deg, #1a3a4a 0%, #0f2a3a 100%);
        border: 2px solid #00d4ff;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 15px 50px rgba(0, 212, 255, 0.15);
        transition: all 0.3s ease;
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
    
    .section-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #00d4ff;
        margin: 2rem 0 1.5rem 0;
        border-bottom: 3px solid #00d4ff;
        padding-bottom: 1rem;
    }
    
    .divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #00d4ff, #00ffff, transparent);
        margin: 2.5rem 0;
        border: none;
    }
    
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
    
    .chart-container {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.05) 0%, rgba(0, 100, 150, 0.05) 100%);
        border: 2px solid rgba(0, 212, 255, 0.2);
        border-radius: 15px;
        padding: 2rem;
        margin: 2rem 0;
    }
    
    .status-good {
        background: rgba(0, 255, 136, 0.15);
        border: 2px solid #00ff88;
        color: #00ff88;
        padding: 0.75rem 1.5rem;
        border-radius: 25px;
        font-weight: bold;
    }
    
    .status-warning {
        background: rgba(255, 165, 0, 0.15);
        border: 2px solid #ff9500;
        color: #ff9500;
        padding: 0.75rem 1.5rem;
        border-radius: 25px;
        font-weight: bold;
    }
    
    .status-critical {
        background: rgba(255, 50, 50, 0.15);
        border: 2px solid #ff3232;
        color: #ff3232;
        padding: 0.75rem 1.5rem;
        border-radius: 25px;
        font-weight: bold;
    }
    
    .footer {
        text-align: center;
        padding: 2rem;
        color: #00d4ff;
        font-size: 1rem;
        border-top: 2px solid rgba(0, 212, 255, 0.2);
        font-weight: 500;
        margin-top: 3rem;
    }
    
    .sidebar-menu-title {
        color: #00d4ff;
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid rgba(0, 212, 255, 0.3);
    }
</style>
""", unsafe_allow_html=True)

AQI_COLORS = {
    1: {'name': '🟢 Good', 'color': '#00ff88', 'bg': 'rgba(0, 255, 136, 0.15)'},
    2: {'name': '🟡 Fair', 'color': '#ffff00', 'bg': 'rgba(255, 255, 0, 0.15)'},
    3: {'name': '🟠 Moderate', 'color': '#ff9500', 'bg': 'rgba(255, 149, 0, 0.15)'},
    4: {'name': '🔴 Poor', 'color': '#ff3232', 'bg': 'rgba(255, 50, 50, 0.15)'},
    5: {'name': '🔴 Very Poor', 'color': '#ff0080', 'bg': 'rgba(255, 0, 128, 0.15)'}
}

@st.cache_data(ttl=3600)
def load_predictions():
    predictions_path = PROCESSED_DATA_DIR / 'latest_predictions.csv'
    comparison_path = PROCESSED_DATA_DIR / 'model_comparison.csv'
    
    if predictions_path.exists() and comparison_path.exists():
        try:
            predictions_df = pd.read_csv(predictions_path)
            comparison_df = pd.read_csv(comparison_path)
            return predictions_df, comparison_df
        except Exception as e:
            st.error(f"Error loading CSV files: {e}")
            return pd.DataFrame(), pd.DataFrame()
    else:
        st.warning("⏳ Predictions not yet generated. Please run the training pipeline first.")
        return pd.DataFrame(), pd.DataFrame()

def get_status_class(aqi_value):
    if aqi_value <= 1:
        return 'status-good'
    elif aqi_value <= 2:
        return 'status-warning'
    else:
        return 'status-critical'

def create_trend_chart(predictions_df):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=predictions_df['date'],
        y=predictions_df['average_aqi'],
        fill='tozeroy',
        fillcolor='rgba(0, 212, 255, 0.2)',
        line=dict(color='#00ffff', width=4),
        mode='lines+markers',
        marker=dict(size=12, color='#00d4ff'),
        hovertemplate='<b>%{x}</b><br>AQI: %{y:.1f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text='Air Quality Trend (1=Good, 5=Very Poor)',
            font=dict(size=18, color='#00d4ff')
        ),
        xaxis=dict(
            title='Date',
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(0, 212, 255, 0.1)',
            color='#888'
        ),
        yaxis=dict(
            title='AQI Category (1-5)',
            range=[0.5, 5.5],
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(0, 212, 255, 0.1)',
            color='#888',
            dtick=1
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
    if comparison_df.empty:
        return None
    
    fig = go.Figure()
    
    colors = ['#00ffff' if x else '#00d4ff' for x in comparison_df.get('is_best', [False]*len(comparison_df))]
    
    fig.add_trace(go.Bar(
        x=comparison_df['model'],
        y=comparison_df['f1_score'],
        marker=dict(
            color=colors,
            line=dict(color='white', width=2)
        ),
        text=[f"{v:.4f}" for v in comparison_df['f1_score']],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>F1 Score: %{y:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text='Model Performance Comparison (F1 Score)',
            font=dict(size=18, color='#00d4ff')
        ),
        xaxis=dict(title='Model', color='#888'),
        yaxis=dict(
            title='F1 Score',
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

def show_forecast(predictions_df, comparison_df):
    
    if predictions_df.empty:
        st.error("❌ No predictions available. Please run the training pipeline.")
        return
    
    st.markdown('<div class="section-header">🔮 Next 3 Days Forecast</div>', unsafe_allow_html=True)
    
    if not comparison_df.empty:
        best_model = comparison_df.loc[comparison_df['f1_score'].idxmax()]
        st.info(f"🤖 Best Model: **{best_model['model']}** (F1 Score: {best_model['f1_score']:.4f})")
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    cols = st.columns(3, gap="large")
    
    for idx, (_, row) in enumerate(predictions_df.iterrows()):
        aqi_value = int(row['average_aqi'])
        color_info = AQI_COLORS.get(aqi_value, AQI_COLORS[1])
        status_class = get_status_class(aqi_value)
        
        with cols[idx]:
            st.markdown(f"""
            <div class="forecast-card">
                <div class="day-name">{row['day_name']}</div>
                <div class="date-text">{row['date']}</div>
                <div class="aqi-value" style="color: {color_info['color']};">{aqi_value}</div>
                <div class="aqi-category" style="color: {color_info['color']};">{color_info['name']}</div>
                <div class="aqi-range">Range: {int(row['min_aqi'])} - {int(row['max_aqi'])}</div>
                <div style="margin-top: 1rem;">
                    <div class="{status_class}">
                        {'✓ SAFE' if aqi_value <= 2 else '⚠️ CAUTION' if aqi_value == 3 else '⛔ WARNING'}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if pd.notna(row.get('warning')) and row.get('warning', '') != '':
                st.warning(row['warning'])
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    fig = create_trend_chart(predictions_df)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

def show_model_comparison(comparison_df):
    if comparison_df.empty:
        st.error("❌ No model data available.")
        return
    
    st.markdown('<div class="section-header">📊 Model Performance Analysis</div>', unsafe_allow_html=True)
    
    best_idx = comparison_df['f1_score'].idxmax()
    best_model = comparison_df.iloc[best_idx]
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<h3 style="color: #00d4ff; text-align: center; margin: 2rem 0 1rem 0;">🏆 Best Performing Model</h3>', unsafe_allow_html=True)
    
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
            <div class="metric-label">F1 Score</div>
            <div class="metric-value">{best_model['f1_score']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Accuracy</div>
            <div class="metric-value">{best_model['accuracy']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Precision</div>
            <div class="metric-value">{best_model['precision']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    fig = create_model_comparison_chart(comparison_df)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    st.markdown('<h3 style="color: #00d4ff; margin: 2rem 0 1rem 0;">📋 All Models Metrics</h3>', unsafe_allow_html=True)
    
    display_df = comparison_df[['model', 'accuracy', 'f1_score', 'precision', 'recall']].copy()
    display_df.columns = ['Model', 'Accuracy', 'F1 Score', 'Precision', 'Recall']
    
    for col in ['Accuracy', 'F1 Score', 'Precision', 'Recall']:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)

def show_about():
    st.markdown('<div class="section-header">ℹ️ About This Project</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="about-section">
        <div class="about-title">🌍 Project Overview</div>
        <div class="about-text">
            This AQI Forecasting System is an advanced machine learning solution designed to predict 
            air quality conditions for the next 3 days in Multan, Pakistan. The system uses recursive 
            forecasting with 5 classification models to generate accurate predictions that help citizens 
            make informed decisions about outdoor activities and health precautions.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    st.markdown('<h3 style="color: #00d4ff; margin: 2rem 0 1rem 0;">🔄 How It Works</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="about-section">
            <div style="color: #00d4ff; font-weight: bold; margin-bottom: 1rem;">📊 Data Pipeline</div>
            <ul style="color: #bbb; line-height: 2;">
                <li>✓ Hourly OpenWeather API data</li>
                <li>✓ Real-time feature engineering</li>
                <li>✓ Hopsworks feature store</li>
                <li>✓ Historical backfilling</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="about-section">
            <div style="color: #00d4ff; font-weight: bold; margin-bottom: 1rem;">🤖 ML Pipeline</div>
            <ul style="color: #bbb; line-height: 2;">
                <li>✓ Daily model retraining</li>
                <li>✓ 5 classification models</li>
                <li>✓ 72-hour recursive forecast</li>
                <li>✓ Auto model selection</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    st.markdown('<h3 style="color: #00d4ff; margin: 2rem 0 1rem 0;">🧠 Classification Models</h3>', unsafe_allow_html=True)
    
    models_info = [
        ("🌲 Random Forest", "Ensemble of decision trees with balanced weights"),
        ("📈 Histogram Gradient Boosting", "Fast gradient boosting with built-in NaN handling"),
        ("⚡ LightGBM", "Distributed gradient boosting, optimized for speed"),
        ("🌳 Decision Tree", "Simple interpretable tree-based classifier"),
        ("🚀 Sklearn Gradient Boosting", "Scikit-learn's gradient boosting with NaN imputation")
    ]
    
    for model_name, description in models_info:
        st.markdown(f"""
        <div class="model-info">
            <div class="model-name">{model_name}</div>
            <p style="color: #aaa;">{description}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    st.markdown('<h3 style="color: #00d4ff; margin: 2rem 0 1rem 0;">📍 AQI Categories</h3>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="about-section">
        <div style="color: #bbb; line-height: 2.5; font-size: 1.05rem;">
            <p><span style="color: #00ff88;">●</span> <strong>1 - Good:</strong> Satisfactory air quality</p>
            <p><span style="color: #ffff00;">●</span> <strong>2 - Fair:</strong> Acceptable air quality</p>
            <p><span style="color: #ff9500;">●</span> <strong>3 - Moderate:</strong> Some pollutant concerns</p>
            <p><span style="color: #ff3232;">●</span> <strong>4 - Poor:</strong> Health effects possible</p>
            <p><span style="color: #ff0080;">●</span> <strong>5 - Very Poor:</strong> Health alert condition</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<div class="main-title">🌫️ AQI Forecasting System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Next 3-days Air Quality Index Prediction for Multan</div>', unsafe_allow_html=True)
    st.markdown('<div class="tagline">Stay informed. Stay safe.</div>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner("⏳ Loading predictions..."):
        predictions_df, comparison_df = load_predictions()
    
    pkt = pytz.timezone(TIMEZONE)
    current_time = datetime.now(pkt)
    st.markdown(f'<div class="datetime-display">📅 {current_time.strftime("%A, %B %d, %Y | %I:%M %p PKT")}</div>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <div class="sidebar-menu-title">📋 MENU</div>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.sidebar.radio(
        "Navigate to:",
        ["🔮 3-Day Forecast", "📊 Model Comparison", "ℹ️ About"],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("""
    <div style="margin-top: 3rem; color: #00d4ff; font-size: 0.85rem; text-align: center; padding-top: 1rem; border-top: 1px solid rgba(0, 212, 255, 0.2);">
        <p style="margin: 0.5rem 0;">✅ Status: Active</p>
        <p style="margin: 0.5rem 0;">📊 Type: Classification</p>
        <p style="margin: 0.5rem 0;">📡 Source: OpenWeather API</p>
        <p style="margin: 0.5rem 0;">⏰ Update: Daily 05:00 AM PST</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Page routing
    if "3-Day Forecast" in page:
        show_forecast(predictions_df, comparison_df)
    elif "Model Comparison" in page:
        show_model_comparison(comparison_df)
    else:
        show_about()
    
    # Footer
    st.markdown('<div style="height: 50px;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="footer">Created by Saqib Ahmad Siddiqui | AQI Forecasting System v1.0</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
