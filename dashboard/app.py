import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import pytz
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Add project root to system path
sys.path.append(str(Path(__file__).parent.parent))
try:
    from src.config.config import (
        PROCESSED_DATA_DIR,
        TIMEZONE,
        LAT,
        LON,
        PREDICTION_HORIZON_DAYS,
        MODEL_NAMES
    )
except ImportError:
    PROCESSED_DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
    TIMEZONE = "Asia/Karachi"
    LAT = 30.1979793
    LON = 71.4724978
    PREDICTION_HORIZON_DAYS = 3
    MODEL_NAMES = ["random_forest", "gradient_boosting", "lightgbm", "decision_tree", "sklearn_gradient_boosting"]

# Configure Streamlit page
st.set_page_config(
    page_title="Multan Air Quality Intelligence",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Session State & Theme Preference
# ---------------------------------------------------------
if "theme_preference" not in st.session_state:
    st.session_state.theme_preference = "System"

# ---------------------------------------------------------
# Dynamic CSS Design System (Light, Dark, & System)
# ---------------------------------------------------------
def generate_theme_css(theme_mode: str) -> str:
    """
    Generates CSS variable definitions based on the selected theme mode:
    Light, Dark, or System (via CSS media query).
    """
    light_tokens = """
        --bg-page: #f8fafc;
        --bg-surface: #ffffff;
        --bg-surface-elevated: #f1f5f9;
        --bg-surface-hover: #e2e8f0;
        --bg-sidebar: #f8fafc;
        --border-default: #94a3b8;
        --border-subtle: #cbd5e1;
        --border-focus: #0284c7;
        --text-primary: #090d16;
        --text-secondary: #1e293b;
        --text-muted: #334155;
        --accent-primary: #0284c7;
        --accent-subtle: #e0f2fe;
        --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.08), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -4px rgba(0, 0, 0, 0.05);
        
        --aqi-1-color: #047857;
        --aqi-1-bg: #ecfdf5;
        --aqi-1-border: #6ee7b7;
        --aqi-2-color: #4d7c0f;
        --aqi-2-bg: #f7fee7;
        --aqi-2-border: #a3e635;
        --aqi-3-color: #b45309;
        --aqi-3-bg: #fffbeb;
        --aqi-3-border: #fcd34d;
        --aqi-4-color: #b91c1c;
        --aqi-4-bg: #fef2f2;
        --aqi-4-border: #fca5a5;
        --aqi-5-color: #881337;
        --aqi-5-bg: #fff1f2;
        --aqi-5-border: #fda4af;
    """

    dark_tokens = """
        --bg-page: #0b0f19;
        --bg-surface: #111827;
        --bg-surface-elevated: #1e293b;
        --bg-surface-hover: #334155;
        --bg-sidebar: #0e1422;
        --border-default: #374151;
        --border-subtle: #1f2937;
        --border-focus: #38bdf8;
        --text-primary: #f9fafb;
        --text-secondary: #9ca3af;
        --text-muted: #6b7280;
        --accent-primary: #38bdf8;
        --accent-subtle: rgba(56, 189, 248, 0.12);
        --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.3);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.4);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.5);
        
        --aqi-1-color: #34d399;
        --aqi-1-bg: rgba(16, 185, 129, 0.15);
        --aqi-1-border: rgba(16, 185, 129, 0.3);
        --aqi-2-color: #a3e635;
        --aqi-2-bg: rgba(132, 204, 22, 0.15);
        --aqi-2-border: rgba(132, 204, 22, 0.3);
        --aqi-3-color: #fbbf24;
        --aqi-3-bg: rgba(245, 158, 11, 0.15);
        --aqi-3-border: rgba(245, 158, 11, 0.3);
        --aqi-4-color: #f87171;
        --aqi-4-bg: rgba(239, 68, 68, 0.15);
        --aqi-4-border: rgba(239, 68, 68, 0.3);
        --aqi-5-color: #fb7185;
        --aqi-5-bg: rgba(244, 63, 94, 0.18);
        --aqi-5-border: rgba(244, 63, 94, 0.35);
    """

    if theme_mode == "Light":
        root_css = f":root {{ {light_tokens} }}"
    elif theme_mode == "Dark":
        root_css = f":root {{ {dark_tokens} }}"
    else:  # System preference
        root_css = f"""
            :root {{
                {light_tokens}
            }}
            @media (prefers-color-scheme: dark) {{
                :root {{
                    {dark_tokens}
                }}
            }}
        """

    base_css = f"""
    <style>
        {root_css}

        /* Modern Typography & Smooth Rendering */
        html, body, [class*="css"] {{
            font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }}

        /* App Backgrounds & Surface styling */
        .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {{
            background-color: var(--bg-page) !important;
            color: var(--text-primary) !important;
        }}

        /* Eliminate excessive Streamlit bottom gap across all pages */
        .block-container, [data-testid="stMainBlockContainer"] {{
            padding-top: 2rem !important;
            padding-bottom: 1.25rem !important;
            max-width: 100% !important;
        }}

        /* Hide default Streamlit footer */
        footer {{
            display: none !important;
            visibility: hidden !important;
        }}

        [data-testid="stHeader"] {{
            background-color: transparent !important;
            z-index: 1000 !important;
        }}

        /* Sidebar Styling */
        [data-testid="stSidebar"] {{
            background-color: var(--bg-sidebar) !important;
            border-right: 1px solid var(--border-subtle) !important;
        }}

        [data-testid="stSidebar"] * {{
            color: var(--text-primary);
        }}

        /* High-contrast floating toggle for collapsed sidebar */
        div[data-testid="collapsedControl"],
        div[data-testid="stSidebarCollapsedControl"] {{
            position: fixed !important;
            top: 14px !important;
            left: 14px !important;
            z-index: 9999999 !important;
            display: block !important;
            visibility: visible !important;
        }}

        /* Style the actual button for both open and close states across all Streamlit versions */
        div[data-testid="collapsedControl"] button,
        div[data-testid="stSidebarCollapsedControl"] button,
        header[data-testid="stHeader"] button[kind="headerNoPadding"],
        header[data-testid="stHeader"] button[kind="header"],
        div[data-testid="stSidebarCollapseButton"] button,
        button[data-testid="stSidebarCollapseButton"] {{
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            visibility: visible !important;
            opacity: 1 !important;
            background-color: var(--bg-surface-elevated) !important;
            border: 2px solid var(--border-default) !important;
            border-radius: 8px !important;
            padding: 6px 8px !important;
            min-width: 36px !important;
            min-height: 36px !important;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.25) !important;
            cursor: pointer !important;
            transition: all 0.2s ease !important;
        }}

        /* Force high contrast on SVGs and icons inside */
        div[data-testid="collapsedControl"] svg,
        div[data-testid="collapsedControl"] button svg,
        div[data-testid="stSidebarCollapsedControl"] svg,
        div[data-testid="stSidebarCollapsedControl"] button svg,
        div[data-testid="stSidebarCollapseButton"] svg,
        div[data-testid="stSidebarCollapseButton"] button svg,
        header[data-testid="stHeader"] button svg,
        button[kind="headerNoPadding"] svg {{
            fill: var(--text-primary) !important;
            stroke: var(--text-primary) !important;
            color: var(--text-primary) !important;
            width: 20px !important;
            height: 20px !important;
            display: inline-block !important;
            visibility: visible !important;
            opacity: 1 !important;
        }}

        div[data-testid="collapsedControl"] button:hover,
        div[data-testid="stSidebarCollapseButton"] button:hover,
        header[data-testid="stHeader"] button[kind="headerNoPadding"]:hover {{
            background-color: var(--bg-surface-hover) !important;
            border-color: var(--accent-primary) !important;
            box-shadow: 0 6px 14px rgba(0, 0, 0, 0.35) !important;
        }}

        div[data-testid="collapsedControl"] button:hover svg,
        div[data-testid="stSidebarCollapseButton"] button:hover svg,
        header[data-testid="stHeader"] button[kind="headerNoPadding"]:hover svg {{
            fill: var(--accent-primary) !important;
            stroke: var(--accent-primary) !important;
            color: var(--accent-primary) !important;
        }}

        /* Clean Section Headers */
        .page-header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            flex-wrap: wrap;
            gap: 1rem;
            padding-bottom: 1.25rem;
            margin-bottom: 1.75rem;
            border-bottom: 1px solid var(--border-subtle);
        }}

        .page-title {{
            font-size: 1.65rem;
            font-weight: 700;
            color: var(--text-primary);
            letter-spacing: -0.025em;
            margin: 0;
            line-height: 1.25;
        }}

        .page-subtitle {{
            font-size: 0.88rem;
            color: var(--text-secondary);
            margin-top: 0.25rem;
        }}

        .header-meta {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
            flex-wrap: wrap;
        }}

        /* Badges & Chips */
        .badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.25rem 0.65rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.02em;
            white-space: nowrap;
        }}

        .badge-live {{
            background-color: rgba(16, 185, 129, 0.12);
            color: #10b981;
            border: 1px solid rgba(16, 185, 129, 0.25);
        }}

        .badge-live::before {{
            content: "";
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background-color: #10b981;
            display: inline-block;
        }}

        .badge-neutral {{
            background-color: var(--bg-surface-elevated);
            color: var(--text-secondary);
            border: 1px solid var(--border-subtle);
        }}

        .badge-brand {{
            background-color: var(--accent-subtle);
            color: var(--accent-primary);
            border: 1px solid var(--border-subtle);
        }}

        /* KPI & Metric Cards */
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1rem;
            margin-bottom: 1.75rem;
        }}

        .kpi-card {{
            background-color: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: 12px;
            padding: 1.25rem;
            box-shadow: var(--shadow-sm);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }}

        .kpi-card:hover {{
            box-shadow: var(--shadow-md);
            border-color: var(--border-default);
        }}

        .kpi-label {{
            font-size: 0.78rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 0.5rem;
        }}

        .kpi-value {{
            font-size: 1.85rem;
            font-weight: 700;
            color: var(--text-primary);
            letter-spacing: -0.03em;
            line-height: 1.1;
        }}

        .kpi-subtext {{
            font-size: 0.8rem;
            color: var(--text-secondary);
            margin-top: 0.4rem;
        }}

        /* Forecast Card Grid */
        .forecast-card {{
            background-color: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: 14px;
            padding: 1.5rem;
            box-shadow: var(--shadow-sm);
            transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }}

        .forecast-card:hover {{
            transform: translateY(-2px);
            box-shadow: var(--shadow-lg);
            border-color: var(--border-default);
        }}

        .forecast-card-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 0.75rem;
            border-bottom: 1px solid var(--border-subtle);
            margin-bottom: 1rem;
        }}

        .forecast-day-name {{
            font-size: 1.15rem;
            font-weight: 700;
            color: var(--text-primary);
        }}

        .forecast-date {{
            font-size: 0.82rem;
            color: var(--text-muted);
        }}

        .aqi-pill-hero {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.85rem 1rem;
            border-radius: 10px;
            margin: 0.75rem 0 1rem 0;
            border-width: 1px;
            border-style: solid;
        }}

        .aqi-pill-val {{
            font-size: 2.25rem;
            font-weight: 800;
            line-height: 1;
            letter-spacing: -0.04em;
        }}

        .aqi-pill-meta {{
            text-align: right;
        }}

        .aqi-pill-cat {{
            font-size: 0.95rem;
            font-weight: 700;
        }}

        .aqi-pill-range {{
            font-size: 0.78rem;
            opacity: 0.85;
            margin-top: 0.2rem;
        }}

        /* Custom Panels & Content Containers */
        .panel-container {{
            background-color: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: 14px;
            padding: 1.5rem;
            box-shadow: var(--shadow-sm);
            margin-bottom: 1.5rem;
        }}

        .panel-title {{
            font-size: 1.05rem;
            font-weight: 700;
            color: var(--text-primary);
            letter-spacing: -0.015em;
            margin-bottom: 0.35rem;
        }}

        .panel-desc {{
            font-size: 0.84rem;
            color: var(--text-secondary);
            margin-bottom: 1.25rem;
        }}

        /* Advisory Grid */
        .advisory-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }}

        .advisory-item {{
            background-color: var(--bg-surface-elevated);
            border: 1px solid var(--border-subtle);
            border-radius: 10px;
            padding: 1rem 1.15rem;
        }}

        .advisory-item-title {{
            font-size: 0.88rem;
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: 0.35rem;
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }}

        .advisory-item-desc {{
            font-size: 0.81rem;
            color: var(--text-secondary);
            line-height: 1.5;
        }}

        /* Alert Warning Banner */
        .alert-banner {{
            display: flex;
            align-items: flex-start;
            gap: 0.85rem;
            background-color: var(--aqi-5-bg);
            border: 1px solid var(--aqi-5-border);
            color: var(--aqi-5-color);
            padding: 1rem 1.25rem;
            border-radius: 12px;
            margin-bottom: 1.5rem;
            font-size: 0.88rem;
            line-height: 1.5;
        }}

        .alert-banner-title {{
            font-weight: 700;
            margin-bottom: 0.2rem;
            font-size: 0.92rem;
        }}

        /* Architecture & Info Grid */
        .bento-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1rem;
        }}

        .bento-card {{
            background-color: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: 12px;
            padding: 1.35rem;
            box-shadow: var(--shadow-sm);
        }}

        .bento-step-num {{
            font-size: 0.72rem;
            font-weight: 700;
            color: var(--accent-primary);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.25rem;
        }}

        .bento-title {{
            font-size: 1rem;
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: 0.5rem;
        }}

        .bento-desc {{
            font-size: 0.83rem;
            color: var(--text-secondary);
            line-height: 1.6;
        }}

        /* Sidebar Metadata & Controls */
        .sidebar-section-title {{
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--text-muted);
            margin: 1.5rem 0 0.5rem 0;
            padding-left: 0.25rem;
        }}

        .sidebar-brand {{
            padding: 0.5rem 0.25rem 1rem 0.25rem;
            border-bottom: 1px solid var(--border-subtle);
            margin-bottom: 1rem;
        }}

        .sidebar-brand-title {{
            font-size: 1.05rem;
            font-weight: 700;
            color: var(--text-primary);
            letter-spacing: -0.02em;
        }}

        .sidebar-brand-subtitle {{
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-top: 0.15rem;
        }}

        .sidebar-meta-box {{
            background-color: var(--bg-surface-elevated);
            border: 1px solid var(--border-subtle);
            border-radius: 10px;
            padding: 0.85rem;
            margin-top: 1.5rem;
            font-size: 0.78rem;
            color: var(--text-secondary);
            line-height: 1.6;
        }}

        /* Subtle modern footer with tight bottom gap */
        .footer-bar {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 0.75rem;
            padding: 0.85rem 0 0.25rem 0;
            margin-top: 1.25rem;
            border-top: 1px solid var(--border-subtle);
            color: var(--text-muted);
            font-size: 0.78rem;
        }}

        /* Override Streamlit Widget Defaults for Professional Look */
        [data-testid="stSidebar"] [data-testid="stRadio"] label {{
            font-size: 0.84rem !important;
            font-weight: 500 !important;
            cursor: pointer;
        }}

        [data-testid="stRadio"] label {{
            font-size: 0.88rem !important;
            font-weight: 500 !important;
        }}
        
        div[data-testid="stDataFrame"] {{
            border: 1px solid var(--border-subtle);
            border-radius: 10px;
            overflow: hidden;
        }}
    </style>
    """
    return base_css

# Apply dynamic CSS
st.markdown(generate_theme_css(st.session_state.theme_preference), unsafe_allow_html=True)

# ---------------------------------------------------------
# Semantic AQI Configuration (Refined Design Tokens)
# ---------------------------------------------------------
AQI_THEME_TOKENS = {
    1: {
        "name": "Good",
        "badge_text": "Safe Air",
        "color_var": "var(--aqi-1-color)",
        "bg_var": "var(--aqi-1-bg)",
        "border_var": "var(--aqi-1-border)",
        "advisory": "Air quality is satisfactory; poses little or no risk."
    },
    2: {
        "name": "Fair",
        "badge_text": "Acceptable",
        "color_var": "var(--aqi-2-color)",
        "bg_var": "var(--aqi-2-bg)",
        "border_var": "var(--aqi-2-border)",
        "advisory": "Air quality is acceptable for most; mild concern for unusually sensitive individuals."
    },
    3: {
        "name": "Moderate",
        "badge_text": "Moderate Caution",
        "color_var": "var(--aqi-3-color)",
        "bg_var": "var(--aqi-3-bg)",
        "border_var": "var(--aqi-3-border)",
        "advisory": "Sensitive groups may experience minor symptoms; general public should moderate exertion."
    },
    4: {
        "name": "Poor",
        "badge_text": "Unhealthy",
        "color_var": "var(--aqi-4-color)",
        "bg_var": "var(--aqi-4-bg)",
        "border_var": "var(--aqi-4-border)",
        "advisory": "Health effects likely for sensitive individuals; general public may experience irritation."
    },
    5: {
        "name": "Very Poor",
        "badge_text": "Hazardous",
        "color_var": "var(--aqi-5-color)",
        "bg_var": "var(--aqi-5-bg)",
        "border_var": "var(--aqi-5-border)",
        "advisory": "Health emergency warning. Serious respiratory irritation risk for the entire population."
    }
}

# ---------------------------------------------------------
# Data Loader
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def load_predictions():
    predictions_path = PROCESSED_DATA_DIR / "latest_predictions.csv"
    comparison_path = PROCESSED_DATA_DIR / "model_comparison.csv"

    predictions_df = pd.DataFrame()
    comparison_df = pd.DataFrame()

    if predictions_path.exists():
        try:
            predictions_df = pd.read_csv(predictions_path)
        except Exception as e:
            st.error(f"Error reading predictions file: {e}")

    if comparison_path.exists():
        try:
            comparison_df = pd.read_csv(comparison_path)
        except Exception as e:
            st.error(f"Error reading model comparison file: {e}")

    return predictions_df, comparison_df

# ---------------------------------------------------------
# Plotly Chart Creators (Adaptive Theming)
# ---------------------------------------------------------
def create_trend_chart(predictions_df: pd.DataFrame, theme_mode: str) -> go.Figure:
    """
    Renders an interactive 72-hour AQI trend chart with clean engineering styling.
    """
    is_dark = theme_mode == "Dark"
    font_color = "#f9fafb" if is_dark else "#090d16"
    grid_color = "rgba(255, 255, 255, 0.1)" if is_dark else "#cbd5e1"
    line_color = "#38bdf8" if is_dark else "#0284c7"
    fill_color = "rgba(56, 189, 248, 0.15)" if is_dark else "rgba(2, 132, 199, 0.12)"

    fig = go.Figure()

    # Fill and primary curve
    fig.add_trace(go.Scatter(
        x=predictions_df['date'],
        y=predictions_df['average_aqi'],
        fill='tozeroy',
        fillcolor=fill_color,
        line=dict(color=line_color, width=3, shape='spline'),
        mode='lines+markers',
        marker=dict(size=8, color=line_color, line=dict(color="#ffffff", width=2)),
        hovertemplate='<b>%{x}</b><br>Predicted AQI: <b>%{y:.1f}</b><extra></extra>'
    ))

    fig.update_layout(
        margin=dict(l=20, r=20, t=30, b=20),
        height=340,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        hovermode='x unified',
        showlegend=False,
        font=dict(family='Inter, -apple-system, sans-serif', size=12, color=font_color),
        xaxis=dict(
            showgrid=True,
            gridcolor=grid_color,
            gridwidth=1,
            color=font_color,
            linecolor=grid_color,
            tickfont=dict(size=11)
        ),
        yaxis=dict(
            range=[0.8, 5.2],
            dtick=1,
            showgrid=True,
            gridcolor=grid_color,
            gridwidth=1,
            color=font_color,
            tickmode='array',
            tickvals=[1, 2, 3, 4, 5],
            ticktext=['1 - Good', '2 - Fair', '3 - Moderate', '4 - Poor', '5 - Very Poor'],
            tickfont=dict(size=11)
        )
    )

    return fig

def create_model_comparison_chart(comparison_df: pd.DataFrame, theme_mode: str) -> go.Figure:
    """
    Renders a comparative horizontal/vertical bar chart for model performance metrics.
    """
    if comparison_df.empty:
        return None

    is_dark = theme_mode == "Dark"
    font_color = "#f9fafb" if is_dark else "#090d16"
    grid_color = "rgba(255, 255, 255, 0.1)" if is_dark else "#cbd5e1"
    accent_best = "#38bdf8" if is_dark else "#0284c7"
    accent_others = "#475569" if is_dark else "#64748b"

    # Identify best model
    best_idx = comparison_df['f1_score'].idxmax()
    colors = [accent_best if i == best_idx else accent_others for i in range(len(comparison_df))]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=comparison_df['model'],
        y=comparison_df['f1_score'],
        marker=dict(color=colors, line=dict(color='rgba(0,0,0,0)', width=0), cornerradius=6),
        text=[f"{v:.4f}" for v in comparison_df['f1_score']],
        textposition='outside',
        textfont=dict(color=font_color, size=12, family="Inter, -apple-system, sans-serif"),
        hovertemplate='<b>%{x}</b><br>Macro F1: %{y:.4f}<extra></extra>'
    ))

    fig.update_layout(
        margin=dict(l=20, r=20, t=30, b=20),
        height=340,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter, -apple-system, sans-serif', size=12, color=font_color),
        showlegend=False,
        xaxis=dict(
            showgrid=False,
            color=font_color,
            tickfont=dict(size=11)
        ),
        yaxis=dict(
            title="Macro F1 Score",
            range=[min(comparison_df['f1_score']) * 0.995, 1.002],
            showgrid=True,
            gridcolor=grid_color,
            color=font_color,
            tickfont=dict(size=11)
        )
    )

    return fig

# ---------------------------------------------------------
# View 1: 🔮 3-Day Forecast View
# ---------------------------------------------------------
def render_forecast_page(predictions_df: pd.DataFrame, comparison_df: pd.DataFrame, theme_mode: str):
    if predictions_df.empty:
        st.markdown("""
        <div class="panel-container">
            <div class="panel-title">Data Pipeline Notice</div>
            <div class="panel-desc">
                Predictions have not been generated yet or the processed data directory is unpopulated.
                Trigger the training and recursive inference pipeline to populate <code>latest_predictions.csv</code>.
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Local Multan Date Context
    try:
        pkt = pytz.timezone(TIMEZONE)
    except Exception:
        pkt = pytz.timezone("Asia/Karachi")
    current_time = datetime.now(pkt)
    today_date = current_time.date()

    # Calculate average AQI across the 3 forecast days
    avg_aqi = float(predictions_df['average_aqi'].mean())
    avg_aqi_level = int(round(avg_aqi))
    avg_token = AQI_THEME_TOKENS.get(avg_aqi_level, AQI_THEME_TOKENS[1])

    first_day_row = predictions_df.iloc[0]
    first_day_aqi = int(first_day_row['average_aqi'])
    first_day_token = AQI_THEME_TOKENS.get(first_day_aqi, AQI_THEME_TOKENS[1])
    first_day_date = first_day_row['date']
    first_day_name = first_day_row['day_name']

    last_day_date = predictions_df.iloc[-1]['date']
    max_forecast_aqi = int(predictions_df['average_aqi'].max())
    max_token = AQI_THEME_TOKENS.get(max_forecast_aqi, AQI_THEME_TOKENS[1])
    has_severe_warning = max_forecast_aqi >= 4

    if has_severe_warning:
        st.markdown(f"""
        <div class="alert-banner">
            <div>
                <div class="alert-banner-title">⚠️ Severe Atmospheric Alert: Category {max_forecast_aqi} ({max_token['name']}) Expected Over Next 3 Days</div>
                Particulate concentrations across Multan are projected to remain hazardous from {first_day_date} through {last_day_date}.
                Sensitive groups (asthma, cardiovascular conditions, elderly, and children) must avoid outdoor activities. Certified N95/FFP2 respirators are strongly recommended when commuting.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Executive KPI Summary Grid
    best_model_name = "LightGBM"
    best_model_f1 = "N/A"
    if not comparison_df.empty:
        best_row = comparison_df.loc[comparison_df['f1_score'].idxmax()]
        best_model_name = str(best_row['model'])
        best_model_f1 = f"{best_row['f1_score']:.4f}"

    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-label">3-Day Projected Average AQI</div>
            <div class="kpi-value" style="color: {avg_token['color_var']};">{avg_aqi:.1f} · {avg_token['name']}</div>
            <div class="kpi-subtext">Multan average for next 3 days ({first_day_date} to {last_day_date})</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Day 1 Forecast ({first_day_name})</div>
            <div class="kpi-value" style="color: {first_day_token['color_var']};">Level {first_day_aqi} · {first_day_token['name']}</div>
            <div class="kpi-subtext">Next-day projection for {first_day_date}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">3-Day Maximum Peak Risk</div>
            <div class="kpi-value">Level {max_forecast_aqi} ({max_token['name']})</div>
            <div class="kpi-subtext">Highest projected level across horizon</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Production Machine Learning Model</div>
            <div class="kpi-value" style="font-size: 1.45rem;">{best_model_name}</div>
            <div class="kpi-subtext">Validation Macro F1: {best_model_f1}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 3-Day Forecast Cards Grid
    st.markdown(f'<div class="panel-title" style="margin-bottom: 0.85rem;">Daily Forecast Trajectory (Next 3 Days: {first_day_date} to {last_day_date})</div>', unsafe_allow_html=True)
    cols = st.columns(len(predictions_df), gap="medium")

    for idx, (_, row) in enumerate(predictions_df.iterrows()):
        aqi_val = int(row['average_aqi'])
        token = AQI_THEME_TOKENS.get(aqi_val, AQI_THEME_TOKENS[1])
        min_aqi = int(row['min_aqi'])
        max_aqi = int(row['max_aqi'])
        
        # Calculate dynamic relative label from current date
        pred_date_obj = pd.to_datetime(row['date']).date()
        diff = (pred_date_obj - today_date).days
        if diff == 1:
            rel_label = "Tomorrow"
        elif diff == 0:
            rel_label = "Today"
        elif diff > 1:
            rel_label = f"In {diff} Days"
        else:
            rel_label = f"Day {idx + 1}"
            
        badge_text = f"Day {idx + 1} · {rel_label}"

        with cols[idx]:
            st.markdown(f"""
            <div class="forecast-card">
                <div>
                    <div class="forecast-card-header">
                        <div>
                            <div class="forecast-day-name">{row['day_name']}</div>
                            <div class="forecast-date">{row['date']}</div>
                        </div>
                        <span class="badge badge-brand">{badge_text}</span>
                    </div>
                    <div class="aqi-pill-hero" style="background-color: {token['bg_var']}; border-color: {token['border_var']}; color: {token['color_var']};">
                        <div class="aqi-pill-val">{aqi_val}</div>
                        <div class="aqi-pill-meta">
                            <div class="aqi-pill-cat">{token['name']}</div>
                            <div class="aqi-pill-range">Range: {min_aqi} – {max_aqi}</div>
                        </div>
                    </div>
                    <div style="font-size: 0.83rem; color: var(--text-secondary); line-height: 1.5; margin-bottom: 0.75rem;">
                        {token['advisory']}
                    </div>
                </div>
                <div style="padding-top: 0.75rem; border-top: 1px solid var(--border-subtle); display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 0.75rem; color: var(--text-muted);">Health Impact</span>
                    <span class="badge" style="background-color: {token['bg_var']}; color: {token['color_var']}; border: 1px solid {token['border_var']};">
                        {token['badge_text']}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # 72-Hour Interactive Chart
    st.markdown('<div style="height: 1.5rem;"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="panel-container">
        <div class="panel-title">Air Quality Index Projection (Next 3 Days)</div>
        <div class="panel-desc">Multi-step recursive projection for Multan across the upcoming 72-hour period (1=Good to 5=Very Poor).</div>
    """, unsafe_allow_html=True)

    fig = create_trend_chart(predictions_df, theme_mode)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

    # Actionable Health & Safety Guidelines
    st.markdown("""
    <div class="panel-container">
        <div class="panel-title">Actionable Health & Exposure Guidelines</div>
        <div class="panel-desc">Evidence-based precautions calibrated to prevailing atmospheric particulate forecasts.</div>
        <div class="advisory-grid">
            <div class="advisory-item">
                <div class="advisory-item-title">🫁 Sensitive Groups</div>
                <div class="advisory-item-desc">Individuals with asthma, chronic bronchitis, or cardiovascular illness should minimize outdoor exposure and keep inhalers accessible.</div>
            </div>
            <div class="advisory-item">
                <div class="advisory-item-title">🏃 Physical Activity</div>
                <div class="advisory-item-desc">Shift aerobic exercise indoors. Avoid running, cycling, or heavy athletic training near traffic corridors or unpaved zones.</div>
            </div>
            <div class="advisory-item">
                <div class="advisory-item-title">😷 Protective Equipment</div>
                <div class="advisory-item-desc">Use certified N95 or KN95 respirators. Cloth and standard surgical masks provide minimal filtration against PM2.5 particulates.</div>
            </div>
            <div class="advisory-item">
                <div class="advisory-item-title">🏠 Indoor Ventilation</div>
                <div class="advisory-item-desc">Keep exterior windows closed during peak inversion hours (dawn and dusk). Utilize HEPA air purification where available.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# View 2: 📊 Model Performance & Benchmarks View
# ---------------------------------------------------------
def render_model_comparison_page(comparison_df: pd.DataFrame, theme_mode: str):
    if comparison_df.empty:
        st.markdown("""
        <div class="panel-container">
            <div class="panel-title">Evaluation Records Unavailable</div>
            <div class="panel-desc">Run model training and cross-validation to populate <code>model_comparison.csv</code>.</div>
        </div>
        """, unsafe_allow_html=True)
        return

    best_idx = comparison_df['f1_score'].idxmax()
    best_model = comparison_df.iloc[best_idx]

    # Champion Model Hero Metric Strip
    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card" style="border-left: 3px solid var(--accent-primary);">
            <div class="kpi-label">Production Champion</div>
            <div class="kpi-value" style="font-size: 1.5rem; color: var(--accent-primary);">{best_model['model']}</div>
            <div class="kpi-subtext">Selected for automated inference</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Macro F1 Score</div>
            <div class="kpi-value">{best_model['f1_score']:.4f}</div>
            <div class="kpi-subtext">Harmonic mean of precision & recall</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Overall Accuracy</div>
            <div class="kpi-value">{best_model['accuracy']:.4f}</div>
            <div class="kpi-subtext">Multi-class classification accuracy</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Precision & Recall</div>
            <div class="kpi-value" style="font-size: 1.4rem;">{best_model['precision']:.4f} / {best_model['recall']:.4f}</div>
            <div class="kpi-subtext">Class-balanced validation metrics</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Comparative Visualizer
    st.markdown("""
    <div class="panel-container">
        <div class="panel-title">Model Architecture Comparison (Macro F1)</div>
        <div class="panel-desc">Comparative evaluation across 5 classification pipelines evaluated on unseen validation holdouts.</div>
    """, unsafe_allow_html=True)

    fig = create_model_comparison_chart(comparison_df, theme_mode)
    if fig:
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

    # Detailed Leaderboard Table
    st.markdown("""
    <div class="panel-container">
        <div class="panel-title">Validation Metric Leaderboard</div>
        <div class="panel-desc">Rigorous multi-class evaluation scores rounded to 4 significant digits.</div>
    """, unsafe_allow_html=True)

    display_df = comparison_df[['model', 'f1_score', 'accuracy', 'precision', 'recall']].copy()
    display_df['Deployment Status'] = ["Champion" if i == best_idx else "Candidate" for i in range(len(display_df))]
    display_df.columns = ['Model Architecture', 'Macro F1', 'Accuracy', 'Precision', 'Recall', 'Deployment Status']

    for metric in ['Macro F1', 'Accuracy', 'Precision', 'Recall']:
        display_df[metric] = display_df[metric].apply(lambda x: f"{float(x):.4f}")

    table_rows = []
    for _, row in display_df.iterrows():
        is_champ = row['Deployment Status'] == "Champion"
        status_badge = (
            '<span class="badge" style="background-color: var(--accent-subtle); color: var(--accent-primary); border: 1.5px solid var(--border-focus); font-weight: 700;">Champion</span>'
            if is_champ else
            '<span class="badge badge-neutral">Candidate</span>'
        )
        row_bg = "background-color: var(--bg-surface-elevated);" if is_champ else ""
        row_weight = "font-weight: 700;" if is_champ else "font-weight: 500;"

        table_rows.append(f"""
            <tr style="border-bottom: 1px solid var(--border-subtle); {row_bg}">
                <td style="padding: 0.85rem 1rem; {row_weight} color: var(--text-primary);">{row['Model Architecture']}</td>
                <td style="padding: 0.85rem 1rem; color: var(--text-primary); font-family: monospace; font-size: 0.9rem;">{row['Macro F1']}</td>
                <td style="padding: 0.85rem 1rem; color: var(--text-primary); font-family: monospace; font-size: 0.9rem;">{row['Accuracy']}</td>
                <td style="padding: 0.85rem 1rem; color: var(--text-primary); font-family: monospace; font-size: 0.9rem;">{row['Precision']}</td>
                <td style="padding: 0.85rem 1rem; color: var(--text-primary); font-family: monospace; font-size: 0.9rem;">{row['Recall']}</td>
                <td style="padding: 0.85rem 1rem;">{status_badge}</td>
            </tr>
        """)

    table_html = f"""
    <div style="overflow-x: auto; border: 1px solid var(--border-subtle); border-radius: 10px; background-color: var(--bg-surface);">
        <table style="width: 100%; border-collapse: collapse; font-size: 0.86rem; text-align: left;">
            <thead>
                <tr style="border-bottom: 2px solid var(--border-subtle); background-color: var(--bg-surface-elevated); color: var(--text-primary);">
                    <th style="padding: 0.85rem 1rem; font-weight: 700;">Model Architecture</th>
                    <th style="padding: 0.85rem 1rem; font-weight: 700;">Macro F1</th>
                    <th style="padding: 0.85rem 1rem; font-weight: 700;">Accuracy</th>
                    <th style="padding: 0.85rem 1rem; font-weight: 700;">Precision</th>
                    <th style="padding: 0.85rem 1rem; font-weight: 700;">Recall</th>
                    <th style="padding: 0.85rem 1rem; font-weight: 700;">Deployment Status</th>
                </tr>
            </thead>
            <tbody>
                {''.join(table_rows)}
            </tbody>
        </table>
    </div>
    """
    st.markdown(table_html, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Engineering Insights Card
    st.markdown("""
    <div class="panel-container">
        <div class="panel-title">Engineering Observations & Model Mechanics</div>
        <div class="advisory-grid">
            <div class="advisory-item">
                <div class="advisory-item-title">⚡ Histogram & Gradient Boosting</div>
                <div class="advisory-item-desc">Gradient boosted tree architectures consistently outperform standalone decision trees by capturing non-linear interactions between weather factors and atmospheric stagnation.</div>
            </div>
            <div class="advisory-item">
                <div class="advisory-item-title">⚖️ Imbalance Mitigation</div>
                <div class="advisory-item-desc">Class weighting policies ensure that minority severity classes (such as Category 1 and 2 during smog season) maintain high sensitivity without sacrificing overall precision.</div>
            </div>
            <div class="advisory-item">
                <div class="advisory-item-title">🔄 Recursive Stability</div>
                <div class="advisory-item-desc">Recursive auto-regressive feature rollouts retain consistent predictive boundaries across the 72-hour forecast horizon without variance explosion.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# View 3: ℹ️ Architecture & System Info View
# ---------------------------------------------------------
def render_about_page():
    st.markdown("""
    <div class="panel-container">
        <div class="panel-title">System Architecture Overview</div>
        <div class="panel-desc">End-to-end automated machine learning pipeline deployed for continuous environmental intelligence.</div>
        <div class="bento-grid">
            <div class="bento-card">
                <div class="bento-step-num">Stage 01</div>
                <div class="bento-title">Hourly Ingestion</div>
                <div class="bento-desc">Scheduled fetch of 8 key atmospheric pollutants (PM2.5, PM10, NO2, SO2, CO, O3, NH3, NO) and local weather metrics via OpenWeather API.</div>
            </div>
            <div class="bento-card">
                <div class="bento-step-num">Stage 02</div>
                <div class="bento-title">Hopsworks Feature Store</div>
                <div class="bento-desc">Centralized temporal feature repository managing feature groups, cyclical time transformations, and lag features with point-in-time correctness.</div>
            </div>
            <div class="bento-card">
                <div class="bento-step-num">Stage 03</div>
                <div class="bento-title">Daily CI/CD Retraining</div>
                <div class="bento-desc">Automated training comparing 5 classification models on sliding historical data, selecting the top macro F1 model for production.</div>
            </div>
            <div class="bento-card">
                <div class="bento-step-num">Stage 04</div>
                <div class="bento-title">72h Recursive Inference</div>
                <div class="bento-desc">Forward-projecting auto-regressive inference yielding daily maximum, minimum, and average AQI categories with safety advisories.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Evaluated Models Catalog
    st.markdown("""
    <div class="panel-container">
        <div class="panel-title">Model Catalog & Algorithmic Design</div>
        <div class="panel-desc">Summary of evaluated classification algorithms and their specific capabilities.</div>
        <div class="bento-grid">
            <div class="bento-card">
                <div class="bento-title">LightGBM</div>
                <div class="bento-desc">Fast, distributed histogram-based gradient boosting optimized for speed and memory efficiency. Employs leaf-wise tree growth.</div>
            </div>
            <div class="bento-card">
                <div class="bento-title">Histogram Gradient Boosting</div>
                <div class="bento-desc">Scikit-learn implementation with native handling of missing features and integer-based binning for accelerated training.</div>
            </div>
            <div class="bento-card">
                <div class="bento-title">Random Forest Classifier</div>
                <div class="bento-desc">Bagging ensemble of decorrelated decision trees, providing robust variance reduction and resilience against isolated noisy sensor values.</div>
            </div>
            <div class="bento-card">
                <div class="bento-title">Decision Tree Baseline</div>
                <div class="bento-desc">Interpretable baseline benchmark demonstrating the performance improvement achieved by subsequent ensemble architectures.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Standard AQI Index Standards
    st.markdown("""
    <div class="panel-container">
        <div class="panel-title">Air Quality Index Classification Standards</div>
        <div class="panel-desc">Standardized European / OpenWeather 5-level scale corresponding to ambient particulate thresholds.</div>
        <div style="overflow-x: auto;">
            <table style="width: 100%; border-collapse: collapse; font-size: 0.85rem; color: var(--text-secondary); text-align: left;">
                <thead>
                    <tr style="border-bottom: 2px solid var(--border-subtle); color: var(--text-primary);">
                        <th style="padding: 0.75rem 0.5rem;">Index</th>
                        <th style="padding: 0.75rem 0.5rem;">Category</th>
                        <th style="padding: 0.75rem 0.5rem;">PM2.5 (μg/m³)</th>
                        <th style="padding: 0.75rem 0.5rem;">PM10 (μg/m³)</th>
                        <th style="padding: 0.75rem 0.5rem;">Health Implications</th>
                    </tr>
                </thead>
                <tbody>
                    <tr style="border-bottom: 1px solid var(--border-subtle);">
                        <td style="padding: 0.75rem 0.5rem; font-weight: 700; color: var(--aqi-1-color);">1</td>
                        <td style="padding: 0.75rem 0.5rem; font-weight: 600;">Good</td>
                        <td style="padding: 0.75rem 0.5rem;">0 – 10</td>
                        <td style="padding: 0.75rem 0.5rem;">0 – 20</td>
                        <td style="padding: 0.75rem 0.5rem;">Ideal air quality; minimal or no health hazard.</td>
                    </tr>
                    <tr style="border-bottom: 1px solid var(--border-subtle);">
                        <td style="padding: 0.75rem 0.5rem; font-weight: 700; color: var(--aqi-2-color);">2</td>
                        <td style="padding: 0.75rem 0.5rem; font-weight: 600;">Fair</td>
                        <td style="padding: 0.75rem 0.5rem;">10 – 25</td>
                        <td style="padding: 0.75rem 0.5rem;">20 – 50</td>
                        <td style="padding: 0.75rem 0.5rem;">Acceptable quality; mild irritation possible for highly sensitive people.</td>
                    </tr>
                    <tr style="border-bottom: 1px solid var(--border-subtle);">
                        <td style="padding: 0.75rem 0.5rem; font-weight: 700; color: var(--aqi-3-color);">3</td>
                        <td style="padding: 0.75rem 0.5rem; font-weight: 600;">Moderate</td>
                        <td style="padding: 0.75rem 0.5rem;">25 – 50</td>
                        <td style="padding: 0.75rem 0.5rem;">50 – 100</td>
                        <td style="padding: 0.75rem 0.5rem;">Respiratory symptoms emerge in sensitive populations.</td>
                    </tr>
                    <tr style="border-bottom: 1px solid var(--border-subtle);">
                        <td style="padding: 0.75rem 0.5rem; font-weight: 700; color: var(--aqi-4-color);">4</td>
                        <td style="padding: 0.75rem 0.5rem; font-weight: 600;">Poor</td>
                        <td style="padding: 0.75rem 0.5rem;">50 – 75</td>
                        <td style="padding: 0.75rem 0.5rem;">100 – 200</td>
                        <td style="padding: 0.75rem 0.5rem;">Adverse health symptoms likely for broader public; mask advised.</td>
                    </tr>
                    <tr>
                        <td style="padding: 0.75rem 0.5rem; font-weight: 700; color: var(--aqi-5-color);">5</td>
                        <td style="padding: 0.75rem 0.5rem; font-weight: 600;">Very Poor</td>
                        <td style="padding: 0.75rem 0.5rem;">> 75</td>
                        <td style="padding: 0.75rem 0.5rem;">> 200</td>
                        <td style="padding: 0.75rem 0.5rem;">Emergency conditions; significant risk for the entire population.</td>
                    </tr>
                </tbody>
            </table>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Main Application Flow
# ---------------------------------------------------------
def main():
    # Load dataset
    predictions_df, comparison_df = load_predictions()

    # Timezone & Current Timestamp
    try:
        pkt = pytz.timezone(TIMEZONE)
    except Exception:
        pkt = pytz.timezone("Asia/Karachi")
    current_time = datetime.now(pkt)
    formatted_time = current_time.strftime("%A, %b %d · %I:%M %p PKT")

    # Sidebar Header & Brand
    st.sidebar.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-brand-title">Multan AQI Monitor</div>
        <div class="sidebar-brand-subtitle">Predictive Environmental Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar Navigation
    st.sidebar.markdown('<div class="sidebar-section-title">Navigation</div>', unsafe_allow_html=True)
    page_options = [
        "🔮 3-Day Forecast",
        "📊 Model Benchmarks",
        "ℹ️ System Architecture"
    ]
    selected_page = st.sidebar.radio(
        "Navigation",
        options=page_options,
        label_visibility="collapsed"
    )

    # Sidebar Theme Preferences (Light / Dark / System)
    st.sidebar.markdown('<div class="sidebar-section-title">Interface Theme</div>', unsafe_allow_html=True)
    theme_options = ["System", "Light", "Dark"]
    theme_icons = {
        "System": "🖥️ System",
        "Light": "☀️ Light",
        "Dark": "🌙 Dark"
    }

    selected_theme = st.sidebar.radio(
        "Theme Preference",
        options=theme_options,
        format_func=lambda x: theme_icons.get(x, x),
        index=theme_options.index(st.session_state.theme_preference),
        horizontal=True,
        label_visibility="collapsed",
        key="theme_radio_input"
    )

    # Handle theme change trigger
    if selected_theme != st.session_state.theme_preference:
        st.session_state.theme_preference = selected_theme
        st.rerun()

    # Station Metadata
    st.sidebar.markdown('<div class="sidebar-section-title">Station Info</div>', unsafe_allow_html=True)
    st.sidebar.markdown(f"""
    <div class="sidebar-meta-box">
        <div><strong>Monitoring Site:</strong> Multan City</div>
        <div><strong>Coordinates:</strong> {LAT:.2f}° N, {LON:.2f}° E</div>
        <div><strong>Feature Store:</strong> Hopsworks v1</div>
        <div><strong>Pipeline Cycle:</strong> Daily 11:00 AM PKT</div>
        <div><strong>Status:</strong> <span style="color: #10b981; font-weight: 600;">● Active</span></div>
    </div>
    """, unsafe_allow_html=True)

    # Persistent Clean Top Header
    st.markdown(f"""
    <div class="page-header">
        <div>
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.35rem;">
                <span class="badge badge-brand">Multan, Pakistan</span>
                <span class="badge badge-live">Operational Feed</span>
            </div>
            <h1 class="page-title">Air Quality Intelligence Dashboard</h1>
            <div class="page-subtitle">Predicting the average Air Quality Index (AQI) of Multan for the next 3 days</div>
        </div>
        <div class="header-meta">
            <span class="badge badge-neutral">📅 {formatted_time}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Page Routing
    if "3-Day Forecast" in selected_page:
        render_forecast_page(predictions_df, comparison_df, st.session_state.theme_preference)
    elif "Model Benchmarks" in selected_page:
        render_model_comparison_page(comparison_df, st.session_state.theme_preference)
    else:
        render_about_page()

    # Refined Enterprise Footer
    st.markdown("""
    <div class="footer-bar">
        <div>AQI Forecasting System · Production Release v2.0</div>
        <div>Multan Environmental Observation Project · Crafted with Precision</div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
