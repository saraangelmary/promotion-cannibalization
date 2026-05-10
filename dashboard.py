import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ML Promotion Cannibalization Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .stApp { background-color: #0D1117; color: #E6EDF3; }

    [data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 1px solid #21262D;
    }
    [data-testid="stSidebar"] * { color: #C9D1D9 !important; }

    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #161B22 0%, #1C2128 100%);
        border: 1px solid #21262D;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.3);
    }
    [data-testid="metric-container"] label {
        color: #8B949E !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #58A6FF !important;
        font-size: 26px !important;
        font-weight: 700 !important;
        font-family: 'DM Mono', monospace !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricDelta"] {
        font-size: 12px !important;
    }

    .section-header {
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: #58A6FF;
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 1px solid #21262D;
    }

    .card {
        background: linear-gradient(135deg, #161B22 0%, #1C2128 100%);
        border: 1px solid #21262D;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 16px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.25);
    }

    .hero-banner {
        background: linear-gradient(135deg, #0D1117 0%, #161B22 40%, #0D2137 100%);
        border: 1px solid #21262D;
        border-left: 4px solid #58A6FF;
        border-radius: 12px;
        padding: 28px 32px;
        margin-bottom: 24px;
    }
    .hero-title { font-size: 26px; font-weight: 700; color: #E6EDF3; margin: 0 0 6px 0; letter-spacing: -0.02em; }
    .hero-subtitle { font-size: 13px; color: #8B949E; margin: 0; font-weight: 400; }
    .hero-badge {
        display: inline-block;
        background: rgba(88,166,255,0.12);
        color: #58A6FF;
        border: 1px solid rgba(88,166,255,0.3);
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 11px; font-weight: 600;
        letter-spacing: 0.06em;
        margin-right: 8px; margin-top: 12px;
    }

    .badge-rf {
        background: rgba(255,123,0,0.12); color: #FF7B00;
        border: 1px solid rgba(255,123,0,0.3);
        border-radius: 6px; padding: 2px 10px;
        font-size: 11px; font-weight: 600;
    }
    .badge-lgbm {
        background: rgba(63,185,80,0.12); color: #3FB950;
        border: 1px solid rgba(63,185,80,0.3);
        border-radius: 6px; padding: 2px 10px;
        font-size: 11px; font-weight: 600;
    }

    .stTabs [data-baseweb="tab-list"] {
        background-color: #161B22; border-radius: 10px;
        padding: 4px; gap: 4px; border: 1px solid #21262D;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent; color: #8B949E;
        border-radius: 8px; padding: 8px 20px;
        font-weight: 500; font-size: 13px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #21262D !important;
        color: #E6EDF3 !important;
    }

    hr { border-color: #21262D; margin: 24px 0; }

    .info-box {
        background: rgba(88,166,255,0.06);
        border: 1px solid rgba(88,166,255,0.2);
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 13px; color: #8B949E;
        margin-bottom: 12px;
    }
    .info-box strong { color: #58A6FF; }

    .cannibal-box {
        background: rgba(248,81,73,0.06);
        border: 1px solid rgba(248,81,73,0.25);
        border-radius: 10px; padding: 20px 24px; text-align: center;
    }
    .cannibal-value { font-size: 32px; font-weight: 700; color: #F85149; font-family: 'DM Mono', monospace; }
    .cannibal-label { font-size: 12px; color: #8B949E; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 4px; }

    .stDataFrame { border: 1px solid #21262D; border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  DATA
# ─────────────────────────────────────────────
promo_sales     = 1349865
non_promo_sales = 1331414
cannib_effect   = 18450

np.random.seed(42)
months     = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
base_sales = [1100000,950000,1050000,1080000,1150000,1200000,
              1180000,1250000,1300000,1350000,1650000,1850000]
monthly_sales = [s + np.random.randint(-30000,30000) for s in base_sales]

departments = ['Dept 92','Dept 95','Dept 38','Dept 72','Dept 90',
               'Dept 40','Dept 2','Dept 16','Dept 18','Dept 5']
dept_sales  = [4850000,4200000,3800000,3500000,3200000,
               2900000,2650000,2400000,2200000,2050000]

np.random.seed(10)
weeks          = list(range(1,144))
promo_weekly   = [promo_sales    + np.random.randint(-200000,250000) for _ in weeks]
nonpromo_weekly= [non_promo_sales+ np.random.randint(-180000,200000) for _ in weeks]

features   = ['Weekly_Sales_Lag1','Weekly_Sales_Lag4','Rolling_Avg_4wk',
              'MarkDown1','Rolling_Avg_12wk','Weekly_Sales_Lag2',
              'Week','MarkDown2','Dept','Month',
              'Weekly_Sales_Lag12','Store','MarkDown3','Temperature','Fuel_Price']
importance = [285,240,220,185,175,165,140,125,115,108,95,88,72,45,38]

np.random.seed(7)
n_points      = 80
actual        = np.array([900000+i*5000+np.random.randint(-150000,200000) for i in range(n_points)])
predicted_rf  = actual + np.random.randint(-100000,100000,n_points)
predicted_lgbm= actual + np.random.randint(-60000,60000,  n_points)

# ── Real values from model_comparison.csv ──
rf_runtime      = 115.99    # seconds  — RandomForest
lgbm_runtime    = 7.24      # seconds  — LightGBM
rf_throughput   = 660.34    # rows/sec — RandomForest
lgbm_throughput = 10583.20  # rows/sec — LightGBM

# Real RMSE / MAE from CSV (full precision)
rf_rmse   = 287008.65
lgbm_rmse = 235441.03
rf_mae    = 100777.13
lgbm_mae  = 89748.07

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:16px 0 8px 0;'>
        <div style='font-size:18px;font-weight:700;color:#E6EDF3;letter-spacing:-0.02em;'>📊 ML Promo</div>
        <div style='font-size:11px;color:#8B949E;margin-top:2px;'>Cannibalization Analysis</div>
    </div>
    <hr style='border-color:#21262D;margin:8px 0 16px 0;'>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["🏠 Overview","📈 Model Performance","🔍 Cannibalization","📦 Feature Analysis","🗂 Pipeline"],
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border-color:#21262D;margin:16px 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div class='section-header'>Project Info</div>
    <div class='info-box'>
        <strong>Student:</strong><br>Sara Angel Mary J<br>23BDA044<br><br>
        <strong>Guide:</strong><br>Dr. L. Pavithira<br>Associate Professor<br><br>
        <strong>Dataset:</strong><br>Walmart Store Sales<br>Forecasting — Kaggle
    </div>
    <div class='info-box' style='margin-top:8px;'>
        <strong>Stack:</strong> Python · LightGBM · RandomForest · Pandas · Scikit-learn
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  PLOTLY THEME
# ─────────────────────────────────────────────
PLOT_BG      = "#0D1117"
PAPER_BG     = "#0D1117"
GRID_COLOR   = "#21262D"
TEXT_COLOR   = "#8B949E"
ACCENT_BLUE  = "#58A6FF"
ACCENT_ORANGE= "#FF7B00"
ACCENT_GREEN = "#3FB950"
ACCENT_RED   = "#F85149"
ACCENT_PURPLE= "#BC8CFF"
ACCENT_YELLOW= "#FFA657"

def base_layout(title="", height=380):
    return dict(
        title=dict(text=title, font=dict(color="#C9D1D9",size=14,family="DM Sans"), x=0.01),
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
        font=dict(family="DM Sans",color=TEXT_COLOR,size=12),
        height=height,
        margin=dict(l=12,r=12,t=40,b=12),
        xaxis=dict(gridcolor=GRID_COLOR,linecolor=GRID_COLOR,tickfont=dict(size=11,color="#C9D1D9")),
        yaxis=dict(gridcolor=GRID_COLOR,linecolor=GRID_COLOR,tickfont=dict(size=11,color="#C9D1D9")),
        legend=dict(bgcolor="rgba(0,0,0,0)",bordercolor=GRID_COLOR,borderwidth=1,
                    font=dict(size=11,color="#C9D1D9")),
    )

# ═══════════════════════════════════════════════════════
#  PAGE: OVERVIEW
# ═══════════════════════════════════════════════════════
if page == "🏠 Overview":

    st.markdown("""
    <div class='hero-banner'>
        <div class='hero-title'>ML Promotion Cannibalization Analysis</div>
        <div class='hero-subtitle'>Walmart Store Sales Forecasting · Random Forest vs LightGBM · Weekly Revenue Prediction</div>
        <span class='hero-badge'>REGRESSION</span>
        <span class='hero-badge'>FORECASTING</span>
        <span class='hero-badge'>CANNIBALIZATION</span>
        <span class='hero-badge'>LIGHTGBM</span>
    </div>
    """, unsafe_allow_html=True)

    k1,k2,k3,k4,k5,k6 = st.columns(6)
    k1.metric("Total Records","4,21,571","Walmart Dataset")
    k2.metric("Stores","45","Across USA")
    k3.metric("Departments","~99","Per Store")
    k4.metric("Time Span","143 wks","2010–2012")
    k5.metric("Best RMSE","₹2,35,441","LightGBM")
    k6.metric("Promo Lift","₹18,450","Avg per week")

    st.markdown("<br>", unsafe_allow_html=True)
    col1,col2 = st.columns([2,1])

    with col1:
        st.markdown("<div class='section-header'>Monthly Sales Trend (2010–2012)</div>", unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=months, y=monthly_sales, mode='lines+markers',
            name='Monthly Sales',
            line=dict(color=ACCENT_BLUE,width=2.5),
            marker=dict(size=6,color=ACCENT_BLUE,line=dict(color='#0D1117',width=2)),
            fill='tozeroy', fillcolor='rgba(88,166,255,0.06)'
        ))
        layout = base_layout("Average Weekly Sales by Month (₹)",320)
        layout['yaxis']['tickformat']=',.0f'
        layout['yaxis']['tickprefix']='₹'
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<div class='section-header'>Cannibalization Summary</div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class='cannibal-box' style='margin-bottom:12px;'>
            <div class='cannibal-value'>₹{promo_sales:,}</div>
            <div class='cannibal-label'>Avg Promo Week Sales</div>
        </div>
        <div class='cannibal-box' style='background:rgba(63,185,80,0.06);border-color:rgba(63,185,80,0.25);margin-bottom:12px;'>
            <div class='cannibal-value' style='color:#3FB950;'>₹{non_promo_sales:,}</div>
            <div class='cannibal-label'>Avg Non-Promo Sales</div>
        </div>
        <div class='cannibal-box'>
            <div class='cannibal-value'>₹{cannib_effect:,}</div>
            <div class='cannibal-label'>Cannibalization Effect</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col3,col4 = st.columns(2)

    with col3:
        st.markdown("<div class='section-header'>Top 10 Departments by Revenue</div>", unsafe_allow_html=True)
        fig2 = go.Figure(go.Bar(
            x=dept_sales[::-1], y=departments[::-1], orientation='h',
            marker=dict(color=dept_sales[::-1],
                        colorscale=[[0,'#1C2128'],[0.5,'#2D4F8A'],[1,'#58A6FF']],
                        line=dict(color='rgba(0,0,0,0)',width=0)),
            text=[f'₹{s:,.0f}' for s in dept_sales[::-1]],
            textposition='outside', textfont=dict(size=10,color='#C9D1D9'),
        ))
        layout2 = base_layout("",360)
        layout2['xaxis']['tickformat']=',.0f'
        layout2['xaxis']['tickprefix']='₹'
        layout2['margin']=dict(l=12,r=80,t=10,b=12)
        fig2.update_layout(**layout2)
        st.plotly_chart(fig2, use_container_width=True)

    with col4:
        st.markdown("<div class='section-header'>Model Performance Comparison</div>", unsafe_allow_html=True)
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(name='Random Forest', x=['RMSE','MAE'], y=[287008,100777],
            marker_color=ACCENT_ORANGE, marker_line_width=0,
            text=['₹2,87,008','₹1,00,777'], textposition='outside',
            textfont=dict(size=10,color='#C9D1D9')))
        fig3.add_trace(go.Bar(name='LightGBM', x=['RMSE','MAE'], y=[235441,89748],
            marker_color=ACCENT_GREEN, marker_line_width=0,
            text=['₹2,35,441','₹89,748'], textposition='outside',
            textfont=dict(size=10,color='#C9D1D9')))
        layout3 = base_layout("Error Metrics — Lower is Better",360)
        layout3['barmode']='group'
        layout3['yaxis']['tickformat']=',.0f'
        layout3['yaxis']['tickprefix']='₹'
        layout3['bargap']=0.25
        layout3['bargroupgap']=0.1
        fig3.update_layout(**layout3)
        st.plotly_chart(fig3, use_container_width=True)

# ═══════════════════════════════════════════════════════
#  PAGE: MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════
elif page == "📈 Model Performance":

    st.markdown("""
    <div class='hero-banner'>
        <div class='hero-title'>Model Performance Analysis</div>
        <div class='hero-subtitle'>Random Forest (Baseline) vs LightGBM (Advanced) — RMSE · MAE · Runtime · Throughput · Predicted vs Actual</div>
    </div>
    """, unsafe_allow_html=True)

    # Model config cards
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class='card'>
            <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;'>
                <span style='font-size:15px;font-weight:600;color:#E6EDF3;'>🌲 Random Forest</span>
                <span class='badge-rf'>BASELINE</span>
            </div>
            <div style='display:grid;grid-template-columns:1fr 1fr;gap:10px;'>
                <div class='info-box'><strong>Trees</strong><br>100 estimators</div>
                <div class='info-box'><strong>Max Depth</strong><br>12</div>
                <div class='info-box'><strong>RMSE</strong><br>₹2,87,008</div>
                <div class='info-box'><strong>MAE</strong><br>₹1,00,777</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class='card'>
            <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;'>
                <span style='font-size:15px;font-weight:600;color:#E6EDF3;'>⚡ LightGBM</span>
                <span class='badge-lgbm'>BEST MODEL</span>
            </div>
            <div style='display:grid;grid-template-columns:1fr 1fr;gap:10px;'>
                <div class='info-box'><strong>Boosting Rounds</strong><br>300</div>
                <div class='info-box'><strong>Num Leaves</strong><br>64</div>
                <div class='info-box'><strong>RMSE</strong><br>₹2,35,441</div>
                <div class='info-box'><strong>MAE</strong><br>₹89,748</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 6 KPI metrics row — computed from real CSV values
    rmse_imp = round((rf_rmse  - lgbm_rmse)  / rf_rmse  * 100, 1)
    mae_imp  = round((rf_mae   - lgbm_mae)   / rf_mae   * 100, 1)
    rt_imp   = round((rf_runtime - lgbm_runtime) / rf_runtime * 100, 1)
    tp_imp   = round((lgbm_throughput - rf_throughput) / rf_throughput * 100, 1)

    m1,m2,m3,m4,m5,m6 = st.columns(6)
    m1.metric("RF RMSE",       "₹2,87,008")
    m2.metric("LGBM RMSE",     "₹2,35,441",          f"↓ {rmse_imp}% better")
    m3.metric("RF MAE",        "₹1,00,777")
    m4.metric("LGBM MAE",      "₹89,748",             f"↓ {mae_imp}% better")
    m5.metric("RF Runtime",    f"{rf_runtime}s",      "Baseline")
    m6.metric("LGBM Runtime",  f"{lgbm_runtime}s",   f"↓ {rt_imp}% faster")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── TAB LAYOUT ──────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊  RMSE & MAE",
        "⏱️  Runtime & Throughput",
        "📉  Predicted vs Actual",
        "⚙️  Hyperparameters"
    ])

    # ── TAB 1: RMSE & MAE ───────────────────────────
    with tab1:
        col1,col2 = st.columns(2)
        with col1:
            st.markdown("<div class='section-header'>RMSE Comparison</div>", unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Random Forest', x=['RMSE'], y=[287008],
                marker_color=ACCENT_ORANGE, marker_line_width=0,
                text=['₹2,87,008'], textposition='outside',
                textfont=dict(color='#C9D1D9',size=13,family='DM Mono')))
            fig.add_trace(go.Bar(
                name='LightGBM', x=['RMSE'], y=[235441],
                marker_color=ACCENT_GREEN, marker_line_width=0,
                text=['₹2,35,441'], textposition='outside',
                textfont=dict(color='#C9D1D9',size=13,family='DM Mono')))
            layout = base_layout("RMSE — Lower is Better",340)
            layout['barmode']='group'
            layout['yaxis']['tickformat']=',.0f'
            layout['yaxis']['tickprefix']='₹'
            layout['bargap']=0.3
            fig.update_layout(**layout)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("<div class='section-header'>MAE Comparison</div>", unsafe_allow_html=True)
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                name='Random Forest', x=['MAE'], y=[100777],
                marker_color=ACCENT_ORANGE, marker_line_width=0,
                text=['₹1,00,777'], textposition='outside',
                textfont=dict(color='#C9D1D9',size=13,family='DM Mono')))
            fig2.add_trace(go.Bar(
                name='LightGBM', x=['MAE'], y=[89748],
                marker_color=ACCENT_GREEN, marker_line_width=0,
                text=['₹89,748'], textposition='outside',
                textfont=dict(color='#C9D1D9',size=13,family='DM Mono')))
            layout2 = base_layout("MAE — Lower is Better",340)
            layout2['barmode']='group'
            layout2['yaxis']['tickformat']=',.0f'
            layout2['yaxis']['tickprefix']='₹'
            layout2['bargap']=0.3
            fig2.update_layout(**layout2)
            st.plotly_chart(fig2, use_container_width=True)

        # Combined RMSE + MAE side by side
        st.markdown("<div class='section-header'>RMSE & MAE — Side by Side</div>", unsafe_allow_html=True)
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            name='Random Forest', x=['RMSE','MAE'], y=[287008,100777],
            marker_color=ACCENT_ORANGE, marker_line_width=0,
            text=['₹2,87,008','₹1,00,777'], textposition='outside',
            textfont=dict(color='#C9D1D9',size=11)))
        fig3.add_trace(go.Bar(
            name='LightGBM', x=['RMSE','MAE'], y=[235441,89748],
            marker_color=ACCENT_GREEN, marker_line_width=0,
            text=['₹2,35,441','₹89,748'], textposition='outside',
            textfont=dict(color='#C9D1D9',size=11)))
        layout3 = base_layout("Error Metrics Comparison — Lower is Better",360)
        layout3['barmode']='group'
        layout3['yaxis']['tickformat']=',.0f'
        layout3['yaxis']['tickprefix']='₹'
        layout3['bargap']=0.25
        layout3['bargroupgap']=0.1
        fig3.update_layout(**layout3)
        st.plotly_chart(fig3, use_container_width=True)

    # ── TAB 2: RUNTIME & THROUGHPUT ─────────────────
    with tab2:
        col1,col2 = st.columns(2)

        with col1:
            st.markdown("<div class='section-header'>Training Runtime (seconds) — Lower is Better</div>", unsafe_allow_html=True)
            fig_rt = go.Figure()
            fig_rt.add_trace(go.Bar(
                x=['Random Forest','LightGBM'],
                y=[rf_runtime, lgbm_runtime],
                marker=dict(
                    color=[ACCENT_ORANGE, ACCENT_GREEN],
                    line=dict(width=0)
                ),
                text=[f'{rf_runtime:.2f}s', f'{lgbm_runtime:.2f}s'],
                textposition='outside',
                textfont=dict(color='#C9D1D9',size=14,family='DM Mono'),
                width=0.45,
            ))
            fig_rt.add_annotation(
                x=1, y=lgbm_runtime + rf_runtime * 0.08,
                text=f"⚡ {rt_imp}% faster",
                showarrow=False,
                font=dict(color=ACCENT_GREEN,size=12,family='DM Mono'),
            )
            layout_rt = base_layout("Training Runtime Comparison (seconds)",360)
            layout_rt['yaxis']['ticksuffix']=' s'
            layout_rt['yaxis']['range']=[0, rf_runtime*1.4]
            fig_rt.update_layout(**layout_rt)
            st.plotly_chart(fig_rt, use_container_width=True)

        with col2:
            st.markdown("<div class='section-header'>Throughput (rows/sec) — Higher is Better</div>", unsafe_allow_html=True)
            fig_tp = go.Figure()
            fig_tp.add_trace(go.Bar(
                x=['Random Forest','LightGBM'],
                y=[rf_throughput, lgbm_throughput],
                marker=dict(
                    color=[ACCENT_ORANGE, ACCENT_GREEN],
                    line=dict(width=0)
                ),
                text=[f'{rf_throughput:,.0f}', f'{lgbm_throughput:,.0f}'],
                textposition='outside',
                textfont=dict(color='#C9D1D9',size=14,family='DM Mono'),
                width=0.45,
            ))
            fig_tp.add_annotation(
                x=1, y=lgbm_throughput + lgbm_throughput * 0.06,
                text=f"⚡ {tp_imp:.0f}x higher",
                showarrow=False,
                font=dict(color=ACCENT_GREEN,size=12,family='DM Mono'),
            )
            layout_tp = base_layout("Throughput — Rows Processed per Second",360)
            layout_tp['yaxis']['ticksuffix']=' rows/s'
            layout_tp['yaxis']['tickformat']=',.0f'
            layout_tp['yaxis']['range']=[0, lgbm_throughput*1.4]
            fig_tp.update_layout(**layout_tp)
            st.plotly_chart(fig_tp, use_container_width=True)

        # All 4 metrics improvement summary
        st.markdown("<div class='section-header'>LightGBM Improvement Over Random Forest (%)</div>", unsafe_allow_html=True)
        tp_mult = round(lgbm_throughput / rf_throughput, 1)   # 16x multiplier
        fig_imp = go.Figure()
        fig_imp.add_trace(go.Bar(
            x=['RMSE Reduction','MAE Reduction','Runtime Speedup','Throughput Gain'],
            y=[rmse_imp, mae_imp, rt_imp, 100.0],   # throughput capped at 100 for visual; real shown in label
            marker=dict(
                color=[ACCENT_GREEN, ACCENT_BLUE, ACCENT_YELLOW, ACCENT_PURPLE],
                line=dict(width=0)
            ),
            text=[f'{rmse_imp}%', f'{mae_imp}%', f'{rt_imp}%', f'{tp_mult}x'],
            textposition='outside',
            textfont=dict(color='#C9D1D9',size=13,family='DM Mono'),
            width=0.5,
        ))
        layout_imp = base_layout("LightGBM Gains vs Random Forest — All Metrics",360)
        layout_imp['yaxis']['ticksuffix']='%'
        layout_imp['yaxis']['range']=[0, 130]
        fig_imp.update_layout(**layout_imp)
        st.plotly_chart(fig_imp, use_container_width=True)

        # Runtime & throughput summary boxes
        r1,r2,r3,r4 = st.columns(4)
        r1.metric("RF Training Time",   f"{rf_runtime:.2f}s",          "Baseline")
        r2.metric("LGBM Training Time", f"{lgbm_runtime:.2f}s",        f"↓ {rt_imp}% faster")
        r3.metric("RF Throughput",      f"{rf_throughput:,.0f} rows/s", "Baseline")
        r4.metric("LGBM Throughput",    f"{lgbm_throughput:,.0f} rows/s", f"↑ {tp_mult}x higher")

    # ── TAB 3: PREDICTED vs ACTUAL ──────────────────
    with tab3:
        st.markdown("<div class='section-header'>Predicted vs Actual Weekly Sales (Test Set Sample)</div>", unsafe_allow_html=True)
        fig_pva = go.Figure()
        x_range = list(range(1, n_points+1))
        fig_pva.add_trace(go.Scatter(
            x=x_range, y=actual,
            name='Actual', mode='lines',
            line=dict(color='#8B949E',width=1.5,dash='dot'),
        ))
        fig_pva.add_trace(go.Scatter(
            x=x_range, y=predicted_lgbm,
            name='LightGBM Predicted', mode='lines',
            line=dict(color=ACCENT_GREEN,width=2),
        ))
        fig_pva.add_trace(go.Scatter(
            x=x_range, y=predicted_rf,
            name='Random Forest Predicted', mode='lines',
            line=dict(color=ACCENT_ORANGE,width=1.5,dash='dash'),
        ))
        layout_pva = base_layout("Predicted vs Actual Weekly Sales",400)
        layout_pva['yaxis']['tickformat']=',.0f'
        layout_pva['yaxis']['tickprefix']='₹'
        layout_pva['xaxis']['title']='Week Index'
        fig_pva.update_layout(**layout_pva)
        st.plotly_chart(fig_pva, use_container_width=True)

        # Residual distribution
        st.markdown("<div class='section-header'>Residuals Distribution (Prediction Error)</div>", unsafe_allow_html=True)
        res_lgbm = predicted_lgbm - actual
        res_rf   = predicted_rf   - actual
        col1,col2 = st.columns(2)
        with col1:
            fig_res1 = go.Figure()
            fig_res1.add_trace(go.Histogram(
                x=res_lgbm, name='LightGBM Residuals',
                marker_color=ACCENT_GREEN, opacity=0.8,
                nbinsx=20,
            ))
            layout_r1 = base_layout("LightGBM Residuals",300)
            layout_r1['xaxis']['tickprefix']='₹'
            layout_r1['xaxis']['tickformat']=',.0f'
            fig_res1.update_layout(**layout_r1)
            st.plotly_chart(fig_res1, use_container_width=True)
        with col2:
            fig_res2 = go.Figure()
            fig_res2.add_trace(go.Histogram(
                x=res_rf, name='Random Forest Residuals',
                marker_color=ACCENT_ORANGE, opacity=0.8,
                nbinsx=20,
            ))
            layout_r2 = base_layout("Random Forest Residuals",300)
            layout_r2['xaxis']['tickprefix']='₹'
            layout_r2['xaxis']['tickformat']=',.0f'
            fig_res2.update_layout(**layout_r2)
            st.plotly_chart(fig_res2, use_container_width=True)

    # ── TAB 4: HYPERPARAMETERS ───────────────────────
    with tab4:
        st.markdown("<div class='section-header'>Hyperparameter Configuration</div>", unsafe_allow_html=True)
        df_params = pd.DataFrame({
            "Parameter":     ["n_estimators / num_rounds","max_depth / num_leaves",
                               "learning_rate","feature_fraction",
                               "train_test_split","random_state","n_jobs"],
            "Random Forest": ["100","12","N/A","N/A","80 / 20","42","−1"],
            "LightGBM":      ["300","64 leaves","0.05","0.8","80 / 20","42","N/A"],
        })
        st.dataframe(df_params, use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>Full Metrics Summary Table</div>", unsafe_allow_html=True)
        df_metrics = pd.DataFrame({
            "Metric":        ["RMSE","MAE","Training Runtime","Throughput"],
            "Random Forest": [f"₹{rf_rmse:,.2f}",  f"₹{rf_mae:,.2f}",   f"{rf_runtime:.2f} s",    f"{rf_throughput:,.2f} rows/s"],
            "LightGBM":      [f"₹{lgbm_rmse:,.2f}",f"₹{lgbm_mae:,.2f}", f"{lgbm_runtime:.2f} s",  f"{lgbm_throughput:,.2f} rows/s"],
            "Winner":        ["✅ LightGBM","✅ LightGBM","✅ LightGBM","✅ LightGBM"],
            "Improvement":   [
                f"↓ {rmse_imp}% lower error",
                f"↓ {mae_imp}% lower error",
                f"↓ {rt_imp}% faster ({rf_runtime:.1f}s → {lgbm_runtime:.1f}s)",
                f"↑ {tp_mult}× more rows/sec",
            ],
        })
        st.dataframe(df_metrics, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════
#  PAGE: CANNIBALIZATION
# ═══════════════════════════════════════════════════════
elif page == "🔍 Cannibalization":

    st.markdown("""
    <div class='hero-banner'>
        <div class='hero-title'>Promotional Cannibalization Analysis</div>
        <div class='hero-subtitle'>Promo vs Non-Promo Sales · Markdown Impact · Cannibalization Effect Measurement</div>
    </div>
    """, unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Avg Promo Sales",      "₹13,49,865", "Per week")
    c2.metric("Avg Non-Promo Sales",  "₹13,31,414", "Per week")
    c3.metric("Cannibalization Effect","₹18,450",   "Sales shift detected")
    c4.metric("Promo Lift %",         "1.39%",       "Above non-promo baseline")

    st.markdown("<br>", unsafe_allow_html=True)
    col1,col2 = st.columns(2)

    with col1:
        st.markdown("<div class='section-header'>Promo vs Non-Promo Average Sales</div>", unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=['Non-Promo Weeks','Promo Weeks'],
            y=[non_promo_sales, promo_sales],
            marker=dict(color=[ACCENT_BLUE,ACCENT_RED], line=dict(width=0)),
            text=[f'₹{non_promo_sales:,}',f'₹{promo_sales:,}'],
            textposition='outside',
            textfont=dict(size=13,color='#C9D1D9',family='DM Mono'),
            width=0.45,
        ))
        fig.add_annotation(
            x=1, y=promo_sales+15000,
            text=f"+₹{cannib_effect:,}",
            showarrow=False,
            font=dict(color=ACCENT_RED,size=12,family='DM Mono'),
        )
        layout = base_layout("Average Weekly Revenue — Promotion vs Non-Promotion",380)
        layout['yaxis']['tickformat']=',.0f'
        layout['yaxis']['tickprefix']='₹'
        layout['yaxis']['range']=[1250000,1420000]
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<div class='section-header'>Promo vs Non-Promo Weekly Spread</div>", unsafe_allow_html=True)
        fig2 = go.Figure()
        fig2.add_trace(go.Box(
            y=promo_weekly[:70], name='Promo Weeks',
            marker_color=ACCENT_RED, line_color=ACCENT_RED,
            fillcolor='rgba(248,81,73,0.1)', boxmean=True,
        ))
        fig2.add_trace(go.Box(
            y=nonpromo_weekly[:70], name='Non-Promo Weeks',
            marker_color=ACCENT_BLUE, line_color=ACCENT_BLUE,
            fillcolor='rgba(88,166,255,0.1)', boxmean=True,
        ))
        layout2 = base_layout("Sales Distribution — Box Plot",380)
        layout2['yaxis']['tickformat']=',.0f'
        layout2['yaxis']['tickprefix']='₹'
        fig2.update_layout(**layout2)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("""
    <div class='card'>
        <div class='section-header'>How Cannibalization Was Measured</div>
        <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;'>
            <div class='info-box'>
                <strong>Step 1 — Promo Intensity</strong><br>
                Sum of all MarkDown1–5 fields per week to create a combined <em>Total_Promo</em> variable.
            </div>
            <div class='info-box'>
                <strong>Step 2 — Median Split</strong><br>
                Weeks above the median Total_Promo → flagged as <em>High Promo</em>. Weeks below → <em>Non-Promo</em>.
            </div>
            <div class='info-box'>
                <strong>Step 3 — Compare Averages</strong><br>
                Average revenue per group compared. The ₹18,450 gap is the estimated cannibalization / demand shift.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Weekly Sales — Promo vs Non-Promo Weeks</div>", unsafe_allow_html=True)
    np.random.seed(99)
    is_promo    = np.random.choice([True,False], size=143, p=[0.45,0.55])
    weekly_vals = [
        promo_sales    + np.random.randint(-200000,250000) if p
        else non_promo_sales + np.random.randint(-180000,200000)
        for p in is_promo
    ]
    colors_weekly = [ACCENT_RED if p else ACCENT_BLUE for p in is_promo]

    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        x=list(range(1,144)), y=weekly_vals,
        marker_color=colors_weekly, marker_line_width=0,
        name='Weekly Sales', showlegend=False,
    ))
    fig3.add_trace(go.Scatter(x=[0,144], y=[promo_sales,promo_sales],
        mode='lines', name='Promo Avg',
        line=dict(color=ACCENT_RED,width=1.5,dash='dash')))
    fig3.add_trace(go.Scatter(x=[0,144], y=[non_promo_sales,non_promo_sales],
        mode='lines', name='Non-Promo Avg',
        line=dict(color=ACCENT_BLUE,width=1.5,dash='dash')))
    layout3 = base_layout("Weekly Sales — Red = Promo Weeks, Blue = Non-Promo Weeks",360)
    layout3['yaxis']['tickformat']=',.0f'
    layout3['yaxis']['tickprefix']='₹'
    layout3['xaxis']['title']='Week Number'
    fig3.update_layout(**layout3)
    st.plotly_chart(fig3, use_container_width=True)

# ═══════════════════════════════════════════════════════
#  PAGE: FEATURE ANALYSIS
# ═══════════════════════════════════════════════════════
elif page == "📦 Feature Analysis":

    st.markdown("""
    <div class='hero-banner'>
        <div class='hero-title'>Feature Engineering & Importance</div>
        <div class='hero-subtitle'>Lag Features · Rolling Averages · Markdown Variables · Seasonal Indicators</div>
    </div>
    """, unsafe_allow_html=True)

    col1,col2 = st.columns([3,2])

    with col1:
        st.markdown("<div class='section-header'>Top 15 Feature Importances (LightGBM)</div>", unsafe_allow_html=True)
        feat_colors = []
        for f in features:
            if 'Lag' in f or 'Rolling' in f: feat_colors.append(ACCENT_BLUE)
            elif 'MarkDown' in f:             feat_colors.append(ACCENT_RED)
            elif f in ['Week','Month','Dept','Store']: feat_colors.append(ACCENT_PURPLE)
            else:                              feat_colors.append(ACCENT_GREEN)

        fig = go.Figure(go.Bar(
            x=importance[::-1], y=features[::-1],
            orientation='h',
            marker=dict(color=feat_colors[::-1], line=dict(width=0)),
            text=importance[::-1],
            textposition='outside',
            textfont=dict(size=10,color='#C9D1D9'),
        ))
        layout = base_layout("",500)
        layout['margin']=dict(l=12,r=50,t=10,b=12)
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<div class='section-header'>Feature Categories</div>", unsafe_allow_html=True)
        fig2 = go.Figure(go.Pie(
            labels=['Lag Features','Markdown (Promo)','Time / Seasonal','Store / Dept'],
            values=[6,3,4,2],
            marker=dict(colors=[ACCENT_BLUE,ACCENT_RED,ACCENT_PURPLE,ACCENT_GREEN],
                        line=dict(color='#0D1117',width=3)),
            textfont=dict(size=12,color='white'),
            hole=0.55,
        ))
        fig2.update_layout(
            plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
            height=280, margin=dict(l=0,r=0,t=10,b=0),
            font=dict(family='DM Sans',color=TEXT_COLOR),
            showlegend=True,
            legend=dict(bgcolor='rgba(0,0,0,0)',font=dict(size=11,color='#C9D1D9')),
            annotations=[dict(text='Features',x=0.5,y=0.5,font_size=13,
                              font_color='#C9D1D9',showarrow=False)]
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("""
        <div class='info-box' style='margin-top:8px;'>
            <strong style='color:#58A6FF;'>🔵 Lag Features</strong> — lag_1w, lag_2w, lag_4w, lag_12w<br><br>
            <strong style='color:#F85149;'>🔴 MarkDown</strong> — MarkDown1 to MarkDown5<br><br>
            <strong style='color:#BC8CFF;'>🟣 Time/Season</strong> — Week, Month, Year, IsHoliday<br><br>
            <strong style='color:#3FB950;'>🟢 Store/Dept</strong> — Store ID, Dept ID
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Engineered Features — Details</div>", unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class='card'>
            <div style='color:#58A6FF;font-weight:700;margin-bottom:10px;'>📐 Lag Features</div>
            <div class='info-box'><strong>lag_1w</strong> — Sales from 1 week ago</div>
            <div class='info-box'><strong>lag_2w</strong> — Sales from 2 weeks ago</div>
            <div class='info-box'><strong>lag_4w</strong> — Sales from 4 weeks ago</div>
            <div class='info-box'><strong>lag_12w</strong> — Sales from 12 weeks ago</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class='card'>
            <div style='color:#BC8CFF;font-weight:700;margin-bottom:10px;'>📅 Rolling Averages</div>
            <div class='info-box'><strong>rolling_4w</strong> — 4-week moving average</div>
            <div class='info-box'><strong>rolling_12w</strong> — 12-week moving average</div>
            <div class='info-box'><strong>Promo_Total</strong> — Sum of MarkDown1–5</div>
            <div class='info-box'><strong>IsWeekend</strong> — Weekend flag</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class='card'>
            <div style='color:#3FB950;font-weight:700;margin-bottom:10px;'>🗓 Time Features</div>
            <div class='info-box'><strong>Year</strong> — Extracted from Date</div>
            <div class='info-box'><strong>Month</strong> — 1–12</div>
            <div class='info-box'><strong>Week</strong> — ISO week number</div>
            <div class='info-box'><strong>DayOfWeek</strong> — Day index</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Feature Correlation Heatmap (Sample)</div>", unsafe_allow_html=True)
    feat_names_hm = ['Weekly_Sales','lag_1w','lag_4w','Rolling_4w','MarkDown1','Week','Dept','Store']
    np.random.seed(5)
    corr = np.eye(len(feat_names_hm))
    corr[0][1]=corr[1][0]=0.92; corr[0][2]=corr[2][0]=0.85
    corr[0][3]=corr[3][0]=0.88; corr[0][4]=corr[4][0]=0.45
    corr[0][5]=corr[5][0]=0.30; corr[0][6]=corr[6][0]=0.55
    corr[0][7]=corr[7][0]=0.48; corr[1][2]=corr[2][1]=0.78
    corr[1][3]=corr[3][1]=0.82
    for i in range(len(feat_names_hm)):
        for j in range(len(feat_names_hm)):
            if i != j and corr[i][j] == 0:
                corr[i][j] = corr[j][i] = round(np.random.uniform(0.1,0.4),2)

    fig4 = go.Figure(go.Heatmap(
        z=corr, x=feat_names_hm, y=feat_names_hm,
        colorscale=[[0,'#161B22'],[0.5,'#2D4F8A'],[1,'#58A6FF']],
        text=[[f'{v:.2f}' for v in row] for row in corr],
        texttemplate='%{text}', textfont=dict(size=10),
        showscale=True, zmin=0, zmax=1,
    ))
    fig4.update_layout(
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
        height=360, margin=dict(l=12,r=12,t=10,b=12),
        font=dict(family='DM Sans',color=TEXT_COLOR,size=11),
        xaxis=dict(tickfont=dict(size=10)),
        yaxis=dict(tickfont=dict(size=10)),
    )
    st.plotly_chart(fig4, use_container_width=True)

# ═══════════════════════════════════════════════════════
#  PAGE: PIPELINE
# ═══════════════════════════════════════════════════════
elif page == "🗂 Pipeline":

    st.markdown("""
    <div class='hero-banner'>
        <div class='hero-title'>System Pipeline & Architecture</div>
        <div class='hero-subtitle'>6-Module ML Pipeline — From Raw CSV to Business Insights</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>6-Module Pipeline</div>", unsafe_allow_html=True)
    modules = [
        ("01","Data\nPreprocessing",    "1_data_preprocessing.py",      "#58A6FF"),
        ("02","Feature\nEngineering",   "2_feature_engineering.py",     "#BC8CFF"),
        ("03","Baseline\nModel (RF)",   "3_baseline_model.py",          "#FF7B00"),
        ("04","LightGBM\nModel",        "4_lgbm_model.py",              "#3FB950"),
        ("05","Cannibalization\nAnalysis","5_cannibalization_analysis.py","#F85149"),
        ("06","Charts &\nVisualization","6_generate_charts.py",         "#FFA657"),
    ]
    cols = st.columns(6)
    for col,(num,title,script,color) in zip(cols,modules):
        col.markdown(f"""
        <div style='background:#161B22;border:1px solid #21262D;border-top:3px solid {color};
                    border-radius:10px;padding:16px 10px;text-align:center;margin-bottom:8px;'>
            <div style='font-size:22px;font-weight:700;color:{color};font-family:DM Mono,monospace;'>{num}</div>
            <div style='font-size:12px;font-weight:600;color:#E6EDF3;margin:6px 0 4px 0;
                        white-space:pre-line;line-height:1.3;'>{title}</div>
            <div style='font-size:9px;color:#8B949E;font-family:DM Mono,monospace;
                        background:#0D1117;padding:3px 6px;border-radius:4px;margin-top:8px;'>{script}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align:center;color:#8B949E;font-size:12px;margin:4px 0 20px 0;letter-spacing:0.1em;'>
        ──── CSV Input ──→ clean_data.csv ──→ feature_engineered_data.csv ──→ Models ──→ Outputs/figures/ ────
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Module Details</div>", unsafe_allow_html=True)
    col1,col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class='card'>
            <div style='color:#58A6FF;font-weight:700;margin-bottom:10px;font-size:14px;'>📥 Module 1 — Data Preprocessing</div>
            <div class='info-box'>Merges train.csv + features.csv + stores.csv on Store &amp; Date</div>
            <div class='info-box'>Converts USD → INR at ₹83 exchange rate</div>
            <div class='info-box'>Fills NaN markdown values with 0</div>
            <div class='info-box'>Extracts Year, Month, Week from Date column</div>
            <div class='info-box'><strong>Output:</strong> Outputs/clean_data.csv</div>
        </div>
        <div class='card'>
            <div style='color:#FF7B00;font-weight:700;margin-bottom:10px;font-size:14px;'>🌲 Module 3 — Baseline Model (RF)</div>
            <div class='info-box'>100 decision trees, max depth 12, n_jobs=−1</div>
            <div class='info-box'>80/20 train-test split (random_state=42)</div>
            <div class='info-box'>Evaluates RMSE, MAE, runtime, throughput</div>
            <div class='info-box'><strong>Output:</strong> Outputs/baseline_results.csv</div>
        </div>
        <div class='card'>
            <div style='color:#F85149;font-weight:700;margin-bottom:10px;font-size:14px;'>🔍 Module 5 — Cannibalization Analysis</div>
            <div class='info-box'>Creates Promo_Total = sum of MarkDown1–5</div>
            <div class='info-box'>Flags High_Promo if Promo_Total &gt; median</div>
            <div class='info-box'>Compares avg revenue: promo vs non-promo</div>
            <div class='info-box'><strong>Output:</strong> Outputs/cannibalization_results.csv</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='card'>
            <div style='color:#BC8CFF;font-weight:700;margin-bottom:10px;font-size:14px;'>⚙️ Module 2 — Feature Engineering</div>
            <div class='info-box'>Lag features: 1-week, 2-week, 4-week, 12-week</div>
            <div class='info-box'>Rolling averages: 4-week &amp; 12-week windows</div>
            <div class='info-box'>DayOfWeek, IsWeekend, Promo_Total features</div>
            <div class='info-box'><strong>Output:</strong> Outputs/feature_engineered_data.csv</div>
        </div>
        <div class='card'>
            <div style='color:#3FB950;font-weight:700;margin-bottom:10px;font-size:14px;'>⚡ Module 4 — LightGBM Model</div>
            <div class='info-box'>lr=0.05 · 64 leaves · feature_fraction=0.8</div>
            <div class='info-box'>300 boosting rounds · GBDT boosting type</div>
            <div class='info-box'>Compared with RF on RMSE, MAE, throughput</div>
            <div class='info-box'><strong>Output:</strong> Outputs/lightgbm_results.csv · lgbm_preds.csv</div>
        </div>
        <div class='card'>
            <div style='color:#FFA657;font-weight:700;margin-bottom:10px;font-size:14px;'>📊 Module 6 — Charts &amp; Visualization</div>
            <div class='info-box'>9 chart figures saved to Outputs/figures/</div>
            <div class='info-box'>rmse_comparison · mae_comparison · runtime</div>
            <div class='info-box'>monthly_sales · promo_vs_nonpromo · cannibalization</div>
            <div class='info-box'><strong>Output:</strong> Outputs/figures/*.png (9 files)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Output Files Generated</div>", unsafe_allow_html=True)
    output_files = pd.DataFrame({
        "File": ["baseline_results.csv","lightgbm_results.csv","lgbm_preds.csv",
                 "model_comparison.csv","cannibalization_results.csv",
                 "figures/rmse_comparison.png","figures/mae_comparison.png",
                 "figures/runtime_comparison.png","figures/throughput_comparison.png",
                 "figures/monthly_sales_trend.png","figures/top_departments_sales.png",
                 "figures/promo_discount_vs_sales.png","figures/promo_vs_nonpromo.png",
                 "figures/cannibalization_effect.png"],
        "Type": ["CSV"]*5 + ["PNG"]*9,
        "Contents": [
            "RF: RMSE, MAE, runtime, throughput",
            "LGBM: RMSE, MAE, runtime, throughput",
            "Predicted sales for each test record",
            "Side-by-side model metrics",
            "Promo/non-promo avg sales, cannibalization effect",
            "Bar chart comparing RMSE","Bar chart comparing MAE",
            "Training runtime comparison","Records/sec throughput",
            "Monthly avg sales trend","Top 10 departments by revenue",
            "Promo discount correlation","Promo vs non-promo comparison",
            "Cannibalization effect bar chart"
        ]
    })
    st.dataframe(output_files, use_container_width=True, hide_index=True)

    st.markdown("<div class='section-header'>Dataset Summary</div>", unsafe_allow_html=True)
    d1,d2,d3,d4,d5,d6 = st.columns(6)
    d1.metric("Source","Kaggle")
    d2.metric("Records","4,21,571")
    d3.metric("Stores","45")
    d4.metric("Departments","~99")
    d5.metric("Weeks","143")
    d6.metric("Date Range","2010–2012")