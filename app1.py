import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
import time
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Insurance Policy Analytics", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Professional Corporate CSS styling
st.markdown("""
<style>
    /* Hide default streamlit elements */
    .stDeployButton {display: none;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Main container styling */
    .main > .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 1400px;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #004A94 0%, #1e3a8a 100%);
        padding: 1.2rem 2rem;
        border-radius: 8px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(0, 74, 148, 0.2);
        display: flex;
        align-items: center;
        justify-content: space-between;
        min-height: 70px;
    }
    
    .logo-space {
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .logo-space img {
        border-radius: 4px;
    }
    
    .header-content {
        text-align: center;
        flex-grow: 1;
    }
    
    .header-content h1 {
        font-size: 1.8rem;
        font-weight: 600;
        margin-bottom: 0.2rem;
        color: white;
    }
    
    .header-content p {
        font-size: 0.9rem;
        opacity: 0.9;
        margin: 0;
    }
    
    /* Welcome section */
    .welcome-section {
        text-align: center;
        margin: 1.5rem 0;
        color: #1f2937;
    }
    
    .welcome-section h2 {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 0.3rem;
        color: #1f2937;
    }
    
    .welcome-section p {
        font-size: 1rem;
        color: #6b7280;
        margin: 0;
    }
    
    /* --- Base Card Style with Shimmer Prep --- */
.feature-card {
    padding: 1.3rem;
    border-radius: 8px;
    margin: 0.3rem;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    transition: all 0.2s ease;
    color: white; /* Dark text for light backgrounds */
    height: 140px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    border: 1px solid rgba(0,0,0,0.05);
    
    /* Added for shimmer effect */
    position: relative;
    overflow: hidden;
}

/* --- CARD-SPECIFIC GRADIENTS (More Reliable) --- */

/* Card 1: Light Blue to Light Pink */
.card-1 {
    background: linear-gradient(135deg, #a8c0ff, #fbc2eb);
}

/* Card 2: Light Yellow to Light Orange */
.card-2 {
    background: linear-gradient(135deg, #fdfd96, #ffcf96);
}

/* Card 3: Light Red to Light Lavender */
.card-3 {
    background: linear-gradient(135deg, #ffafbd, #dcd0ff);
}

/* --- Gleaming Shimmer Effect --- */
.feature-card {
    position: relative;
    overflow: hidden;
}

.feature-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: -60%;
    width: 60%;
    height: 100%;
    background: linear-gradient(
        to right,
        rgba(255,255,255,0) 0%,
        rgba(255,255,255,0.1) 50%,
        rgba(255,255,255,0) 100%
    );
    transform: skewX(-20deg);
    animation: shimmer 4s linear infinite;
    pointer-events: none;
}

@keyframes shimmer {
    0% {
        left: -60%;
    }
    100% {
        left: 120%;
    }
}



.feature-card:hover::before {
    left: 150%;
}


/* --- Hover & Text Styles --- */
.feature-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}

.feature-card h3 {
    font-size: 1.1rem;
    margin-bottom: 0.6rem;
    font-weight: 600;
    position: relative; /* Ensures text is on top of shimmer */
}

.feature-card p {
    font-size: 0.85rem;
    margin: 0;
    line-height: 1.3;
    opacity: 0.95;
    position: relative; /* Ensures text is on top of shimmer */
}
    /* Clean file upload section */
    .upload-container {
        margin: 1.5rem 0;
    }
    
    .uploaded-file-display {
        background: white;
        padding: 0.8rem 1.2rem;
        border-radius: 6px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border: 1px solid #e5e7eb;
        color: #374151;
    }
    
    .file-info {
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .file-name {
        font-size: 0.95rem;
        font-weight: 500;
        color: #1f2937;
    }
    
    .file-size {
        font-size: 0.8rem;
        color: #6b7280;
    }
    
    .delete-icon {
        background: none;
        border: none;
        color: #ef4444;
        cursor: pointer;
        font-size: 1.1rem;
        padding: 0.2rem;
        border-radius: 4px;
        transition: all 0.2s ease;
    }
    
    .delete-icon:hover {
        background: #fef2f2;
        color: #dc2626;
    }
    
    /* File uploader styling */
    .stFileUploader > div {
        background: #f8fafc;
        border-radius: 8px;
        padding: 2rem;
        text-align: center;
        border: 2px dashed #cbd5e1;
        transition: all 0.2s ease;
    }
    
    .stFileUploader > div:hover {
        border-color: #004A94;
        background: #f1f5f9;
    }
    
    .stFileUploader label {
        color: #475569 !important;
        font-size: 1.1rem !important;
        font-weight: 500 !important;
    }
    
    /* Professional analyze button */
    .analyze-button {
        display: flex;
        justify-content: center;
        margin: 1.5rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #7dd3fc 0%, #38bdf8 100%) !important;
        color: #1e40af !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.6rem 2rem !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 4px rgba(56, 189, 248, 0.2) !important;
        font-size: 0.9rem !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #38bdf8 0%, #0284c7 100%) !important;
        color: white !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 8px rgba(56, 189, 248, 0.3) !important;
    }
    
    /* Loader styling */
    .loader-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 2rem;
        flex-direction: column;
    }
    
    .loader {
        border: 3px solid #e5e7eb;
        border-top: 3px solid #004A94;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin-bottom: 1rem;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .loader-text {
        color: #6b7280;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    /* Compact metric cards */
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 6px;
        color: #1f2937;
        text-align: center;
        margin: 0.3rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        transition: all 0.2s ease;
        border-left: 4px solid #004A94;
    }
    
    .metric-card:nth-child(2n) {
        border-left-color: #0ea5e9;
    }
    
    .metric-card:nth-child(3n) {
        border-left-color: #8b5cf6;
    }
    
    .metric-card:nth-child(4n) {
        border-left-color: #f59e0b;
    }
    
    .metric-card:nth-child(5n) {
        border-left-color: #10b981;
    }
    
    .metric-card:hover {
        transform: translateY(-1px);
        box-shadow: 0 2px 6px rgba(0,0,0,0.12);
    }
    
    .metric-card .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        color: #1f2937;
    }
    
    .metric-card .metric-label {
        font-size: 0.75rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
    }
    
    /* Compact section headers */
    .section-header {
        background: #f8fafc;
        padding: 1rem 1rem;
        border-radius: 6px;
        margin: 1.5rem 0 1rem 0;
        color: #1f2937;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        border-left: 4px solid #004A94;
    }
    
    .section-header h3 {
        margin: 0;
        font-weight: 600;
        font-size: 1.1rem;
        color: #004A94;
    }
    
    /* Professional insight boxes */
    .insight-box {
        background: #f8fafc;
        padding: 1.3rem;
        border-radius: 6px;
        margin: 1rem 0;
        border-left: 4px solid #004A94;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        color: #1f2937;
    }
    
    .insight-box h3 {
        color: #004A94;
        margin-bottom: 0.8rem;
        font-size: 1rem;
        font-weight: 600;
    }
    
    .stSelectbox > div[data-baseweb="select"] > div {
    background-color: white !important;
    color: #374151 !important; /* Dark text for selected value */
    border: 1px solid #d1d5db !important;
    border-radius: 6px !important;
}

/* This targets the placeholder text */
.stSelectbox > div[data-baseweb="select"] > div > div {
    color: #6b7280 !important;
}

/* --- Dropdown Menu & Options --- */

/* This targets an individual option in the dropdown */
[data-baseweb="popover"] li {
    color: #374151 !important; /* Dark text for dropdown options */
    font-size: 0.95rem !important;
}

/* Style for the HIGHLIGHTED/SELECTED option in the dropdown list */
[data-baseweb="popover"] li[aria-selected="true"] {
    background-color: #e0f2ff !important; /* A light blue background */
    color: #003b73 !important; /* A contrasting dark blue text */
}

/* Style for a HOVERED option in the dropdown list */
[data-baseweb="popover"] li:hover {
    background-color: #f0f0f0 !important; /* A light gray background on hover */
    color: #374151 !important;
}
    
    
    
    /* Charts styling */
    .stPlotlyChart {
        border-radius: 6px;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin: 0.8rem 0;
        background: white;
    }
    
    .stDataFrame {
        border-radius: 6px;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin: 0.8rem 0;
    }
    
    /* Compact analysis container */
    .analysis-container {
        margin-top: 1.5rem;
    }
    
    /* Reduce column gaps */
    .block-container .element-container {
        margin-bottom: 0.5rem;
    }
    
    /* Compact spacing */
    .stColumns > div {
        padding-left: 0.25rem !important;
        padding-right: 0.25rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Header with logo
st.markdown('''
<div class="main-header">
    <div class="logo-space">
        <img src="https://raw.githubusercontent.com/AyushiR0y/acturialapp/b441a13bec87acbf09781c05ca84a6c0c70b53c7/logo.png" alt="Logo" style="height: 50px;">
    </div>
    <div class="header-content">
        <h1>Insurance Policy Analytics</h1>
        <p>Advanced Actuarial Analysis & Risk Management Platform</p>
    </div>
    <div style="width: 60px;"></div>
</div>
''', unsafe_allow_html=True)

# Welcome section
st.markdown('''
<div class="welcome-section">
    <h2>Welcome to your Insurance Analytics tool!</h2>
    <p>Transform your insurance data into actionable business insights with advanced analytics</p>
</div>
''', unsafe_allow_html=True)

# Feature cards with corporate colors
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('''
    <div class="feature-card" style="background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);">
        <h3>Risk Analytics</h3>
        <p>Automated risk scoring, anomaly detection, and predictive risk modeling</p>
    </div>
    ''', unsafe_allow_html=True)

with col2:
    st.markdown('''
    <div class="feature-card" style="background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%);">
        <h3>Premium Intelligence</h3>
        <p>Loss ratio analysis, premium adequacy assessment, and profitability insights</p>
    </div>
    ''', unsafe_allow_html=True)

with col3:
    st.markdown('''
    <div class="feature-card" style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);">
        <h3>Advanced Analytics</h3>
        <p>Machine learning clustering, temporal patterns, and statistical analysis</p>
    </div>
    ''', unsafe_allow_html=True)

# File upload section
st.markdown('<div class="upload-container">', unsafe_allow_html=True)

# Initialize session state for file management
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'show_loader' not in st.session_state:
    st.session_state.show_loader = False

# File upload logic
if st.session_state.uploaded_file is None:
    uploaded_file = st.file_uploader("Upload your insurance data", type=["csv", "xlsx"], 
                                   help="Upload CSV or Excel file containing policy data")
    if uploaded_file:
        st.session_state.uploaded_file = uploaded_file
        st.rerun()
else:
    uploaded_file = st.session_state.uploaded_file
    
    # Display uploaded file info with clean design
    st.markdown(f'''
    <div class="uploaded-file-display">
        <div class="file-info">
            <span style="font-size: 1.2rem; color: #374151;">📄</span>
            <div>
                <div class="file-name">{uploaded_file.name}</div>
                <div class="file-size">({uploaded_file.size / 1024:.1f} KB)</div>
            </div>
        </div>
        <button class="delete-icon" onclick="window.parent.postMessage({{type: 'streamlit:setComponentValue', key: 'delete_file', value: true}}, '*')">✕</button>
    </div>
    ''', unsafe_allow_html=True)
    
    # Delete button (hidden, triggered by the X icon)
    if st.button("Remove File", key="delete_file", help="Remove uploaded file"):
        st.session_state.uploaded_file = None
        st.session_state.show_loader = False
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# Analyze button
if st.session_state.uploaded_file:
    st.markdown('<div class="analyze-button">', unsafe_allow_html=True)
    analyze_data = st.button(" Analyze Data", key="analyze_btn")
    st.markdown('</div>', unsafe_allow_html=True)
    
    if analyze_data:
        st.session_state.show_loader = True
        st.rerun()
else:
    analyze_data = False

# Show loader
if st.session_state.show_loader:
    st.markdown('''
    <div class="loader-container">
        <div class="loader"></div>
        <div class="loader-text">Processing your data...</div>
    </div>
    ''', unsafe_allow_html=True)
    
    # Simulate processing time
    time.sleep(2)
    st.session_state.show_loader = False
    st.rerun()

if st.session_state.uploaded_file and not st.session_state.show_loader:
    try:
        # Load data
        if st.session_state.uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(st.session_state.uploaded_file)
        elif st.session_state.uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(st.session_state.uploaded_file)
        
        # Clean column names
        df.columns = df.columns.str.strip().str.upper()
        
        # Convert numeric columns
        numeric_cols = ['ANNUAL_PREM', 'EXP_GP_PUP', 'ACT_GP_PUP', 'PREM_GP_PUPS', 'RES_GP_PUPS', 'NZ_RES_IF_94']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Calculate derived metrics
        df['LOSS_RATIO'] = df['RES_GP_PUPS'] / (df['PREM_GP_PUPS'] + 1e-6)
        df['PREMIUM_ADEQUACY'] = df['PREM_GP_PUPS'] - df['RES_GP_PUPS']
        df['EXPECTED_VS_ACTUAL'] = df['ACT_GP_PUP'] / (df['EXP_GP_PUP'] + 1e-6)
        df['RISK_SCORE'] = (df['RES_GP_PUPS'] / (df['ANNUAL_PREM'] + 1e-6)) * 100
        
        # Create vintage columns
        if 'ENTRY_YEAR' in df.columns:
            df['POLICY_VINTAGE'] = 2024 - df['ENTRY_YEAR']
        if 'ENTRY_MONTH' in df.columns:
            df['ENTRY_SEASON'] = df['ENTRY_MONTH'].map({
                12: 'Q4', 1: 'Q1', 2: 'Q1', 3: 'Q1',
                4: 'Q2', 5: 'Q2', 6: 'Q2',
                7: 'Q3', 8: 'Q3', 9: 'Q3',
                10: 'Q4', 11: 'Q4'
            })
        
        
        # Display key metrics with compact cards
        st.markdown('<div style="margin: 1.5rem 0;">', unsafe_allow_html=True)
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            total_policies = len(df)
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-value">{total_policies:,}</div>
                <div class="metric-label">Total Policies</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            total_premium = df['ANNUAL_PREM'].sum()
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-value">₹{total_premium/1e6:.1f}M</div>
                <div class="metric-label">Total Premium</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col3:
            avg_loss_ratio = df['LOSS_RATIO'].mean()
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-value">{avg_loss_ratio:.2f}</div>
                <div class="metric-label">Avg Loss Ratio</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col4:
            profitable_count = (df['PREMIUM_ADEQUACY'] > 0).sum()
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-value">{profitable_count:,}</div>
                <div class="metric-label">Profitable Policies</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col5:
            high_risk_count = (df['RISK_SCORE'] > df['RISK_SCORE'].quantile(0.9)).sum()
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-value">{high_risk_count:,}</div>
                <div class="metric-label">High Risk Policies</div>
            </div>
            ''', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Analysis selection
        st.markdown('<div class="section-header"><h3> Select Analysis Type</h3></div>', unsafe_allow_html=True)
        
        analysis_type = st.selectbox("Choose analysis:", [
            "Executive Dashboard",
            "Risk Segmentation Analysis", 
            "Premium & Loss Analysis",
            "Temporal Patterns",
            "Anomaly Detection",
            "Advanced Clustering",
            "Predictive Analytics",
            "Detailed Policy Explorer"
        ], label_visibility="collapsed")
        
        # Main analysis area
        st.markdown('<div class="analysis-container">', unsafe_allow_html=True)
        
        if analysis_type == "Executive Dashboard":
            st.markdown('<div class="section-header"><h3> Executive Summary Dashboard</h3></div>', unsafe_allow_html=True)
            
            # Charts with professional styling
            col1, col2 = st.columns(2)
            
            with col1:
                if 'CL_PBAND' in df.columns:
                    loss_by_band = df.groupby('CL_PBAND').agg({
                        'LOSS_RATIO': 'mean',
                        'POL_NUMBER': 'count',
                        'ANNUAL_PREM': 'sum'
                    }).reset_index()
                    
                    fig = px.bar(loss_by_band, x='CL_PBAND', y='LOSS_RATIO', 
                                title="Loss Ratio by Premium Band",
                                color='LOSS_RATIO',
                                color_continuous_scale=['#7dd3fc', '#004A94'])
                    fig.add_hline(y=1.0, line_dash="dash", line_color="#ef4444", 
                                 annotation_text="Break-even line")
                    fig.update_layout(
                        paper_bgcolor='white',
                        plot_bgcolor='white',
                        font=dict(color='#374151', size=11),
                        title_font_size=14,
                        margin=dict(t=50, b=40, l=40, r=40)
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                sample_df = df.sample(min(1000, len(df)))
                fig = px.scatter(sample_df, 
                               x='ANNUAL_PREM', y='RES_GP_PUPS',
                               color='LOSS_RATIO',
                               title="Premium vs Claims Distribution",
                               color_continuous_scale=['#e0f2fe', '#004A94'])
                max_val = max(df['ANNUAL_PREM'].max(), df['RES_GP_PUPS'].max())
                fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], 
                                       mode='lines', line=dict(dash='dash', color='#ef4444'),
                                       name='Break-even line'))
                fig.update_layout(
                    paper_bgcolor='white',
                    plot_bgcolor='white',
                    font=dict(color='#374151', size=11),
                    title_font_size=14,
                    margin=dict(t=50, b=40, l=40, r=40)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Business insights
            st.markdown('''
            <div class="insight-box">
                <h3>💡 Key Business Insights</h3>
            ''', unsafe_allow_html=True)
            
            profitable_ratio = (df['PREMIUM_ADEQUACY'] > 0).mean()
            high_loss_ratio_count = (df['LOSS_RATIO'] > 1.5).sum()
            
            insights = f"""
            **Portfolio Health:** {profitable_ratio:.1%} of policies are profitable<br>
            **Risk Concentration:** {high_loss_ratio_count:,} policies have loss ratios > 150%<br>
            **Premium Adequacy:** Average premium adequacy is ₹{df['PREMIUM_ADEQUACY'].mean():,.0f}<br>
            **Performance Indicator:** Current portfolio loss ratio is {avg_loss_ratio:.2f}
            """
            st.markdown(insights, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        elif analysis_type == "Risk Segmentation Analysis":
            st.markdown('<div class="section-header"><h3>⚠️ Risk Segmentation Analysis</h3></div>', unsafe_allow_html=True)
            
            # Risk scoring and segmentation
            df['COMBINED_RISK_SCORE'] = (
                (df['LOSS_RATIO'] * 0.4) + 
                (df['RISK_SCORE'] / 100 * 0.3) + 
                (df['EXPECTED_VS_ACTUAL'] * 0.3)
            )
            
            # Define risk segments
            df['RISK_SEGMENT'] = pd.cut(df['COMBINED_RISK_SCORE'], 
                                       bins=[0, 0.5, 1.0, 1.5, float('inf')],
                                       labels=['Low Risk', 'Medium Risk', 'High Risk', 'Critical Risk'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Risk distribution
                risk_dist = df['RISK_SEGMENT'].value_counts().reset_index()
                risk_dist.columns = ['Risk_Segment', 'Count']
                
                fig = px.pie(risk_dist, values='Count', names='Risk_Segment', 
                           title="Risk Distribution Across Portfolio",
                           color_discrete_map={
                               'Low Risk': '#10b981',
                               'Medium Risk': '#f59e0b', 
                               'High Risk': '#f97316',
                               'Critical Risk': '#ef4444'
                           })
                fig.update_layout(
                    paper_bgcolor='white',
                    font=dict(color='#374151', size=11),
                    title_font_size=14
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Risk vs Premium scatter
                sample_df = df.sample(min(1000, len(df)))
                fig = px.scatter(sample_df, 
                               x='ANNUAL_PREM', y='COMBINED_RISK_SCORE',
                               color='RISK_SEGMENT',
                               title="Risk Score vs Premium Analysis",
                               color_discrete_map={
                                   'Low Risk': '#10b981',
                                   'Medium Risk': '#f59e0b', 
                                   'High Risk': '#f97316',
                                   'Critical Risk': '#ef4444'
                               })
                fig.update_layout(
                    paper_bgcolor='white',
                    plot_bgcolor='white',
                    font=dict(color='#374151', size=11),
                    title_font_size=14
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Risk segment analysis
            risk_analysis = df.groupby('RISK_SEGMENT').agg({
                'ANNUAL_PREM': ['count', 'sum', 'mean'],
                'LOSS_RATIO': 'mean',
                'PREMIUM_ADEQUACY': 'mean',
                'RES_GP_PUPS': 'sum'
            }).round(2)
            
            st.markdown('<div class="section-header"><h3> Risk Segment Performance</h3></div>', unsafe_allow_html=True)
            st.dataframe(risk_analysis, use_container_width=True)
            
            # Risk insights
            st.markdown('''
            <div class="insight-box">
                <h3>⚠️ Risk Management Insights</h3>
            ''', unsafe_allow_html=True)
            
            critical_risk_count = len(df[df['RISK_SEGMENT'] == 'Critical Risk'])
            high_risk_premium = df[df['RISK_SEGMENT'].isin(['High Risk', 'Critical Risk'])]['ANNUAL_PREM'].sum()
            
            insights = f"""
            **Critical Risk Policies:** {critical_risk_count:,} policies require immediate attention<br>
            **High-Risk Premium Exposure:** ₹{high_risk_premium/1e6:.1f}M in high-risk segments<br>
            **Risk Concentration:** {(critical_risk_count/len(df)*100):.1f}% of portfolio is critical risk<br>
            **Recommended Action:** Review underwriting criteria for high-risk segments
            """
            st.markdown(insights, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        elif analysis_type == "Premium & Loss Analysis":
            st.markdown('<div class="section-header"><h3> Premium & Loss Analysis</h3></div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Loss ratio distribution
                fig = px.histogram(df, x='LOSS_RATIO', bins=50,
                                 title="Loss Ratio Distribution",
                                 color_discrete_sequence=['#004A94'])
                fig.add_vline(x=1.0, line_dash="dash", line_color="#ef4444",
                             annotation_text="Break-even")
                fig.add_vline(x=df['LOSS_RATIO'].mean(), line_dash="dot", line_color="#10b981",
                             annotation_text=f"Mean: {df['LOSS_RATIO'].mean():.2f}")
                fig.update_layout(
                    paper_bgcolor='white',
                    plot_bgcolor='white',
                    font=dict(color='#374151', size=11),
                    title_font_size=14
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Premium adequacy analysis
                df['ADEQUACY_CATEGORY'] = pd.cut(df['PREMIUM_ADEQUACY'], 
                                               bins=[-float('inf'), -10000, 0, 10000, float('inf')],
                                               labels=['Highly Inadequate', 'Inadequate', 'Adequate', 'Highly Adequate'])
                
                adequacy_dist = df['ADEQUACY_CATEGORY'].value_counts().reset_index()
                adequacy_dist.columns = ['Category', 'Count']
                
                fig = px.bar(adequacy_dist, x='Category', y='Count',
                           title="Premium Adequacy Distribution",
                           color='Count',
                           color_continuous_scale=['#ef4444', '#10b981'])
                fig.update_layout(
                    paper_bgcolor='white',
                    plot_bgcolor='white',
                    font=dict(color='#374151', size=11),
                    title_font_size=14
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Combined loss and premium analysis
            if 'CL_PBAND' in df.columns:
                band_analysis = df.groupby('CL_PBAND').agg({
                    'ANNUAL_PREM': ['sum', 'mean', 'count'],
                    'RES_GP_PUPS': ['sum', 'mean'],
                    'LOSS_RATIO': 'mean',
                    'PREMIUM_ADEQUACY': ['mean', 'sum']
                }).round(2)
                
                st.markdown('<div class="section-header"><h3> Premium Band Analysis</h3></div>', unsafe_allow_html=True)
                st.dataframe(band_analysis, use_container_width=True)
            
            # Profitability insights
            st.markdown('''
            <div class="insight-box">
                <h3>💼 Profitability Insights</h3>
            ''', unsafe_allow_html=True)
            
            total_profit = df['PREMIUM_ADEQUACY'].sum()
            profitable_policies = (df['PREMIUM_ADEQUACY'] > 0).sum()
            avg_loss_ratio = df['LOSS_RATIO'].mean()
            
            insights = f"""
            **Total Portfolio Profit:** ₹{total_profit/1e6:.1f}M<br>
            **Profitable Policies:** {profitable_policies:,} ({profitable_policies/len(df)*100:.1f}%)<br>
            **Average Loss Ratio:** {avg_loss_ratio:.2f}<br>
            **Break-even Analysis:** {(df['LOSS_RATIO'] <= 1.0).sum():,} policies are profitable
            """
            st.markdown(insights, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        elif analysis_type == "Temporal Patterns":
            st.markdown('<div class="section-header"><h3> Temporal Patterns Analysis</h3></div>', unsafe_allow_html=True)
            
            if 'ENTRY_YEAR' in df.columns and 'ENTRY_MONTH' in df.columns:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Yearly trends
                    yearly_trends = df.groupby('ENTRY_YEAR').agg({
                        'POL_NUMBER': 'count',
                        'ANNUAL_PREM': 'sum',
                        'LOSS_RATIO': 'mean',
                        'RES_GP_PUPS': 'sum'
                    }).reset_index()
                    
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    
                    fig.add_trace(
                        go.Bar(x=yearly_trends['ENTRY_YEAR'], y=yearly_trends['POL_NUMBER'],
                              name="Policy Count", marker_color='#7dd3fc'),
                        secondary_y=False,
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=yearly_trends['ENTRY_YEAR'], y=yearly_trends['LOSS_RATIO'],
                                  mode='lines+markers', name="Loss Ratio", line=dict(color='#ef4444')),
                        secondary_y=True,
                    )
                    
                    fig.update_xaxes(title_text="Entry Year")
                    fig.update_yaxes(title_text="Policy Count", secondary_y=False)
                    fig.update_yaxes(title_text="Loss Ratio", secondary_y=True)
                    fig.update_layout(title_text="Yearly Policy and Loss Ratio Trends")
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Monthly seasonality
                    monthly_trends = df.groupby('ENTRY_MONTH').agg({
                        'POL_NUMBER': 'count',
                        'ANNUAL_PREM': 'mean',
                        'LOSS_RATIO': 'mean'
                    }).reset_index()
                    
                    fig = px.line(monthly_trends, x='ENTRY_MONTH', y='LOSS_RATIO',
                                title="Monthly Loss Ratio Seasonality",
                                markers=True, line_shape='spline')
                    fig.add_hline(y=1.0, line_dash="dash", line_color="#ef4444",
                                 annotation_text="Break-even")
                    fig.update_layout(
                        paper_bgcolor='white',
                        plot_bgcolor='white',
                        font=dict(color='#374151', size=11),
                        title_font_size=14
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Quarterly analysis
                if 'ENTRY_SEASON' in df.columns:
                    quarterly_analysis = df.groupby('ENTRY_SEASON').agg({
                        'POL_NUMBER': 'count',
                        'ANNUAL_PREM': ['sum', 'mean'],
                        'LOSS_RATIO': 'mean',
                        'PREMIUM_ADEQUACY': 'mean'
                    }).round(2)
                    
                    st.markdown('<div class="section-header"><h3> Quarterly Performance</h3></div>', unsafe_allow_html=True)
                    st.dataframe(quarterly_analysis, use_container_width=True)
            
            # Vintage analysis
            if 'POLICY_VINTAGE' in df.columns:
                vintage_analysis = df.groupby('POLICY_VINTAGE').agg({
                    'LOSS_RATIO': 'mean',
                    'ANNUAL_PREM': 'sum',
                    'POL_NUMBER': 'count'
                }).reset_index()
                
                fig = px.scatter(vintage_analysis, x='POLICY_VINTAGE', y='LOSS_RATIO',
                               size='POL_NUMBER', 
                               title="Policy Vintage vs Loss Ratio",
                               color='LOSS_RATIO',
                               color_continuous_scale=['#10b981', '#ef4444'])
                fig.add_hline(y=1.0, line_dash="dash", line_color="#ef4444")
                fig.update_layout(
                    paper_bgcolor='white',
                    plot_bgcolor='white',
                    font=dict(color='#374151', size=11),
                    title_font_size=14
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Temporal insights
            st.markdown('''
            <div class="insight-box">
                <h3>📈 Temporal Insights</h3>
            ''', unsafe_allow_html=True)
            
            if 'ENTRY_YEAR' in df.columns:
                latest_year_lr = df[df['ENTRY_YEAR'] == df['ENTRY_YEAR'].max()]['LOSS_RATIO'].mean()
                best_month = monthly_trends.loc[monthly_trends['LOSS_RATIO'].idxmin(), 'ENTRY_MONTH']
                worst_month = monthly_trends.loc[monthly_trends['LOSS_RATIO'].idxmax(), 'ENTRY_MONTH']
                
                insights = f"""
                **Latest Year Performance:** {latest_year_lr:.2f} loss ratio<br>
                **Best Month:** Month {best_month} (lowest loss ratio)<br>
                **Worst Month:** Month {worst_month} (highest loss ratio)<br>
                **Trend Analysis:** {'Improving' if latest_year_lr < 1.0 else 'Needs attention'}
                """
                st.markdown(insights, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        elif analysis_type == "Anomaly Detection":
            st.markdown('<div class="section-header"><h3> Anomaly Detection Analysis</h3></div>', unsafe_allow_html=True)
            
            # Prepare features for anomaly detection
            feature_cols = ['ANNUAL_PREM', 'LOSS_RATIO', 'RISK_SCORE', 'EXPECTED_VS_ACTUAL']
            available_features = [col for col in feature_cols if col in df.columns]
            
            if len(available_features) >= 2:
                # Isolation Forest for anomaly detection
                X = df[available_features].fillna(df[available_features].mean())
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                isolation_forest = IsolationForest(contamination=0.1, random_state=42)
                df['ANOMALY'] = isolation_forest.fit_predict(X_scaled)
                df['ANOMALY_SCORE'] = isolation_forest.score_samples(X_scaled)
                
                # Convert to binary (1 = normal, -1 = anomaly)
                df['IS_ANOMALY'] = df['ANOMALY'] == -1
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Anomaly scatter plot
                    sample_df = df.sample(min(1000, len(df)))
                    fig = px.scatter(sample_df, 
                                   x='ANNUAL_PREM', y='LOSS_RATIO',
                                   color='IS_ANOMALY',
                                   title="Anomaly Detection: Premium vs Loss Ratio",
                                   color_discrete_map={True: '#ef4444', False: '#10b981'})
                    fig.update_layout(
                        paper_bgcolor='white',
                        plot_bgcolor='white',
                        font=dict(color='#374151', size=11),
                        title_font_size=14
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Anomaly score distribution
                    fig = px.histogram(df, x='ANOMALY_SCORE', 
                                     title="Anomaly Score Distribution",
                                     color_discrete_sequence=['#004A94'])
                    fig.add_vline(x=df['ANOMALY_SCORE'].quantile(0.1), 
                                 line_dash="dash", line_color="#ef4444",
                                 annotation_text="Anomaly Threshold")
                    fig.update_layout(
                        paper_bgcolor='white',
                        plot_bgcolor='white',
                        font=dict(color='#374151', size=11),
                        title_font_size=14
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Top anomalies
                anomalies = df[df['IS_ANOMALY']].nlargest(10, 'LOSS_RATIO')[
                    ['POL_NUMBER', 'ANNUAL_PREM', 'LOSS_RATIO', 'RISK_SCORE', 'ANOMALY_SCORE']
                ].round(2)
                
                st.markdown('<div class="section-header"><h3> Top Anomalous Policies</h3></div>', unsafe_allow_html=True)
                st.dataframe(anomalies, use_container_width=True)
                
                # Anomaly insights
                st.markdown('''
                <div class="insight-box">
                    <h3> Anomaly Detection Insights</h3>
                ''', unsafe_allow_html=True)
                
                anomaly_count = df['IS_ANOMALY'].sum()
                avg_anomaly_loss_ratio = df[df['IS_ANOMALY']]['LOSS_RATIO'].mean()
                anomaly_premium_impact = df[df['IS_ANOMALY']]['ANNUAL_PREM'].sum()
                
                insights = f"""
                **Anomalous Policies Detected:** {anomaly_count:,} ({anomaly_count/len(df)*100:.1f}%)<br>
                **Average Anomaly Loss Ratio:** {avg_anomaly_loss_ratio:.2f}<br>
                **Premium at Risk:** ₹{anomaly_premium_impact/1e6:.1f}M<br>
                **Recommendation:** Investigate top anomalies for potential fraud or underwriting issues
                """
                st.markdown(insights, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
        
        elif analysis_type == "Advanced Clustering":
            st.markdown('<div class="section-header"><h3> Advanced Clustering Analysis</h3></div>', unsafe_allow_html=True)
            
            # Prepare features for clustering
            cluster_features = ['ANNUAL_PREM', 'LOSS_RATIO', 'RISK_SCORE', 'EXPECTED_VS_ACTUAL']
            available_features = [col for col in cluster_features if col in df.columns]
            
            if len(available_features) >= 2:
                # Prepare data
                X = df[available_features].fillna(df[available_features].mean())
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Determine optimal clusters using elbow method
                inertias = []
                K_range = range(2, 8)
                for k in K_range:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    kmeans.fit(X_scaled)
                    inertias.append(kmeans.inertia_)
                
                # Perform clustering with optimal K (let's use 4 for business interpretation)
                optimal_k = 4
                kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
                df['CLUSTER'] = kmeans.fit_predict(X_scaled)
                df['CLUSTER_LABEL'] = df['CLUSTER'].map({
                    0: 'Premium Segment A',
                    1: 'Premium Segment B', 
                    2: 'Premium Segment C',
                    3: 'Premium Segment D'
                })
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Cluster visualization using PCA
                    pca = PCA(n_components=2)
                    X_pca = pca.fit_transform(X_scaled)
                    
                    fig = px.scatter(x=X_pca[:, 0], y=X_pca[:, 1], 
                                   color=df['CLUSTER_LABEL'],
                                   title="Policy Clusters (PCA Visualization)",
                                   labels={'x': f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',
                                          'y': f'PC2 ({pca.explained_variance_ratio_[1]:.1%})'})
                    fig.update_layout(
                        paper_bgcolor='white',
                        plot_bgcolor='white',
                        font=dict(color='#374151', size=11),
                        title_font_size=14
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Cluster size distribution
                    cluster_dist = df['CLUSTER_LABEL'].value_counts().reset_index()
                    cluster_dist.columns = ['Cluster', 'Count']
                    
                    fig = px.pie(cluster_dist, values='Count', names='Cluster',
                               title="Cluster Distribution")
                    fig.update_layout(
                        paper_bgcolor='white',
                        font=dict(color='#374151', size=11),
                        title_font_size=14
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Cluster analysis
                cluster_analysis = df.groupby('CLUSTER_LABEL').agg({
                    'ANNUAL_PREM': ['count', 'mean', 'sum'],
                    'LOSS_RATIO': 'mean',
                    'RISK_SCORE': 'mean',
                    'PREMIUM_ADEQUACY': 'mean'
                }).round(2)
                
                st.markdown('<div class="section-header"><h3> Cluster Characteristics</h3></div>', unsafe_allow_html=True)
                st.dataframe(cluster_analysis, use_container_width=True)
                
                # Clustering insights
                st.markdown('''
                <div class="insight-box">
                    <h3>🎯 Clustering Insights</h3>
                ''', unsafe_allow_html=True)
                
                best_cluster = cluster_analysis.loc[cluster_analysis[('LOSS_RATIO', 'mean')].idxmin()]
                worst_cluster = cluster_analysis.loc[cluster_analysis[('LOSS_RATIO', 'mean')].idxmax()]
                
                insights = f"""
                **Best Performing Cluster:** Lowest average loss ratio segment<br>
                **Highest Risk Cluster:** Requires targeted risk management<br>
                **Cluster Insights:** Clear segmentation reveals distinct risk profiles<br>
                **Business Application:** Use clusters for targeted pricing and underwriting
                """
                st.markdown(insights, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
        
        elif analysis_type == "Predictive Analytics":
            st.markdown('<div class="section-header"><h3>🔮 Predictive Analytics</h3></div>', unsafe_allow_html=True)
            
            # Prepare features for prediction
            feature_cols = ['ANNUAL_PREM', 'EXP_GP_PUP', 'ACT_GP_PUP', 'PREM_GP_PUPS']
            target_col = 'RES_GP_PUPS'
            
            available_features = [col for col in feature_cols if col in df.columns]
            
            if len(available_features) >= 2 and target_col in df.columns:
                # Prepare data
                X = df[available_features].fillna(df[available_features].mean())
                y = df[target_col].fillna(df[target_col].mean())
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Train Random Forest model
                rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                rf_model.fit(X_train, y_train)
                
                # Predictions
                y_pred = rf_model.predict(X_test)
                
                # Model performance
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Actual vs Predicted
                    fig = px.scatter(x=y_test, y=y_pred,
                                   title=f"Actual vs Predicted Claims (R² = {r2:.3f})",
                                   labels={'x': 'Actual Claims', 'y': 'Predicted Claims'})
                    
                    # Add perfect prediction line
                    min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
                    fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                           mode='lines', name='Perfect Prediction',
                                           line=dict(dash='dash', color='red')))
                    
                    fig.update_layout(
                        paper_bgcolor='white',
                        plot_bgcolor='white',
                        font=dict(color='#374151', size=11),
                        title_font_size=14
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Feature importance
                    feature_importance = pd.DataFrame({
                        'Feature': available_features,
                        'Importance': rf_model.feature_importances_
                    }).sort_values('Importance', ascending=True)
                    
                    fig = px.bar(feature_importance, x='Importance', y='Feature',
                               orientation='h', title="Feature Importance",
                               color='Importance', color_continuous_scale='Blues')
                    fig.update_layout(
                        paper_bgcolor='white',
                        plot_bgcolor='white',
                        font=dict(color='#374151', size=11),
                        title_font_size=14
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Model performance metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f'''
                    <div class="metric-card">
                        <div class="metric-value">{r2:.3f}</div>
                        <div class="metric-label">R² Score</div>
                    </div>
                    ''', unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f'''
                    <div class="metric-card">
                        <div class="metric-value">{np.sqrt(mse):,.0f}</div>
                        <div class="metric-label">RMSE</div>
                    </div>
                    ''', unsafe_allow_html=True)
                
                with col3:
                    accuracy = 100 - (abs(y_test - y_pred).mean() / y_test.mean() * 100)
                    st.markdown(f'''
                    <div class="metric-card">
                        <div class="metric-value">{accuracy:.1f}%</div>
                        <div class="metric-label">Accuracy</div>
                    </div>
                    ''', unsafe_allow_html=True)
                
                # Predictive insights
                st.markdown('''
                <div class="insight-box">
                    <h3>🔮 Predictive Model Insights</h3>
                ''', unsafe_allow_html=True)
                
                most_important_feature = feature_importance.iloc[-1]['Feature']
                
                insights = f"""
                **Model Performance:** R² score of {r2:.3f} indicates {'good' if r2 > 0.7 else 'moderate' if r2 > 0.5 else 'poor'} predictive power<br>
                **Key Predictor:** {most_important_feature} is the most important feature<br>
                **Prediction Accuracy:** {accuracy:.1f}% average accuracy on test data<br>
                **Business Application:** Use model for claims reserving and risk assessment
                """
                st.markdown(insights, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
        
        elif analysis_type == "Detailed Policy Explorer":
            st.markdown('<div class="section-header"><h3> Detailed Policy Explorer</h3></div>', unsafe_allow_html=True)
            
            # Policy search and filter
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if 'CL_PBAND' in df.columns:
                    selected_band = st.selectbox("Filter by Premium Band:", 
                                                ['All'] + list(df['CL_PBAND'].unique()))
                else:
                    selected_band = 'All'
            
            with col2:
                loss_ratio_filter = st.slider("Max Loss Ratio:", 0.0, 5.0, 5.0, 0.1)
            
            with col3:
                min_premium = st.number_input("Min Premium:", value=0, step=1000)
            
            # Apply filters
            filtered_df = df.copy()
            if selected_band != 'All':
                filtered_df = filtered_df[filtered_df['CL_PBAND'] == selected_band]
            filtered_df = filtered_df[filtered_df['LOSS_RATIO'] <= loss_ratio_filter]
            filtered_df = filtered_df[filtered_df['ANNUAL_PREM'] >= min_premium]
            
            st.markdown(f"**Showing {len(filtered_df):,} policies (filtered from {len(df):,})**")
            
            # Policy details table
            display_cols = ['POL_NUMBER', 'ANNUAL_PREM', 'LOSS_RATIO', 'PREMIUM_ADEQUACY', 'RISK_SCORE']
            available_display_cols = [col for col in display_cols if col in filtered_df.columns]
            
            if available_display_cols:
                st.dataframe(
                    filtered_df[available_display_cols].head(100).round(2),
                    use_container_width=True
                )
                
                # Summary statistics for filtered data
                st.markdown('<div class="section-header"><h3> Filtered Data Summary</h3></div>', unsafe_allow_html=True)
                
                summary_stats = filtered_df[available_display_cols[1:]].describe().round(2)
                st.dataframe(summary_stats, use_container_width=True)
                
                # Export filtered data option
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label=" Download Filtered Data as CSV",
                    data=csv,
                    file_name=f'filtered_policies_{len(filtered_df)}_records.csv',
                    mime='text/csv'
                )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"❌ Error processing data: {str(e)}")
        st.info("Please ensure your file has the correct format and column names.")