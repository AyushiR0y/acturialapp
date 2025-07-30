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
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Insurance Policy Analytics", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced CSS styling
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
        max-width: 1200px;
    }
    
    /* Header styling - shorter and with logo space */
    .main-header {
        background: linear-gradient(135deg, #004A94 50%, #031c36 100%);
        padding: 1.5rem 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 10px 25px rgba(0,0,0,0.15);
        position: relative;
        overflow: hidden;
        display: flex;
        align-items: center;
        justify-content: space-between;
        min-height: 80px;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: pulse 4s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 0.8; }
    }
    
    .logo-space {
        width: 80px;
        height: 60px;
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2rem;
        z-index: 1;
        border: 2px dashed rgba(255,255,255,0.3);
    }
    
    .header-content {
        text-align: center;
        flex-grow: 1;
        z-index: 1;
    }
    
    .header-content h1 {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .header-content p {
        font-size: 1rem;
        opacity: 0.9;
        margin: 0;
    }
    
    /* Welcome section - no box styling */
    .welcome-section {
        text-align: center;
        margin: 1.5rem 0;
        color: #2d3748;
    }
    
    .welcome-section h2 {
        font-size: 1.8rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #004A94;
    }
    
    .welcome-section p {
        font-size: 1.1rem;
        color: #666;
        margin: 0;
    }
    
    /* Feature cards - smaller and uniform */
    .feature-card {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 0.5rem;
        text-align: center;
        box-shadow: 0 6px 20px rgba(168, 237, 234, 0.2);
        transition: all 0.3s ease;
        color: #2d3748;
        height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .feature-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 25px rgba(168, 237, 234, 0.3);
    }
    
    .feature-card h3 {
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    
    .feature-card p {
        font-size: 0.9rem;
        margin: 0;
        line-height: 1.3;
    }
    
    /* Upload container - greyish blue with better positioning */
    .upload-container {
        background: linear-gradient(135deg, #8da2b5 0%, #6c7b8a 100%);
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 8px 25px rgba(141, 162, 181, 0.3);
        border: 2px dashed rgba(255,255,255,0.4);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .upload-container:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 35px rgba(141, 162, 181, 0.4);
    }
    
    .upload-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        animation: shimmer 2s infinite;
    }
    
    @keyframes shimmer {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    .upload-text {
        color: white;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    
    .upload-subtext {
        color: rgba(255,255,255,0.9);
        font-size: 0.95rem;
        margin-bottom: 1rem;
    }
    
    /* Metric cards - smaller */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.25);
        transition: all 0.3s ease;
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 30px rgba(102, 126, 234, 0.35);
    }
    
    .metric-card .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    
    .metric-card .metric-label {
        font-size: 0.8rem;
        opacity: 0.85;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Section headers - smaller */
    .section-header {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 1.5rem 0 1rem 0;
        color: #2d3748;
        border-left: 4px solid #667eea;
        box-shadow: 0 6px 20px rgba(255, 154, 158, 0.2);
    }
    
    .section-header h3 {
        margin: 0;
        font-weight: 600;
        font-size: 1.2rem;
    }
    
    /* Insight boxes - smaller */
    .insight-box {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid #38b2ac;
        box-shadow: 0 6px 20px rgba(168, 237, 234, 0.25);
        color: #2d3748;
    }
    
    .insight-box h3 {
        color: #2d3748;
        margin-bottom: 0.8rem;
        font-size: 1.1rem;
    }
    
    /* Form elements */
    .stSelectbox > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 8px;
        border: none;
    }
    
    .stSelectbox > div > div > div {
        color: white;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.4rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* File uploader styling - make the uploader itself look like the upload container */
    .stFileUploader {
        margin: 1.5rem 0;
    }
    
    .stFileUploader > div {
        background: linear-gradient(135deg, #8da2b5 0%, #6c7b8a 100%);
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 8px 25px rgba(141, 162, 181, 0.3);
        border: 2px dashed rgba(255,255,255,0.4);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
        cursor: pointer;
    }
    
    .stFileUploader > div:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 35px rgba(141, 162, 181, 0.4);
    }
    
    .stFileUploader > div::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        animation: shimmer 2s infinite;
    }
    
    .stFileUploader > div > div {
        background: transparent !important;
        border: none !important;
    }
    
    .stFileUploader label {
        color: white !important;
        font-size: 1.3rem !important;
        font-weight: 600 !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3) !important;
        margin-bottom: 0.5rem !important;
    }
    
    .stFileUploader label::after {
        content: '\ADrag and drop your CSV or Excel file here, or click to browse';
        display: block;
        color: rgba(255,255,255,0.9) !important;
        font-size: 0.95rem !important;
        font-weight: 400 !important;
        margin-top: 0.5rem;
        text-shadow: none !important;
    }
    
    .uploadedFile {
        background: rgba(255,255,255,0.2) !important;
        border-radius: 8px;
        padding: 0.8rem;
        margin: 0.8rem 0;
        color: white !important;
    }
    
    .stFileUploader button {
        background: rgba(255,255,255,0.2) !important;
        color: white !important;
        border: 1px solid rgba(255,255,255,0.3) !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        margin-top: 1rem !important;
    }
    
    .stFileUploader button:hover {
        background: rgba(255,255,255,0.3) !important;
    }
    
    /* Success message */
    .success-message {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        color: #2d3748;
        padding: 0.8rem 1.5rem;
        border-radius: 8px;
        margin: 0.8rem 0;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(132, 250, 176, 0.25);
    }
    
    /* Charts and dataframes */
    .stPlotlyChart {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
    }
    
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Header with logo space
st.markdown('''
<div class="main-header" style="display: flex; align-items: center; justify-content: space-between;">
    <div class="logo-space">
        <img src="https://raw.githubusercontent.com/AyushiR0y/acturialapp/b441a13bec87acbf09781c05ca84a6c0c70b53c7/logo.png" alt="Logo" style="height: 50px;">
    </div>
    <div class="header-content">
        <h1>Insurance Policy Analytics</h1>
        <p>Advanced Actuarial Analysis & Risk Management</p>
    </div>
    <div style="width: 80px;"></div>
</div>
''', unsafe_allow_html=True)


# Welcome section (no box)
st.markdown('''
<div class="welcome-section">
    <h2>Welcome to Insurance Policy Analytics</h2>
    <p>Transform your insurance data into actionable insights with advanced analytics and machine learning</p>
</div>
''', unsafe_allow_html=True)

# Feature cards - smaller and uniform
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('''
    <div class="feature-card">
        <h3>Risk Analytics</h3>
        <p>Automated risk scoring, anomaly detection, and predictive risk modeling</p>
    </div>
    ''', unsafe_allow_html=True)

with col2:
    st.markdown('''
    <div class="feature-card">
        <h3>Premium Intelligence</h3>
        <p>Loss ratio analysis, premium adequacy assessment, and profitability insights</p>
    </div>
    ''', unsafe_allow_html=True)

with col3:
    st.markdown('''
    <div class="feature-card">
        <h3>Advanced Analytics</h3>
        <p>Machine learning clustering, temporal patterns, and statistical analysis</p>
    </div>
    ''', unsafe_allow_html=True)

# File upload section - clickable upload box
uploaded_file = st.file_uploader(
    label="Upload Your Policy Data",
    type=["csv", "xlsx"], 
    help="Upload CSV or Excel file containing policy data",
    label_visibility="collapsed"
)

if uploaded_file:
    try:
        # Load data
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        
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
        
        st.markdown(f'<div class="success-message">Data loaded successfully: {len(df):,} policies processed</div>', unsafe_allow_html=True)
        
        # Display key metrics
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
        
        # Analysis selection
        st.markdown('<div class="section-header"><h3>Choose Your Analysis</h3></div>', unsafe_allow_html=True)
        
        analysis_type = st.selectbox("", [
            "Executive Dashboard",
            "Risk Segmentation Analysis", 
            "Premium & Loss Analysis",
            "Temporal Patterns",
            "Anomaly Detection",
            "Advanced Clustering",
            "Predictive Analytics",
            "Detailed Policy Explorer"
        ])
        
        # Main analysis area
        if analysis_type == "Executive Dashboard":
            st.markdown('<div class="section-header"><h3>Executive Summary Dashboard</h3></div>', unsafe_allow_html=True)
            
            # Charts
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
                                color_continuous_scale='Viridis')
                    fig.add_hline(y=1.0, line_dash="dash", line_color="red", 
                                 annotation_text="Break-even line")
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#2d3748')
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                sample_df = df.sample(min(1000, len(df)))
                fig = px.scatter(sample_df, 
                               x='ANNUAL_PREM', y='RES_GP_PUPS',
                               color='LOSS_RATIO',
                               title="Premium vs Claims Distribution",
                               color_continuous_scale='Plasma')
                # Add diagonal line
                max_val = max(df['ANNUAL_PREM'].max(), df['RES_GP_PUPS'].max())
                fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], 
                                       mode='lines', line=dict(dash='dash', color='red'),
                                       name='Break-even line'))
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#2d3748')
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Business insights
            st.markdown('<div class="insight-box">', unsafe_allow_html=True)
            st.markdown("### Key Business Insights")
            
            profitable_ratio = (df['PREMIUM_ADEQUACY'] > 0).mean()
            high_loss_ratio_count = (df['LOSS_RATIO'] > 1.5).sum()
            
            insights = f"""
            **Portfolio Health:** {profitable_ratio:.1%} of policies are profitable  
            **Risk Concentration:** {high_loss_ratio_count:,} policies have loss ratios > 150%  
            **Premium Adequacy:** Average premium adequacy is ₹{df['PREMIUM_ADEQUACY'].mean():,.0f}  
            **Performance Indicator:** Current portfolio loss ratio is {avg_loss_ratio:.2f}
            """
            st.markdown(insights)
            st.markdown('</div>', unsafe_allow_html=True)
        
        elif analysis_type == "Risk Segmentation Analysis":
            st.markdown('<div class="section-header"><h3>Risk Segmentation Analysis</h3></div>', unsafe_allow_html=True)
            
            # Risk scoring
            df['RISK_CATEGORY'] = pd.cut(df['RISK_SCORE'], 
                                       bins=[0, 25, 50, 75, 100, float('inf')],
                                       labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                risk_dist = df['RISK_CATEGORY'].value_counts()
                fig = px.pie(values=risk_dist.values, names=risk_dist.index,
                           title="Policy Risk Distribution",
                           color_discrete_sequence=px.colors.qualitative.Set3)
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#2d3748')
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                risk_premium = df.groupby('RISK_CATEGORY')['ANNUAL_PREM'].agg(['mean', 'sum', 'count']).reset_index()
                fig = px.bar(risk_premium, x='RISK_CATEGORY', y='mean',
                           title="Average Premium by Risk Category",
                           color='mean', color_continuous_scale='Viridis')
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#2d3748')
                )
                st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Please ensure your file has the correct format and column names.")