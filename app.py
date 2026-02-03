import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import time

# --- 1. VISUAL CONFIGURATION (NON-NEGOTIABLE DARK THEME) ---
st.set_page_config(
    page_title="BizLens Analytics",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. SESSION STATE MANAGEMENT (MOVED UP FOR UI LOGIC) ---
if 'data_uploaded' not in st.session_state:
    st.session_state['data_uploaded'] = False
if 'df' not in st.session_state:
    st.session_state['df'] = None
if 'processed_df' not in st.session_state:
    st.session_state['processed_df'] = None
if 'show_guide' not in st.session_state:
    st.session_state['show_guide'] = True

# --- 3. DYNAMIC BRANDING LOGIC ---
# Determine branding state based on whether data is uploaded (Dashboard Active = True)
is_dashboard_active = st.session_state['data_uploaded']

brand_class = "brand-collapsed" if is_dashboard_active else "brand-expanded"
brand_content = (
    '<span class="creator-highlight" style="font-size: 1.2rem;">AR</span>' 
    if is_dashboard_active 
    else '<p class="brand-text">Designed & Built by <span class="creator-highlight">Ayushman Raha</span></p>'
)

# Custom CSS for FAANG-style Aesthetics & Animated Branding
st.markdown(f"""
    <style>
    /* Global Dark Theme Overrides */
    .stApp {{
        background-color: #0E1117;
        color: #FAFAFA;
    }}
    
    /* BOTTOM RIGHT BRANDING (ANIMATED) */
    .bottom-right-brand {{
        position: fixed;
        bottom: 1.5rem;
        right: 1.5rem;
        z-index: 99999;
        background: rgba(30, 30, 30, 0.8); /* Glass effect */
        backdrop-filter: blur(10px);
        border: 1px solid #4CAF50;
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3); /* Green Glow */
        font-family: 'Helvetica Neue', sans-serif;
        text-align: center;
        transition: all 0.6s cubic-bezier(0.68, -0.55, 0.27, 1.55); /* Smooth Bouncy Animation */
        overflow: hidden;
        display: flex;
        align-items: center;
        justify-content: center;
    }}
    
    /* State: Welcome Screen (Full Pill) */
    .brand-expanded {{
        padding: 10px 20px;
        border-radius: 50px;
        width: auto;
        height: auto;
    }}
    
    /* State: Dashboard Active (Circle) */
    .brand-collapsed {{
        padding: 0;
        width: 50px;
        height: 50px;
        border-radius: 50%;
    }}

    .bottom-right-brand:hover {{
        transform: scale(1.1); /* Zoom on hover */
        box-shadow: 0 6px 20px rgba(76, 175, 80, 0.5);
    }}
    
    .brand-text {{
        font-size: 0.9rem;
        font-weight: 600;
        color: #E0E0E0;
        margin: 0;
        white-space: nowrap;
    }}
    .creator-highlight {{
        color: #4CAF50; /* Green Accent */
        font-weight: 800;
    }}

    /* Section Headers */
    h1, h2, h3 {{
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        color: #E0E0E0 !important;
        font-weight: 600;
    }}
    
    /* Metrics & Cards */
    div[data-testid="stMetricValue"] {{
        font-size: 1.8rem !important;
        color: #4CAF50 !important;
    }}
    .stCard {{
        background-color: #1E1E1E;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 15px;
    }}
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 20px;
    }}
    .stTabs [data-baseweb="tab"] {{
        height: 50px;
        white-space: pre-wrap;
        background-color: #0E1117;
        border-radius: 4px;
        color: #888;
        font-weight: 500;
    }}
    .stTabs [data-baseweb="tab"][aria-selected="true"] {{
        background-color: #1E1E1E;
        color: #4CAF50;
        border-bottom: 2px solid #4CAF50;
    }}
    
    /* Onboarding Cards */
    .onboarding-card {{
        background: linear-gradient(145deg, #1e1e1e, #252525);
        border-left: 4px solid #4CAF50;
        padding: 20px;
        border-radius: 5px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }}
    </style>
    
    <div class="bottom-right-brand {brand_class}">
        {brand_content}
    </div>
""", unsafe_allow_html=True)

# --- 4. HELPER FUNCTIONS ---
@st.cache_resource
def load_pipeline():
    try:
        return joblib.load('bizlens_churn_pipeline.joblib')
    except FileNotFoundError:
        return None

def reset_app():
    st.session_state['data_uploaded'] = False
    st.session_state['df'] = None
    st.session_state['processed_df'] = None
    st.session_state['show_guide'] = True
    st.rerun()

# Load Assets
pipeline = load_pipeline()
try:
    from src.etl_adapter import DataStandardizer
    etl = DataStandardizer()
except ImportError:
    st.error("⚠️ ETL Module missing. Please ensure src/etl_adapter.py exists.")
    st.stop()

# --- 5. SIDEBAR (CONTROLS ONLY) ---
with st.sidebar:
    st.markdown("# Data Controls")
    
    # File Uploader
    uploaded_file = st.file_uploader("Upload Dataset (CSV)", type=['csv'], help="Upload raw customer data to begin.")
    
    if uploaded_file is not None and not st.session_state['data_uploaded']:
        try:
            raw_df = pd.read_csv(uploaded_file)
            
            # --- ETL & CLEANING ---
            # 1. Hotfix for IBM Data (Space cleaning)
            if 'TotalCharges' in raw_df.columns:
                raw_df['TotalCharges'] = pd.to_numeric(raw_df['TotalCharges'], errors='coerce')
            
            # 2. Schema Check
            missing = etl.check_schema(raw_df)
            if not missing:
                st.session_state['df'] = raw_df
                st.session_state['data_uploaded'] = True
                st.session_state['show_guide'] = False # Auto-minimize guide
                st.success("Data Loaded Successfully")
                time.sleep(1)
                st.rerun()
            else:
                st.error(f"Schema Mismatch: Missing {len(missing)} columns.")
                st.write(missing)
        except Exception as e:
            st.error(f"Error loading file: {e}")

    st.markdown("---")
    if st.button("Start Over / Reset"):
        reset_app()
        
    st.markdown("---")
    st.caption(f"System Status: {'🟢 Online' if pipeline else '🔴 Offline'}")
    st.caption("v1.0.0 | Development Build")

# --- 6. MAIN CONTENT AREA ---

# A. WELCOME SCREEN (Empty State)
if not st.session_state['data_uploaded']:
    st.markdown("# Welcome to BizLens-Analytics")
    st.markdown("### Enterprise Customer Retention Software")
    st.markdown("Transform raw customer data into actionable retention strategies using strict-schema machine learning.")
    
    st.divider()
    
    # VIBRANT ONBOARDING GUIDE
    st.markdown("### Getting Started Guide")
    
    col_guide1, col_guide2, col_guide3 = st.columns(3)
    
    with col_guide1:
        st.markdown("""
        <div class="onboarding-card" style="border-left-color: #00C853;">
            <h4>1. Upload Data</h4>
            <p style="color:#ccc; font-size:0.9rem;">
                Use the sidebar to upload your customer CSV. 
                <br><b>Required:</b> 19-column Telecom schema.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_guide2:
        st.markdown("""
        <div class="onboarding-card" style="border-left-color: #2962FF;">
            <h4>2. Analyze & Predict</h4>
            <p style="color:#ccc; font-size:0.9rem;">
                The system automatically runs ETL cleaning and XGBoost inference to score churn probability.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_guide3:
        st.markdown("""
        <div class="onboarding-card" style="border-left-color: #FF6D00;">
            <h4>3. Act on Insights</h4>
            <p style="color:#ccc; font-size:0.9rem;">
                Navigate tabs to view High-Risk Segments and AI-generated retention strategies.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    st.info("👈 Please upload a dataset in the sidebar to initialize the workspace.")

# B. DASHBOARD (Active State)
else:
    # ... (Keep your existing data processing code here) ...
    df = st.session_state['df']
    if st.session_state['processed_df'] is None and pipeline:
        with st.spinner("Running Inference Engine..."):
            # Predict
            probs = pipeline.predict_proba(df)[:, 1]
            df['Churn_Probability'] = probs
            df['Risk_Tier'] = np.where(probs > 0.75, 'Critical', 
                                np.where(probs > 0.50, 'High', 
                                np.where(probs > 0.25, 'Medium', 'Low')))
            st.session_state['processed_df'] = df

    scored_df = st.session_state['processed_df']

    # --- NEW ANIMATED HEADING ---
    st.markdown("""
        <style>
        @keyframes gradient {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        .animated-header {
            /* Professional deep-sea to electric-cyan palette */
            background: linear-gradient(-45deg, #00C9FF, #92FE9D, #0072FF, #4E54C8);
            background-size: 300% 300%;
            animation: gradient 8s ease-in-out infinite;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3.5rem !important;
            font-weight: 800 !important;
            margin-bottom: 5px;
            padding-bottom: 0px;
            letter-spacing: -1.5px;
            /* Adds a subtle "glow" effect */
            filter: drop-shadow(0px 2px 2px rgba(0, 0, 0, 0.1));
        }
        .subheader-text {
            color: #A0AEC0; /* Slate grey for better readability */
            font-size: 1.2rem;
            font-weight: 400;
            margin-top: -10px;
            margin-bottom: 25px;
            letter-spacing: 0.5px;
            text-transform: uppercase; /* Makes the tagline feel like a lab brand */
        }
        </style>
        <h1 class="animated-header">Core Metrics Lab</h1>
        <p class="subheader-text">End-to-End Retention Diagnostics • Strategic Intervention</p>
    """, unsafe_allow_html=True)

    # --- TABS LAYOUT (Continue with your existing code) ---
    tab_overview, tab_analytics, tab_predict, tab_recommend = st.tabs([
        "Overview", 
        "Analytics", 
        "Predictive Insights", 
        "Recommendations"
    ])

    # --- TAB 1: OVERVIEW ---
    with tab_overview:
        st.markdown("### Executive Summary")
        
        # Top Level Metrics
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        
        churn_risk_count = len(scored_df[scored_df['Risk_Tier'].isin(['Critical', 'High'])])
        churn_rate_est = (churn_risk_count / len(scored_df)) * 100
        revenue_risk = scored_df[scored_df['Risk_Tier'].isin(['Critical', 'High'])]['MonthlyCharges'].sum()
        
        kpi1.metric("Total Customers", f"{len(scored_df):,}", "Active Accounts")
        kpi2.metric("At-Risk Customers", f"{churn_risk_count:,}", "High/Critical Tier")
        kpi3.metric("Projected Churn Rate", f"{churn_rate_est:.1f}%", "Next 30 Days")
        kpi4.metric("Revenue at Risk", f"${revenue_risk:,.0f}", "Monthly Recurring (MRR)")
        
        st.markdown("#### Risk Distribution")
        # Simple Bar Chart for Tiers
        tier_counts = scored_df['Risk_Tier'].value_counts().reset_index()
        tier_counts.columns = ['Tier', 'Count']
        fig_tier = px.bar(
            tier_counts, y='Tier', x='Count', orientation='h',
            color='Tier',
            color_discrete_map={'Critical': '#D32F2F', 'High': '#F57C00', 'Medium': '#FBC02D', 'Low': '#388E3C'},
            template='plotly_dark'
        )
        st.plotly_chart(fig_tier, use_container_width=True)

    # --- TAB 2: ANALYTICS ---
    with tab_analytics:
        st.markdown("### Deep-Dive Visual Analysis")
        
        col_an1, col_an2 = st.columns(2)
        
        with col_an1:
            st.markdown("**Tenure vs. Churn Risk**")
            fig_tenure = px.histogram(
                scored_df, x="tenure", color="Risk_Tier",
                nbins=20, template='plotly_dark',
                labels={"tenure": "Tenure (Months)"},
                color_discrete_map={'Critical': '#D32F2F', 'High': '#F57C00', 'Medium': '#FBC02D', 'Low': '#388E3C'}
            )
            st.plotly_chart(fig_tenure, use_container_width=True)
            
        with col_an2:
            st.markdown("**Payment Method Impact**")
            fig_pay = px.box(
                scored_df, x="PaymentMethod", y="MonthlyCharges", color="Risk_Tier",
                template='plotly_dark',
                labels={"MonthlyCharges": "Monthly Charges ($)"}
            )
            st.plotly_chart(fig_pay, use_container_width=True)

    # --- TAB 3: PREDICTIVE INSIGHTS ---
    with tab_predict:
        st.markdown("### Forward-Looking Signals")
        
        st.warning("⚠️ Showing Top 50 Critical Risk Accounts")
        
        critical_df = scored_df[scored_df['Risk_Tier'] == 'Critical'].sort_values(by='Churn_Probability', ascending=False).head(50)
        
        # Strict Table Formatting
        display_cols = ['customerID', 'Risk_Tier', 'Churn_Probability', 'Contract', 'tenure', 'MonthlyCharges', 'TotalCharges']
        
        st.dataframe(
            critical_df[display_cols].style.format({
                'Churn_Probability': '{:.1%}',
                'MonthlyCharges': '${:.2f}',
                'TotalCharges': '${:.2f}',
                'tenure': '{:} Months'
            }),
            use_container_width=True,
            height=400
        )
        
        if st.button("📥 Download Critical Risk Report"):
            csv = critical_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Confirm Download",
                data=csv,
                file_name="bizlens_critical_risk.csv",
                mime="text/csv"
            )

    # --- TAB 4: RECOMMENDATIONS ---
    with tab_recommend:
        st.subheader("Retention Strategy Console")
        
        # --- 1. DYNAMIC SUGGESTIONS BOX ---
        # Logic to generate high-level suggestions based on dataset stats
        suggestions = []
        
        # Metric Calculations
        avg_monthly = scored_df['MonthlyCharges'].mean()
        m2m_ratio = len(scored_df[scored_df['Contract'] == 'Month-to-month']) / len(scored_df)
        e_check_ratio = len(scored_df[scored_df['PaymentMethod'] == 'Electronic check']) / len(scored_df)
        churn_risk_total = len(scored_df[scored_df['Risk_Tier'].isin(['Critical', 'High'])])
        
        # Rule Engine
        if m2m_ratio > 0.40:
            suggestions.append(f"⚠️ **Contract Stability Alert:** {m2m_ratio:.1%} of your customer base is on 'Month-to-Month' contracts. Prioritize a 'Term Upgrade' campaign to lock in revenue.")
        
        if e_check_ratio > 0.30:
            suggestions.append(f"💳 **Payment Friction:** High usage of Electronic Checks ({e_check_ratio:.1%}). These often lead to involuntary churn due to failed payments. Incentivize 'Credit Card (Auto-Pay)' migration.")
        
        if avg_monthly > 65:
            suggestions.append("💲 **Price Sensitivity:** Average Monthly Charges are high ($65+). Consider introducing a 'Lite' tier to downsell customers rather than losing them entirely.")
            
        if churn_risk_total > 1000:
            suggestions.append("🚨 **Volume Warning:** High volume of at-risk accounts detected. Automated email nurture sequences are recommended over manual calling.")

        # Default fallback
        if not suggestions:
            suggestions.append("✅ **System Nominal:** Key risk indicators (Contracts, Payments) are within healthy ranges. Focus on cross-selling to 'Low Risk' loyalists.")

        # Display Suggestions
        with st.expander("📝 Automated Strategic Suggestions", expanded=True):
            for suggestion in suggestions:
                st.markdown(f"- {suggestion}")
        
        st.divider()

        # --- 2. INTERACTIVE TOOLS (MACRO & MICRO) ---
        # Split into Macro (Strategy) and Micro (Tactics)
        col_macro, col_micro = st.columns([1.2, 1])
        
        # --- LEFT: MACRO STRATEGY SIMULATOR ---
        with col_macro:
            st.markdown("""
            <div class="onboarding-card">
                <h4>ROI Simulator</h4>
                <p style="color:#aaa; font-size:0.8rem; margin-bottom:10px;">
                    Simulate the financial impact of intervention campaigns.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Interactive Controls
            sim_col1, sim_col2 = st.columns(2)
            with sim_col1:
                target_tier = st.selectbox("Target Segment", ["Critical", "High", "Medium"], index=0)
            with sim_col2:
                incentive_pct = st.slider("Discount Offer (%)", 0, 50, 15, format="%d%%")
                
            # Calculation Logic
            target_data = scored_df[scored_df['Risk_Tier'] == target_tier]
            avg_mrr = target_data['MonthlyCharges'].mean() if not target_data.empty else 0
            count_target = len(target_data)
            
            # Assumptions
            acceptance_rate = 0.45 # 45% of people accept the offer and stay
            
            # Math
            retained_customers = int(count_target * acceptance_rate)
            retained_revenue_monthly = retained_customers * avg_mrr
            cost_of_discount = (retained_revenue_monthly * (incentive_pct/100))
            net_saved_mrr = retained_revenue_monthly - cost_of_discount
            
            st.markdown("---")
            
            # Results Display
            st.markdown(f"**Projected Impact (Next 30 Days)**")
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Retained Accounts", f"~{retained_customers}", delta=f"{acceptance_rate*100:.0f}% uptake")
            m2.metric("Campaign Cost", f"${cost_of_discount:,.0f}", delta="- discount cost", delta_color="inverse")
            m3.metric("Net Revenue Saved", f"${net_saved_mrr:,.0f}", delta="mrr saved", delta_color="normal")
            
            st.caption(f"*Simulation based on {count_target} customers in '{target_tier}' tier.*")

        # --- RIGHT: MICRO AGENT DESK ---
        with col_micro:
            st.markdown("""
            <div class="onboarding-card" style="border-left-color: #2962FF;">
                <h4>Smart Agent Desk</h4>
                <p style="color:#aaa; font-size:0.8rem; margin-bottom:10px;">
                    Generate personalized scripts for specific at-risk accounts.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Dynamic Selector
            high_risk_ids = scored_df[scored_df['Risk_Tier'] == 'Critical']['customerID'].head(20).tolist()
            
            if not high_risk_ids:
                st.info("No Critical Risk customers found.")
            else:
                selected_cust = st.selectbox("Select Customer ID", high_risk_ids)
                
                # Fetch Customer Data
                cust_data = scored_df[scored_df['customerID'] == selected_cust].iloc[0]
                
                # Dynamic Logic for Script Generation
                pain_points = []
                offer = "Standard Retention Check-in"
                
                if cust_data['Contract'] == 'Month-to-month':
                    pain_points.append("Unstable Contract")
                    offer = "Upgrade to 1-Year (Lock-in Price)"
                if cust_data['PaymentMethod'] == 'Electronic check':
                    pain_points.append("High Friction Payment")
                if cust_data['TechSupport'] == 'No':
                    pain_points.append("Lack of Support Services")
                    
                # The "AI" Script
                script_template = f"""
**Subject:** Exclusive offer for your account {selected_cust}

**Agent Script / Email Draft:**

"Hi there, I noticed you've been with us for {cust_data['tenure']} months. 
We value your loyalty.

I see you are currently on a {cust_data['Contract']} plan. 
To ensure you don't face price fluctuations, I can switch you 
to our **Preferred Tier** today.

**The Offer:** {offer}
**Your Savings:** 15% off your current bill of ${cust_data['MonthlyCharges']}."

**Internal Note:**
User Risk: {cust_data['Churn_Probability']:.1%}
Pain Points: {', '.join(pain_points)}
                """
                
                st.text_area("Generated Script", script_template, height=250)
                
                if st.button("Copy to Clipboard", key="btn_copy"):
                    st.toast("Script copied to clipboard!", icon="📋")