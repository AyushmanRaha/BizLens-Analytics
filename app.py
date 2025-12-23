import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# 1. Page Configuration
st.set_page_config(
    page_title="BizLens Analytics",
    page_icon="📊",
    layout="wide"
)

# 2. Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Data Explorer", "Settings"])

# 3. Main Header
st.title("BizLens-Analytics Dashboard")
st.markdown("---")

if page == "Dashboard":
    # Create some dummy data for testing
    chart_data = pd.DataFrame(
        np.random.randn(20, 3),
        columns=['Sales', 'Profit', 'Expenses']
    )

    # 4. KPI Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Revenue", "$45,200", "+12%")
    col2.metric("Active Users", "1,240", "-5%")
    col3.metric("Growth Rate", "24%", "+2%")

    st.markdown("### Performance Overview")
    
    # 5. Interactive Plotly Chart
    fig = px.line(chart_data, title="Business Metrics Over Time")
    st.plotly_chart(fig, use_container_view_width=True)

elif page == "Data Explorer":
    st.subheader("Upload Business Data")
    uploaded_file = st.file_file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Data Preview:", df.head())
        st.write("Summary Statistics:", df.describe())
    else:
        st.info("Please upload a CSV file to begin analysis.")

# 6. Footer
st.sidebar.markdown("---")
st.sidebar.write("Logged in as: Admin")