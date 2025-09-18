# Hugging Face Spaces version - uses OpenAI instead of Ollama
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import os
from pathlib import Path

# Import your existing modules
from utils.loader import load_sheet_csv
from data_sources.google_links import SHEET_LINKS
from utils.metrics import calculate_behavior_metrics
from utils.kpi_charts import generate_kpi_charts

# Load environment variables
load_dotenv()

st.set_page_config(
    page_title="Customer Intelligence Engine", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1>📊 Customer Intelligence & Automation Engine</h1><p>AI-Powered Lead Analysis & Business Intelligence Platform</p></div>', unsafe_allow_html=True)

# Note about AI functionality
st.info("🤖 **Note**: This demo version uses simplified AI analysis. For full LLM capabilities, deploy on a platform that supports Ollama.")

# Your existing app logic here (simplified for HF Spaces)
st.markdown("## 📄 Upload Your Data")

uploaded_file = st.file_uploader("Upload your lead data (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.success(f"✅ Loaded {len(df)} records successfully!")
        
        # Show data preview
        st.markdown("### 📊 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Basic analytics
        if len(df) > 0:
            st.markdown("### 📈 Quick Analytics")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Records", f"{len(df):,}")
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                numeric_cols = len(df.select_dtypes(include=['number']).columns)
                st.metric("Numeric Columns", numeric_cols)
            with col4:
                text_cols = len(df.select_dtypes(include=['object']).columns)
                st.metric("Text Columns", text_cols)
            
            # Simple visualizations
            if 'Stage' in df.columns:
                st.markdown("### 🎯 Lead Distribution by Stage")
                fig, ax = plt.subplots(figsize=(10, 6))
                stage_counts = df['Stage'].value_counts()
                stage_counts.plot(kind='bar', ax=ax, color='skyblue')
                ax.set_title('Lead Distribution by Stage')
                ax.set_xlabel('Stage')
                ax.set_ylabel('Count')
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            
            if 'Source' in df.columns:
                st.markdown("### 📊 Lead Distribution by Source")
                fig, ax = plt.subplots(figsize=(10, 6))
                source_counts = df['Source'].value_counts().head(10)
                source_counts.plot(kind='bar', ax=ax, color='lightcoral')
                ax.set_title('Top 10 Lead Sources')
                ax.set_xlabel('Source')
                ax.set_ylabel('Count')
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
        
    except Exception as e:
        st.error(f"❌ Error loading file: {e}")

else:
    st.markdown("""
    ### 🚀 **Welcome to Customer Intelligence Engine!**
    
    This powerful platform helps you:
    - 📊 **Analyze lead data** with AI-powered insights
    - 🎯 **Identify high-potential leads** using behavioral patterns
    - 📈 **Track KPIs** and conversion metrics
    - 🔍 **Filter and export** data based on complex criteria
    
    **To get started:**
    1. Upload your lead data (CSV or Excel format)
    2. Explore the interactive analytics
    3. Use AI-powered insights to optimize your sales process
    
    **Sample columns your data might include:**
    - Lead ID, Stage, Source, Time Spent, Click Count, Page Depth
    - Contact information, Lead scoring, Conversion status
    """)

# Footer
st.markdown("---")
st.markdown("*Built with ❤️ using Streamlit • Powered by AI for Better Lead Intelligence*")