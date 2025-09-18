import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import os
from pathlib import Path
from behavior_match import match_not_interested_to_closure_behavior

from utils.loader import load_sheet_csv
from data_sources.google_links import SHEET_LINKS
from utils.metrics import calculate_behavior_metrics
from utils.kpi_charts import generate_kpi_charts
from llm_interface import run_llm_query_on_df, run_llm_query_and_return_csv, handle_llm_request, show_available_columns, handle_intelligent_request, prompt_suggestion_ui

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
.analysis-container {
    background-color: #f8f9fa;
    padding: 20px;
    border-radius: 10px;
    border-left: 5px solid #007bff;
    margin: 10px 0;
}
.metric-card {
    background-color: white;
    padding: 15px;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin: 10px 0;
}
.quick-action-btn {
    width: 100%;
    margin: 5px 0;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1>📊 Customer Intelligence & Automation Engine</h1></div>', unsafe_allow_html=True)

# Sidebar for Google Sheet selection
st.sidebar.header("🔧 Data Configuration")
selected_project = st.sidebar.selectbox("Select Project", list(SHEET_LINKS.keys()))

if "base_df" not in st.session_state:
    st.session_state.base_df = None

if st.sidebar.button("🚀 Load Sheet", use_container_width=True):
    with st.spinner("Loading base data..."):
        base_df = load_sheet_csv(SHEET_LINKS[selected_project])
        if not base_df.empty:
            for col in base_df.columns:
                if base_df[col].dtype == 'object':
                    base_df[col] = base_df[col].astype(str).str.strip()
                    base_df[col] = pd.to_numeric(base_df[col], errors='ignore')
            st.session_state.base_df = base_df
            st.sidebar.success(f"✅ {len(base_df)} rows loaded")
        else:
            st.sidebar.error("❌ Sheet load failed")

# Upload web events file
st.sidebar.markdown("### 📄 Upload Web Events")
uploaded_file = st.sidebar.file_uploader("Upload Web Events file", type=["xlsx", "csv"])

if uploaded_file and st.session_state.base_df is not None:
    with st.spinner("Processing data..."):
        try:
            if uploaded_file.name.endswith(".csv"):
                web_df = pd.read_csv(uploaded_file)
            else:
                web_df = pd.read_excel(uploaded_file)

            web_df.columns = web_df.columns.str.strip().str.lower()
            rename_map = {
                "masterleadid": "masterLeadId",
                "totaltimespent": "TotalTimeSpent",
                "length-click_events": "ClickCount",
                "length-viewable_events": "PageDepth"
            }
            web_df.rename(columns=rename_map, inplace=True)

            base_df = st.session_state.base_df.copy()
            base_df['masterLeadId'] = base_df['masterLeadId'].astype(str)
            web_df['masterLeadId'] = web_df['masterLeadId'].astype(str)

            merged_df = pd.merge(base_df, web_df, on="masterLeadId", how="left")
            st.session_state.merged_df = merged_df

            behavior_df = web_df[['masterLeadId', 'TotalTimeSpent', 'ClickCount', 'PageDepth']]
            enriched_df = pd.merge(base_df, behavior_df, on="masterLeadId", how="left")
            st.session_state.enriched_df = enriched_df

            st.sidebar.success("✅ Data merged successfully!")
            
            # Add data previews
            st.markdown("### 🧲 Data Sheets Preview")
            with st.expander("🔹 Basic Sheet (Google Sheet)"):
                st.dataframe(base_df.head(100), use_container_width=True)
            with st.expander("🔹 Web Events Sheet (Uploaded File)"):
                st.dataframe(web_df.head(100), use_container_width=True)
            with st.expander("🔹 Merged View"):
                st.dataframe(merged_df.head(100), use_container_width=True)

        except Exception as e:
            st.sidebar.error(f"❌ Error: {e}")

# Add Advanced Filters in Sidebar
if "enriched_df" in st.session_state:
    st.sidebar.markdown("---")
    st.sidebar.header("🎛️ Advanced Filters")
    
    filtered_df = st.session_state.enriched_df.copy()
    
    # Time Spent Filter
    if 'TotalTimeSpent' in filtered_df.columns:
        max_time = int(filtered_df['TotalTimeSpent'].max()) if filtered_df['TotalTimeSpent'].max() > 0 else 100
        time_filter = st.sidebar.slider("Time Spent (sec)", 0, max_time, (0, max_time), key="time_filter")
        filtered_df = filtered_df[(filtered_df['TotalTimeSpent'] >= time_filter[0]) & (filtered_df['TotalTimeSpent'] <= time_filter[1])]

    # Click Count Filter
    if 'ClickCount' in filtered_df.columns:
        max_clicks = int(filtered_df['ClickCount'].max()) if filtered_df['ClickCount'].max() > 0 else 50
        click_filter = st.sidebar.slider("Click Count", 0, max_clicks, (0, max_clicks), key="click_filter")
        filtered_df = filtered_df[(filtered_df['ClickCount'] >= click_filter[0]) & (filtered_df['ClickCount'] <= click_filter[1])]

    # Page Depth Filter
    if 'PageDepth' in filtered_df.columns:
        max_depth = int(filtered_df['PageDepth'].max()) if filtered_df['PageDepth'].max() > 0 else 100
        page_depth_filter = st.sidebar.slider("Page Depth", 0, max_depth, (0, max_depth), key="page_depth_filter")
        filtered_df = filtered_df[(filtered_df['PageDepth'] >= page_depth_filter[0]) & (filtered_df['PageDepth'] <= page_depth_filter[1])]

    # Orange Fields Filters
    orange_fields = ['SFT', 'Budget', 'Floor', 'Handover', 'Site Visit Preference']
    for field in orange_fields:
        if field in filtered_df.columns:
            key_suffix = field.replace(" ", "_").lower()
            unique_values = filtered_df[field].dropna().unique()
            if len(unique_values) > 0:
                selected = st.sidebar.multiselect(f"{field}", unique_values, key=f"orange_{key_suffix}")
                if selected:
                    filtered_df = filtered_df[filtered_df[field].isin(selected)]

    # Source Filter
    if 'Source' in filtered_df.columns:
        unique_sources = filtered_df['Source'].dropna().unique()
        if len(unique_sources) > 0:
            selected_sources = st.sidebar.multiselect("Source", unique_sources, key="source_filter")
            if selected_sources:
                filtered_df = filtered_df[filtered_df['Source'].isin(selected_sources)]

    # NI Reason Filter
    if 'NI Reason' in filtered_df.columns:
        unique_ni = filtered_df['NI Reason'].dropna().unique()
        if len(unique_ni) > 0:
            selected_ni = st.sidebar.multiselect("NI Reason", unique_ni, key="ni_filter")
            if selected_ni:
                filtered_df = filtered_df[filtered_df['NI Reason'].isin(selected_ni)]

    # Call Duration Filter
    if 'Call Duration' in filtered_df.columns:
        max_call = int(filtered_df['Call Duration'].max()) if filtered_df['Call Duration'].max() > 0 else 300
        call_filter = st.sidebar.slider("Call Duration (sec)", 0, max_call, (0, max_call), key="call_duration")
        filtered_df = filtered_df[(filtered_df['Call Duration'] >= call_filter[0]) & (filtered_df['Call Duration'] <= call_filter[1])]

    # Store filtered data in session state
    st.session_state.filtered_df = filtered_df
    
    # Show filtered results summary in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Filtered Results")
    st.sidebar.write(f"**{len(filtered_df)}** leads match your filters")
    
    # Show filtered data preview
    if len(filtered_df) > 0:
        with st.expander("📊 Filtered Data View"):
            st.write(f"{len(filtered_df)} leads match your filters.")
            st.dataframe(filtered_df.head(20), use_container_width=True)

# Add KPI Visualizations Section
if 'enriched_df' in st.session_state:
    st.markdown("## 📈 KPI Visualizations")
    kpi_df = st.session_state['enriched_df']

    # Create columns for KPI charts
    kpi_col1, kpi_col2 = st.columns(2)
    
    with kpi_col1:
        if 'TotalTimeSpent' in kpi_df.columns:
            st.markdown("### ⏱ Time Spent Distribution")
            fig, ax = plt.subplots(figsize=(8, 5))
            time_data = kpi_df['TotalTimeSpent'].dropna()
            if len(time_data) > 0:
                sns.histplot(time_data, bins=30, kde=True, ax=ax)
                ax.set_xlabel("Total Time Spent (sec)")
                ax.set_ylabel("Number of Leads")
                ax.set_title("Distribution of Total Time Spent")
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("No time spent data available")

        if {'PageDepth', 'ClickCount'}.issubset(kpi_df.columns):
            st.markdown("### 🧠 Page Depth vs Click Behavior")
            fig, ax = plt.subplots(figsize=(8, 5))
            scatter_data = kpi_df[['PageDepth', 'ClickCount']].dropna()
            if len(scatter_data) > 0:
                sns.scatterplot(x='PageDepth', y='ClickCount', data=scatter_data, ax=ax)
                ax.set_title("Page Depth vs Click Count")
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("No behavioral data available")

    with kpi_col2:
        if 'ClickCount' in kpi_df.columns:
            st.markdown("### 💚 Click Count Distribution")
            fig, ax = plt.subplots(figsize=(8, 5))
            click_data = kpi_df['ClickCount'].dropna()
            if len(click_data) > 0:
                # Limit to reasonable range for better visualization
                click_data_limited = click_data[click_data <= 50]
                if len(click_data_limited) > 0:
                    value_counts = click_data_limited.value_counts().head(20)
                    fig, ax = plt.subplots(figsize=(8, 5))
                    value_counts.plot(kind='bar', ax=ax)
                    ax.set_title("Click Count per Lead")
                    ax.set_xlabel("Clicks")
                    ax.set_ylabel("Lead Count")
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.info("No click data in reasonable range")
            else:
                st.info("No click data available")

        if {'Stage', 'TotalTimeSpent'}.issubset(kpi_df.columns):
            st.markdown("### 🌟 Time Spent by Stage")
            fig, ax = plt.subplots(figsize=(8, 5))
            stage_time_data = kpi_df[['Stage', 'TotalTimeSpent']].dropna()
            if len(stage_time_data) > 0:
                sns.boxplot(x='Stage', y='TotalTimeSpent', data=stage_time_data, ax=ax)
                ax.set_title("Total Time Spent by Lead Stage")
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("No stage/time data available")

    # KPI Prompt Section
    st.markdown("### 🔍 Custom KPI Analysis")
    kpi_prompt = st.text_input("Enter your KPI instruction", placeholder="e.g., Show me conversion rates by source")
    if st.button("Generate KPI Chart") and kpi_prompt:
        if 'merged_df' in st.session_state:
            generate_kpi_charts(st.session_state.merged_df, kpi_prompt)
        else:
            st.error("Please load and merge data first")

# Main AI Assistant Interface
if "merged_df" in st.session_state:
    st.markdown("## 🧠 **Super Intelligent AI Assistant**")
    st.markdown("*Ask me anything about your data - I'll provide comprehensive analysis with explanations and logic!*")
    
    # Create improved two-column layout
    left_col, right_col = st.columns([1, 2.5])
    
    with left_col:
        st.markdown("### 🎯 **Ask Your Question:**")
        user_query = st.text_area(
            "Type your question in natural language...", 
            placeholder="Examples:\n• Analyse sales closure leads and give me Not Interested leads with potential\n• Show me a chart of engagement by source\n• Which leads are most likely to convert?\n• Calculate conversion rates by stage",
            height=120,
            key="main_query"
        )
        
        # Action buttons
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            analyze_clicked = st.button("🚀 **Analyze**", type="primary", use_container_width=True)
        with col_btn2:
            if st.button("🔄 **Clear**", use_container_width=True):
                if 'conversation_history' in st.session_state:
                    st.session_state.conversation_history = []
                    st.success("History cleared!")
        
        st.markdown("---")
        st.markdown("### 💡 **Quick Actions:**")
        
        quick_actions = [
            ("📊 Lead Distribution", "Show me the distribution of leads across all stages with counts and percentages"),
            ("🎯 High-Potential Leads", "Analyse sales closure leads and give me Not Interested leads with potential to close sales"),
            ("📈 Source Performance", "Which sources are performing best and show me engagement metrics by source"),
            ("🔍 Behavioral Patterns", "Show me behavioral patterns comparing high-intent vs low-intent leads"),
            ("📋 Data Summary", "Give me a comprehensive summary of all the data with key insights"),
            ("💰 Conversion Analysis", "Calculate conversion rates and show me the sales funnel")
        ]
        
        for label, query in quick_actions:
            if st.button(label, use_container_width=True, key=f"quick_{label}"):
                handle_intelligent_request(st.session_state.merged_df, query)
        
        # Show conversation history
        if 'conversation_history' in st.session_state and st.session_state.conversation_history:
            st.markdown("---")
            st.markdown("### 💬 **Recent Queries:**")
            for i, conv in enumerate(reversed(st.session_state.conversation_history[-3:]), 1):
                with st.expander(f"Query {i}: {conv['response_type']}"):
                    st.write(f"**Q:** {conv['query'][:80]}...")
        
        # Collapsible sections for reference
        with st.expander("📋 **Data Columns Reference**"):
            if st.session_state.merged_df is not None:
                cols_info = []
                for col in st.session_state.merged_df.columns[:10]:  # Show first 10 columns
                    dtype = str(st.session_state.merged_df[col].dtype)
                    cols_info.append(f"• **{col}** ({dtype})")
                st.markdown("\n".join(cols_info))
    
    with right_col:
        st.markdown("### 📊 **Analysis Results:**")
        
        # Results container with better styling
        if analyze_clicked and user_query:
            with st.container():
                handle_intelligent_request(st.session_state.merged_df, user_query)
        else:
            # Show attractive placeholder content
            st.markdown("""
            <div class="analysis-container">
                <h4>👆 Enter your question and click 'Analyze' for intelligent insights!</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Show quick data overview
            if st.session_state.merged_df is not None:
                st.markdown("#### 📈 **Quick Data Overview:**")
                
                # Create attractive metrics
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                
                with metric_col1:
                    total_leads = len(st.session_state.merged_df)
                    st.metric("Total Leads", f"{total_leads:,}")
                
                with metric_col2:
                    if 'Stage' in st.session_state.merged_df.columns:
                        unique_stages = st.session_state.merged_df['Stage'].nunique()
                        st.metric("Unique Stages", unique_stages)
                
                with metric_col3:
                    if 'Source' in st.session_state.merged_df.columns:
                        unique_sources = st.session_state.merged_df['Source'].nunique()
                        st.metric("Unique Sources", unique_sources)
                
                with metric_col4:
                    if 'TotalTimeSpent' in st.session_state.merged_df.columns:
                        avg_time = st.session_state.merged_df['TotalTimeSpent'].mean()
                        st.metric("Avg Time Spent", f"{avg_time:.1f}s")
                
                # Show top stages in a nice format
                if 'Stage' in st.session_state.merged_df.columns:
                    st.markdown("#### 🎯 **Lead Stage Distribution:**")
                    stage_counts = st.session_state.merged_df['Stage'].value_counts().head(5)
                    
                    # Create a simple bar chart
                    fig, ax = plt.subplots(figsize=(10, 4))
                    stage_counts.plot(kind='bar', ax=ax, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
                    ax.set_title('Top 5 Lead Stages')
                    ax.set_xlabel('Stage')
                    ax.set_ylabel('Count')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
                
                st.markdown("---")
                st.markdown("### 💡 **Try asking questions like:**")
                example_questions = [
                    "Show me Not Interested leads with high potential",
                    "Which sources convert best?",
                    "Create a chart of engagement patterns",
                    "Calculate conversion rates by stage",
                    "Find leads similar to successful closures"
                ]
                
                for question in example_questions:
                    st.markdown(f"• *'{question}'*")

else:
    # Show data upload instructions
    st.info("ℹ️ Please load a Google Sheet and upload a Web Events file to start using the AI Assistant.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📋 **Step 1: Load Google Sheet**")
        st.markdown("1. Select a project from the sidebar")
        st.markdown("2. Click 'Load Sheet' button")
        st.markdown("3. Wait for data to load")
    
    with col2:
        st.markdown("### 📄 **Step 2: Upload Web Events**")
        st.markdown("1. Upload your .xlsx or .csv file")
        st.markdown("2. Ensure it has required columns")
        st.markdown("3. Data will be automatically merged")

# Add Behavioral Matching Section
if "merged_df" in st.session_state:
    st.markdown("## 🎯 Behavioral Lead Matching")
    st.markdown("*Automatically find Not Interested leads that behave like successful closures*")
    
    if st.button("🔍 Find Similar Leads", type="secondary", use_container_width=True):
        with st.spinner("Analyzing behavioral patterns..."):
            merged_df = st.session_state.merged_df
            similar_leads_df = match_not_interested_to_closure_behavior(merged_df)

            if not similar_leads_df.empty:
                st.success(f"✅ Found {len(similar_leads_df)} Not Interested leads with closure behavior patterns!")
                st.dataframe(similar_leads_df, use_container_width=True)
                st.download_button("📥 Download Matched Leads CSV", 
                                 data=similar_leads_df.to_csv(index=False),
                                 file_name="matched_leads.csv", 
                                 mime="text/csv",
                                 use_container_width=True)
            else:
                st.warning("⚠️ No behavioral matches found. This could mean:")
                st.markdown("• No 'Sales Closure' leads to analyze patterns from")
                st.markdown("• No 'Not Interested' leads to match against")
                st.markdown("• Behavioral data (time spent, clicks, page depth) is missing")

# Footer
st.markdown("---")
st.markdown("*Built with ❤️ by CI Team for better lead intelligence*")