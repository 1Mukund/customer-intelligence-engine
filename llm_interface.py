import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import ollama
import io
import json
import numpy as np
from datetime import datetime

# ---------------------------
# ✅ Built-in Prompt Suggestions
# ---------------------------
SUGGESTED_PROMPTS = [
    # Analysis Questions
    "What are the key insights from this lead data?",
    "Which sources are performing best and why?",
    "What patterns do you see in lead behavior across different stages?",
    "How do high-intent leads differ from low-intent leads?",
    "What factors correlate with successful lead conversion?",
    
    # Visualization Requests
    "Show me a chart comparing lead stages by source",
    "Create a scatter plot of time spent vs click count",
    "Plot the distribution of leads across different stages",
    "Show me engagement patterns by source",
    
    # Data Filtering
    "Get me all high-intent leads from the top 3 sources",
    "Show leads with more than 5 clicks and over 60 seconds time spent",
    "Filter leads that are in RI stage with high engagement",
    "Export leads that match closure behavior patterns",
    
    # Business Intelligence
    "Calculate conversion rates by source and stage",
    "What's the average engagement time by lead quality?",
    "Identify leads most likely to convert based on behavior",
    "Compare performance metrics across different campaigns"
]

# ---------------------------
# ✅ FAST Chart Generator
# ---------------------------
def generate_chart_from_prompt(df, prompt):
    try:
        # Handle specific chart requests directly without LLM
        prompt_lower = prompt.lower()
        
        # Rural leads by source
        if "rural" in prompt_lower and "source" in prompt_lower:
            create_rural_source_chart(df)
            return
        
        # Source vs Stage analysis
        if "source" in prompt_lower and ("stage" in prompt_lower or "vs" in prompt_lower):
            create_source_stage_chart(df)
            return
        
        # Stage distribution
        if "stage" in prompt_lower and "Stage" in df.columns:
            st.info("📊 Creating Stage distribution chart...")
            fig, ax = plt.subplots(figsize=(10, 6))
            stage_counts = df['Stage'].value_counts()
            stage_counts.plot(kind='bar', ax=ax, color='skyblue')
            ax.set_title('Lead Distribution by Stage')
            ax.set_xlabel('Stage')
            ax.set_ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            return
        
        # Source distribution
        if "source" in prompt_lower and "Source" in df.columns:
            st.info("📊 Creating Source distribution chart...")
            fig, ax = plt.subplots(figsize=(12, 6))
            source_counts = df['Source'].value_counts().head(15)
            source_counts.plot(kind='bar', ax=ax, color='lightcoral')
            ax.set_title('Lead Distribution by Source')
            ax.set_xlabel('Source')
            ax.set_ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            return
        
        # Behavioral charts
        if any(col in prompt_lower for col in ['time', 'click', 'page']) and any(col in df.columns for col in ['TotalTimeSpent', 'ClickCount', 'PageDepth']):
            create_behavioral_chart(df, prompt_lower)
            return
        
        # Fallback to LLM for complex requests
        columns_list = list(df.columns)
        
        # Create more specific prompt with actual column values
        column_info = []
        for col in columns_list[:10]:
            if df[col].dtype == 'object':
                unique_vals = df[col].value_counts().head(3).index.tolist()
                column_info.append(f"{col}: {unique_vals}")
            else:
                column_info.append(f"{col}: numeric")
        
        system_message = f"""Generate Python matplotlib code for a chart. DataFrame is 'df'.

AVAILABLE COLUMNS AND SAMPLE VALUES:
{chr(10).join(column_info)}

RULES:
- Use ONLY columns that exist in the data above
- Use matplotlib.pyplot as plt and seaborn as sns
- Create figure with: fig, ax = plt.subplots(figsize=(10, 6))
- End with: plt.tight_layout()
- NO plt.show()
- NO explanations or comments

USER REQUEST: {prompt}

Generate the Python code:"""

        response = ollama.chat(
            model="llama3",
            messages=[{"role": "user", "content": system_message}],
            options={"temperature": 0.1}
        )

        code = response["message"]["content"].strip()

        # Clean the code
        if "```" in code:
            code_blocks = code.split("```")
            for block in code_blocks:
                if any(keyword in block for keyword in ["plt.", "sns.", "df["]):
                    code = block.strip()
                    break
        
        # Remove markdown and comments
        lines = [line.strip() for line in code.split('\n') 
                if line.strip() and not line.strip().startswith('#') 
                and not line.strip().startswith('```')
                and not line.strip().startswith('python')]
        
        code = '\n'.join(lines)
        
        # Add required imports if missing
        if 'import matplotlib.pyplot as plt' not in code:
            code = 'import matplotlib.pyplot as plt\nimport seaborn as sns\n' + code
        
        # Execute with error handling
        local_vars = {"df": df, "plt": plt, "sns": sns, "pd": pd, "np": np}
        exec(code, {}, local_vars)
        
        fig = plt.gcf()
        if fig.get_axes():
            st.pyplot(fig)
        else:
            raise Exception("No chart was generated")
            
    except Exception as e:
        # Smart fallback based on available data
        create_smart_fallback_chart(df, prompt)

def create_rural_source_chart(df):
    """Create chart for rural leads by source"""
    st.info("📊 Analyzing rural leads by source...")
    
    # Look for location columns
    location_columns = [col for col in df.columns if any(keyword in col.lower() 
                       for keyword in ['location', 'area', 'city', 'region', 'zone'])]
    
    if not location_columns or 'Source' not in df.columns:
        st.warning("⚠️ Cannot create rural vs source chart - missing location or source data")
        create_smart_fallback_chart(df, "source distribution")
        return
    
    # Try to identify rural leads
    for loc_col in location_columns:
        rural_mask = df[loc_col].astype(str).str.contains('rural|village|town', case=False, na=False)
        rural_leads = df[rural_mask]
        
        if len(rural_leads) > 0:
            fig, ax = plt.subplots(figsize=(12, 6))
            rural_by_source = rural_leads['Source'].value_counts()
            rural_by_source.plot(kind='bar', ax=ax, color='green')
            ax.set_title(f'Rural Leads by Source (from {loc_col})')
            ax.set_xlabel('Source')
            ax.set_ylabel('Number of Rural Leads')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            return
    
    st.warning("⚠️ No rural indicators found in location data")
    create_smart_fallback_chart(df, "source distribution")

def create_source_stage_chart(df):
    """Create chart for source vs stage analysis"""
    if 'Source' not in df.columns or 'Stage' not in df.columns:
        st.warning("⚠️ Missing Source or Stage columns")
        create_smart_fallback_chart(df, "available data")
        return
    
    st.info("📊 Creating Source vs Stage analysis...")
    
    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create cross-tabulation
    cross_tab = pd.crosstab(df['Source'], df['Stage'])
    cross_tab.plot(kind='bar', ax=ax, stacked=True)
    
    ax.set_title('Lead Distribution: Source vs Stage')
    ax.set_xlabel('Source')
    ax.set_ylabel('Number of Leads')
    plt.xticks(rotation=45)
    plt.legend(title='Stage', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    st.pyplot(fig)

def create_behavioral_chart(df, prompt_lower):
    """Create behavioral analysis charts"""
    behavioral_cols = [col for col in ['TotalTimeSpent', 'ClickCount', 'PageDepth'] if col in df.columns]
    
    if not behavioral_cols:
        st.warning("⚠️ No behavioral data available")
        return
    
    st.info("📊 Creating behavioral analysis chart...")
    
    if len(behavioral_cols) >= 2:
        # Scatter plot for two behavioral metrics
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(df[behavioral_cols[0]], df[behavioral_cols[1]], alpha=0.6)
        ax.set_xlabel(behavioral_cols[0])
        ax.set_ylabel(behavioral_cols[1])
        ax.set_title(f'{behavioral_cols[0]} vs {behavioral_cols[1]}')
        plt.tight_layout()
        st.pyplot(fig)
    else:
        # Histogram for single behavioral metric
        fig, ax = plt.subplots(figsize=(10, 6))
        df[behavioral_cols[0]].hist(bins=30, ax=ax, alpha=0.7)
        ax.set_xlabel(behavioral_cols[0])
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of {behavioral_cols[0]}')
        plt.tight_layout()
        st.pyplot(fig)

def create_smart_fallback_chart(df, prompt):
    """Create intelligent fallback chart based on available data"""
    st.warning("⚠️ Creating fallback chart with available data...")
    
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Priority order for fallback charts
        if 'Stage' in df.columns:
            df['Stage'].value_counts().head(10).plot(kind='bar', ax=ax, color='lightblue')
            ax.set_title('Lead Distribution by Stage')
            ax.set_xlabel('Stage')
        elif 'Source' in df.columns:
            df['Source'].value_counts().head(10).plot(kind='bar', ax=ax, color='lightcoral')
            ax.set_title('Lead Distribution by Source')
            ax.set_xlabel('Source')
        elif len(df.select_dtypes(include=['object']).columns) > 0:
            first_cat_col = df.select_dtypes(include=['object']).columns[0]
            df[first_cat_col].value_counts().head(10).plot(kind='bar', ax=ax)
            ax.set_title(f'Distribution of {first_cat_col}')
            ax.set_xlabel(first_cat_col)
        else:
            ax.text(0.5, 0.5, 'No suitable categorical data for visualization', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Data Overview')
        
        ax.set_ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        
    except Exception as e:
        st.error("❌ Could not create any visualization with the available data.")

# ---------------------------
# ✅ Smart CSV Generator from Prompt
# ---------------------------
def run_llm_query_and_return_csv(df, user_prompt):
    try:
        # Get available columns and sample data
        columns_list = list(df.columns)
        
        system_message = f"""You are a Python code generator. Generate ONLY Python code that filters a DataFrame called 'df' and assigns the result to a variable called 'result'.

Available columns: {columns_list}

Rules:
1. Write ONLY executable Python code
2. Use ONLY columns from the list above
3. Always assign filtered DataFrame to variable 'result'
4. Do NOT include any explanations, comments, or markdown
5. Do NOT use quotes around column names unless they contain spaces
6. Use proper pandas syntax: df[df['column'] == 'value']

Example format:
result = df[df['Stage'] == 'RI']

User request: {user_prompt}

Generate the Python code:"""

        response = ollama.chat(
            model="llama3",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt}
            ]
        )

        code = response["message"]["content"].strip()

        # More robust code cleaning
        if "```" in code:
            # Extract code between triple backticks
            parts = code.split("```")
            for part in parts:
                if "result =" in part or "df[" in part:
                    code = part.strip()
                    break
        
        # Remove any remaining markdown or explanatory text
        lines = code.split('\n')
        clean_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('```') and ('result =' in line or 'df[' in line):
                clean_lines.append(line)
        
        if clean_lines:
            code = '\n'.join(clean_lines)
        
        # Basic validation - ensure code looks safe
        if not code or 'result =' not in code:
            raise ValueError("Generated code doesn't contain proper result assignment")
        
        # Check for dangerous operations
        dangerous_keywords = ['import', 'exec', 'eval', 'open', 'file', '__']
        if any(keyword in code.lower() for keyword in dangerous_keywords):
            raise ValueError("Generated code contains potentially unsafe operations")

        local_vars = {"df": df, "pd": pd}
        exec(code, {}, local_vars)
        result = local_vars.get("result")

        if isinstance(result, pd.DataFrame):
            if len(result) > 0:
                st.success(f"✅ Found {len(result)} matching records:")
                st.dataframe(result.head(100), use_container_width=True)
                st.download_button("📅 Download Full Results as CSV", result.to_csv(index=False), "filtered_data.csv", "text/csv")
            else:
                st.warning("⚠️ No records match your criteria. Try adjusting your filters.")
        else:
            st.warning("⚠️ Could not process your request. Please try rephrasing.")
    except Exception as e:
        st.error(f"❌ Could not filter the data as requested. Please try rephrasing your request or check column names.")

# ---------------------------
# ✅ SUPER INTELLIGENT DATA ANALYZER
# ---------------------------
def analyze_data_comprehensively(df):
    """Generate comprehensive data analysis"""
    analysis = {}
    
    # Basic stats
    analysis['total_rows'] = len(df)
    analysis['total_columns'] = len(df.columns)
    analysis['columns'] = list(df.columns)
    
    # Numeric columns analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    analysis['numeric_columns'] = {}
    for col in numeric_cols:
        analysis['numeric_columns'][col] = {
            'mean': df[col].mean(),
            'median': df[col].median(),
            'min': df[col].min(),
            'max': df[col].max(),
            'std': df[col].std(),
            'null_count': df[col].isnull().sum()
        }
    
    # Categorical columns analysis
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    analysis['categorical_columns'] = {}
    for col in categorical_cols:
        value_counts = df[col].value_counts().head(10)
        analysis['categorical_columns'][col] = {
            'unique_values': df[col].nunique(),
            'top_values': value_counts.to_dict(),
            'null_count': df[col].isnull().sum()
        }
    
    return analysis

def perform_actual_data_analysis(df, user_prompt):
    """Perform actual data analysis and calculations on the real data"""
    
    # Check if this is about lead analysis
    if any(word in user_prompt.lower() for word in ['not interested', 'closure', 'sales', 'convert', 'potential']):
        return analyze_lead_conversion_patterns(df, user_prompt)
    else:
        return run_general_data_analysis(df, user_prompt)

# ---------------------------
# ✅ COMPREHENSIVE Lead Analysis
# ---------------------------
def analyze_lead_conversion_patterns(df, user_prompt):
    """Fast lead conversion analysis"""
    try:
        st.markdown("## 🎯 **Lead Conversion Analysis**")
        
        if 'Stage' not in df.columns:
            st.error("Missing 'Stage' column")
            return
        
        # Quick stats
        stage_counts = df['Stage'].value_counts()
        total_leads = len(df)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Leads", f"{total_leads:,}")
        with col2:
            closure_leads = stage_counts.get('Sales Closure', 0)
            st.metric("Sales Closure", f"{closure_leads:,}")
        with col3:
            ni_leads = stage_counts.get('Not Interested', 0)
            st.metric("Not Interested", f"{ni_leads:,}")
        
        # Stage distribution
        st.markdown("### 📈 **Stage Distribution:**")
        stage_df = pd.DataFrame({
            'Stage': stage_counts.index,
            'Count': stage_counts.values,
            'Percentage': (stage_counts.values / total_leads * 100).round(1)
        })
        st.dataframe(stage_df, use_container_width=True)
        
        # Behavioral analysis if data exists
        behavioral_cols = ['TotalTimeSpent', 'ClickCount', 'PageDepth']
        available_cols = [col for col in behavioral_cols if col in df.columns]
        
        if available_cols and closure_leads > 0 and ni_leads > 0:
            st.markdown("### 🧠 **Behavioral Analysis:**")
            
            closure_df = df[df['Stage'] == 'Sales Closure']
            ni_df = df[df['Stage'] == 'Not Interested']
            
            # Quick comparison
            comparison = {}
            for col in available_cols:
                comparison[col] = {
                    'Closure Avg': closure_df[col].mean(),
                    'Not Interested Avg': ni_df[col].mean()
                }
            
            comp_df = pd.DataFrame(comparison).T.round(2)
            st.dataframe(comp_df, use_container_width=True)
            
            # Find high-potential leads
            st.markdown("### 🎯 **High-Potential Leads:**")
            
            # Simple criteria: above median of closure leads
            criteria = []
            for col in available_cols:
                threshold = closure_df[col].median()
                criteria.append(f"({col} >= {threshold})")
            
            if criteria:
                query = " & ".join(criteria)
                try:
                    high_potential = ni_df.query(query)
                    
                    if len(high_potential) > 0:
                        st.success(f"✅ Found {len(high_potential)} high-potential leads!")
                        
                        display_cols = ['masterLeadId'] + available_cols
                        if 'Source' in df.columns:
                            display_cols.append('Source')
                        
                        st.dataframe(high_potential[display_cols].head(20), use_container_width=True)
                        
                        st.download_button(
                            "📥 Download High-Potential Leads",
                            high_potential.to_csv(index=False),
                            "high_potential_leads.csv",
                            "text/csv"
                        )
                    else:
                        st.warning("⚠️ No high-potential leads found.")
                        
                except Exception as e:
                    st.error(f"Error filtering: {e}")
        
    except Exception as e:
        st.error(f"❌ Analysis error: {e}")

# ---------------------------
# ✅ COMPREHENSIVE General Analysis with LLM
# ---------------------------
def run_general_data_analysis(df, user_prompt):
    """Run comprehensive data analysis with LLM insights"""
    try:
        # First, let's do ACTUAL data analysis instead of relying on LLM
        st.markdown("## 🧠 **Data Analysis Results**")
        
        # Handle specific queries about rural leads and sources
        if "rural" in user_prompt.lower() and "source" in user_prompt.lower():
            analyze_rural_leads_by_source(df, user_prompt)
            return
        
        # Handle source analysis queries
        if "source" in user_prompt.lower() and any(word in user_prompt.lower() for word in ["vs", "comparison", "compare", "performance"]):
            analyze_source_performance(df, user_prompt)
            return
        
        # Handle stage analysis queries
        if "stage" in user_prompt.lower() and "source" in user_prompt.lower():
            analyze_stage_by_source(df, user_prompt)
            return
        
        # Get comprehensive data analysis
        data_analysis = analyze_data_comprehensively(df)
        
        # Get sample data for context
        sample_data = df.head(5).to_dict('records')
        
        # Create a more focused prompt with actual column names
        available_columns = list(df.columns)
        column_info = []
        for col in available_columns[:20]:  # Limit to first 20 columns
            if df[col].dtype == 'object':
                unique_vals = df[col].value_counts().head(5).index.tolist()
                column_info.append(f"- {col}: {unique_vals}")
            else:
                column_info.append(f"- {col}: numeric (min: {df[col].min()}, max: {df[col].max()})")
        
        system_message = f"""You are analyzing lead/customer data. Answer based ONLY on the actual data provided.

AVAILABLE COLUMNS AND VALUES:
{chr(10).join(column_info)}

TOTAL RECORDS: {len(df)}

IMPORTANT RULES:
1. Use ONLY the column names and values shown above
2. If asked about "rural" leads, look for columns that might contain location/area data
3. If asked about sources, use the 'Source' column if it exists
4. Never mention columns that don't exist in the data
5. Provide specific numbers from the actual data
6. If you can't find relevant data for the question, say so clearly

USER QUESTION: {user_prompt}

Analyze the data and provide specific insights:"""

        response = ollama.chat(
            model="llama3",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt}
            ]
        )

        answer = response["message"]["content"]
        st.success("🧠 **Analysis Results:**")
        st.markdown(answer)
        
        # Add actual data insights
        st.markdown("---")
        st.markdown("### 📊 **Actual Data Insights:**")
        
        # Show relevant data based on query
        if "source" in user_prompt.lower() and 'Source' in df.columns:
            st.markdown("**Source Distribution:**")
            source_counts = df['Source'].value_counts().head(10)
            st.dataframe(source_counts.to_frame('Count'), use_container_width=True)
        
        if "stage" in user_prompt.lower() and 'Stage' in df.columns:
            st.markdown("**Stage Distribution:**")
            stage_counts = df['Stage'].value_counts()
            st.dataframe(stage_counts.to_frame('Count'), use_container_width=True)

    except Exception as e:
        st.error(f"❌ Could not analyze the data. Error: {str(e)}")

def analyze_rural_leads_by_source(df, user_prompt):
    """Specific analysis for rural leads by source"""
    st.markdown("### 🏘️ **Rural Leads Analysis by Source**")
    
    # Look for columns that might contain rural/location data
    location_columns = [col for col in df.columns if any(keyword in col.lower() 
                       for keyword in ['location', 'area', 'city', 'region', 'zone', 'rural', 'urban'])]
    
    if not location_columns:
        st.warning("⚠️ No location/area columns found in the data to identify rural leads.")
        st.info("Available columns: " + ", ".join(df.columns.tolist()))
        return
    
    # Check if Source column exists
    if 'Source' not in df.columns:
        st.warning("⚠️ No 'Source' column found in the data.")
        return
    
    # Analyze each location column
    for loc_col in location_columns:
        st.markdown(f"**Analysis for {loc_col}:**")
        
        # Look for rural indicators
        rural_mask = df[loc_col].astype(str).str.contains('rural|village|town', case=False, na=False)
        rural_leads = df[rural_mask]
        
        if len(rural_leads) > 0:
            st.success(f"✅ Found {len(rural_leads)} rural leads in {loc_col}")
            
            # Rural leads by source
            rural_by_source = rural_leads['Source'].value_counts()
            st.dataframe(rural_by_source.to_frame('Rural Leads Count'), use_container_width=True)
            
            # Show sample data
            st.markdown("**Sample Rural Leads:**")
            display_cols = ['Source', loc_col]
            if 'Stage' in df.columns:
                display_cols.append('Stage')
            st.dataframe(rural_leads[display_cols].head(10), use_container_width=True)
        else:
            st.info(f"No obvious rural indicators found in {loc_col}")
            
            # Show unique values to help user understand the data
            unique_vals = df[loc_col].value_counts().head(10)
            st.markdown(f"**Top values in {loc_col}:**")
            st.dataframe(unique_vals.to_frame('Count'), use_container_width=True)

def analyze_source_performance(df, user_prompt):
    """Analyze source performance metrics"""
    st.markdown("### 📈 **Source Performance Analysis**")
    
    if 'Source' not in df.columns:
        st.warning("⚠️ No 'Source' column found in the data.")
        return
    
    # Basic source analysis
    source_counts = df['Source'].value_counts()
    st.markdown("**Lead Count by Source:**")
    st.dataframe(source_counts.to_frame('Total Leads'), use_container_width=True)
    
    # If Stage column exists, analyze conversion
    if 'Stage' in df.columns:
        st.markdown("**Conversion Analysis by Source:**")
        
        # Create cross-tabulation
        source_stage = pd.crosstab(df['Source'], df['Stage'], margins=True)
        st.dataframe(source_stage, use_container_width=True)
        
        # Calculate conversion rates if Sales Closure exists
        if 'Sales Closure' in df['Stage'].values:
            conversion_rates = []
            for source in df['Source'].unique():
                source_data = df[df['Source'] == source]
                total = len(source_data)
                closures = len(source_data[source_data['Stage'] == 'Sales Closure'])
                rate = (closures / total * 100) if total > 0 else 0
                conversion_rates.append({'Source': source, 'Total Leads': total, 'Closures': closures, 'Conversion Rate %': round(rate, 2)})
            
            conv_df = pd.DataFrame(conversion_rates).sort_values('Conversion Rate %', ascending=False)
            st.markdown("**Conversion Rates by Source:**")
            st.dataframe(conv_df, use_container_width=True)
    
    # Behavioral analysis if available
    behavioral_cols = ['TotalTimeSpent', 'ClickCount', 'PageDepth']
    available_behavioral = [col for col in behavioral_cols if col in df.columns]
    
    if available_behavioral:
        st.markdown("**Behavioral Metrics by Source:**")
        behavioral_by_source = df.groupby('Source')[available_behavioral].mean().round(2)
        st.dataframe(behavioral_by_source, use_container_width=True)

def analyze_stage_by_source(df, user_prompt):
    """Analyze stage distribution by source"""
    st.markdown("### 🎯 **Stage Distribution by Source**")
    
    if 'Source' not in df.columns or 'Stage' not in df.columns:
        st.warning("⚠️ Missing 'Source' or 'Stage' columns in the data.")
        return
    
    # Create detailed cross-tabulation
    stage_source = pd.crosstab(df['Source'], df['Stage'], margins=True)
    st.dataframe(stage_source, use_container_width=True)
    
    # Percentage breakdown
    stage_source_pct = pd.crosstab(df['Source'], df['Stage'], normalize='index') * 100
    stage_source_pct = stage_source_pct.round(1)
    
    st.markdown("**Percentage Distribution:**")
    st.dataframe(stage_source_pct, use_container_width=True)
    
    # Create visualization
    try:
        fig, ax = plt.subplots(figsize=(12, 8))
        stage_source_pct.plot(kind='bar', ax=ax, stacked=True)
        ax.set_title('Stage Distribution by Source (%)')
        ax.set_xlabel('Source')
        ax.set_ylabel('Percentage')
        plt.xticks(rotation=45)
        plt.legend(title='Stage', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.info("Could not create visualization")

# ---------------------------
# ✅ MAIN ROUTING FUNCTIONS
# ---------------------------
def run_llm_query_on_df(df, user_prompt):
    """Main function that routes to appropriate analysis"""
    perform_actual_data_analysis(df, user_prompt)

# ---------------------------
# ✅ SUPER INTELLIGENT MAIN HANDLER
# ---------------------------
def intelligent_query_classifier(user_prompt):
    """Classify user intent using LLM"""
    try:
        classification_prompt = f"""Classify this user query into one of these categories:

CATEGORIES:
1. VISUALIZATION - User wants charts, graphs, plots, visual analysis
2. DATA_FILTERING - User wants filtered data, specific records, CSV export
3. ANALYSIS - User wants insights, explanations, trends, comparisons, business intelligence
4. CALCULATION - User wants specific calculations, metrics, KPIs

USER QUERY: "{user_prompt}"

Respond with ONLY the category name (VISUALIZATION, DATA_FILTERING, ANALYSIS, or CALCULATION):"""

        response = ollama.chat(
            model="llama3",
            messages=[
                {"role": "system", "content": "You are a query classifier. Respond with only the category name."},
                {"role": "user", "content": classification_prompt}
            ]
        )
        
        return response["message"]["content"].strip().upper()
    except:
        # Fallback to keyword-based classification
        lower = user_prompt.lower()
        if any(k in lower for k in ["chart", "plot", "graph", "vs", "distribution", "histogram", "scatter", "bar"]):
            return "VISUALIZATION"
        elif any(k in lower for k in ["csv", "export", "download", "filter", "leads with", "show me", "get me"]):
            return "DATA_FILTERING"
        else:
            return "ANALYSIS"

def handle_llm_request(df, user_prompt):
    """Super intelligent handler that can process any user query"""
    
    # Show loading message
    with st.spinner("🧠 Analyzing your request..."):
        
        # Classify the query
        query_type = intelligent_query_classifier(user_prompt)
        
        # Handle based on classification
        if query_type == "VISUALIZATION":
            st.info("📊 Generating visualization...")
            generate_chart_from_prompt(df, user_prompt)
            
        elif query_type == "DATA_FILTERING":
            st.info("🔍 Filtering data...")
            run_llm_query_and_return_csv(df, user_prompt)
            
        elif query_type in ["ANALYSIS", "CALCULATION"]:
            st.info("🧠 Performing comprehensive analysis...")
            run_llm_query_on_df(df, user_prompt)
            
        else:
            # Default to analysis for unknown queries
            st.info("🧠 Analyzing your query...")
            run_llm_query_on_df(df, user_prompt)

# ---------------------------
# ✅ ADVANCED CALCULATION ENGINE
# ---------------------------
def perform_advanced_calculations(df, user_prompt):
    """Perform complex calculations and return results"""
    try:
        data_analysis = analyze_data_comprehensively(df)
        
        system_message = f"""You are a business intelligence expert. Perform calculations based on the user's request using the provided data.

DATA ANALYSIS:
{json.dumps(data_analysis, indent=2, default=str)}

INSTRUCTIONS:
1. Generate Python code to perform the requested calculations
2. Use pandas operations on the DataFrame called 'df'
3. Store results in variables and create a summary
4. Show your calculations step by step
5. Provide business context for the results

USER REQUEST: {user_prompt}

Generate Python code to perform the calculations:"""

        response = ollama.chat(
            model="llama3",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt}
            ]
        )

        code = response["message"]["content"].strip()
        
        # Clean and execute code
        if "```" in code:
            parts = code.split("```")
            for part in parts:
                if "df[" in part or "=" in part:
                    code = part.strip()
                    break
        
        local_vars = {"df": df, "pd": pd, "np": np}
        exec(code, {}, local_vars)
        
        # Display results
        st.success("🧮 **Calculation Results:**")
        for key, value in local_vars.items():
            if key not in ['df', 'pd', 'np'] and not key.startswith('_'):
                st.write(f"**{key}:** {value}")
                
    except Exception as e:
        st.error("❌ Could not perform the requested calculations.")

# ---------------------------
# ✅ CONTEXT-AWARE CONVERSATION SYSTEM
# ---------------------------
def initialize_conversation_context():
    """Initialize conversation context in session state"""
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'data_context' not in st.session_state:
        st.session_state.data_context = {}

def add_to_conversation_history(query, response_type, key_insights=None):
    """Add query and response to conversation history"""
    st.session_state.conversation_history.append({
        'timestamp': datetime.now(),
        'query': query,
        'response_type': response_type,
        'key_insights': key_insights
    })
    
    # Keep only last 10 conversations
    if len(st.session_state.conversation_history) > 10:
        st.session_state.conversation_history = st.session_state.conversation_history[-10:]

def get_conversation_context():
    """Get relevant conversation context for better responses"""
    if not st.session_state.conversation_history:
        return ""
    
    recent_queries = st.session_state.conversation_history[-3:]  # Last 3 queries
    context = "PREVIOUS CONVERSATION CONTEXT:\n"
    for i, conv in enumerate(recent_queries, 1):
        context += f"{i}. User asked: '{conv['query']}' (Type: {conv['response_type']})\n"
    
    return context

# ---------------------------
# ✅ ENHANCED MAIN HANDLER WITH CONTEXT
# ---------------------------
def handle_intelligent_request(df, user_prompt):
    """Enhanced handler with conversation context and memory"""
    
    # Initialize conversation context
    initialize_conversation_context()
    
    # Get conversation context
    context = get_conversation_context()
    
    # Show loading message
    with st.spinner("🧠 Processing your intelligent request..."):
        
        # Classify the query with context
        query_type = intelligent_query_classifier(user_prompt)
        
        # Add context to the prompt if available
        enhanced_prompt = user_prompt
        if context:
            enhanced_prompt = f"{context}\nCURRENT QUERY: {user_prompt}"
        
        # Handle based on classification
        if query_type == "VISUALIZATION":
            st.info("📊 Creating intelligent visualization...")
            generate_chart_from_prompt(df, enhanced_prompt)
            add_to_conversation_history(user_prompt, "VISUALIZATION")
            
        elif query_type == "DATA_FILTERING":
            st.info("🔍 Intelligently filtering data...")
            run_llm_query_and_return_csv(df, enhanced_prompt)
            add_to_conversation_history(user_prompt, "DATA_FILTERING")
            
        elif query_type in ["ANALYSIS", "CALCULATION"]:
            st.info("🧠 Performing comprehensive intelligent analysis...")
            run_llm_query_on_df(df, enhanced_prompt)
            add_to_conversation_history(user_prompt, "ANALYSIS")
            
        else:
            st.info("🧠 Processing with full intelligence...")
            run_llm_query_on_df(df, enhanced_prompt)
            add_to_conversation_history(user_prompt, "GENERAL")

# ---------------------------
# ✅ Helper function to show available columns
# ---------------------------
def show_available_columns(df):
    st.markdown("### 📋 Available Columns in Your Data:")
    col_info = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        sample_vals = df[col].dropna().unique()[:3].tolist() if df[col].dtype == 'object' else [df[col].min(), df[col].max()]
        col_info.append({"Column": col, "Type": dtype, "Sample Values": str(sample_vals)})
    
    col_df = pd.DataFrame(col_info)
    st.dataframe(col_df, use_container_width=True)

# ---------------------------
# ✅ Optional Dropdown UI for Suggestions (call this from app.py)
# ---------------------------
def prompt_suggestion_ui():
    st.markdown("💡 **Need inspiration? Try one of these intelligent prompts:**")
    
    # Categorize prompts
    categories = {
        "🧠 Business Intelligence": SUGGESTED_PROMPTS[12:],
        "📊 Data Analysis": SUGGESTED_PROMPTS[:4],
        "📈 Visualizations": SUGGESTED_PROMPTS[5:9],
        "🔍 Data Filtering": SUGGESTED_PROMPTS[9:12]
    }
    
    selected_category = st.selectbox("Choose Category:", list(categories.keys()))
    selected_prompt = st.selectbox("Select Prompt:", categories[selected_category])
    
    if st.button("🚀 Run Intelligent Analysis"):
        handle_llm_request(st.session_state.merged_df, selected_prompt)
