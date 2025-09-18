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
        # Quick fallback for common chart requests
        if "stage" in prompt.lower() and "Stage" in df.columns:
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
        
        # Get only essential column info for speed
        columns_list = list(df.columns)
        
        # Simplified, faster prompt
        system_message = f"""Generate Python code for a chart. DataFrame is 'df'.
Available columns: {columns_list[:15]}  

Rules:
- Use matplotlib/seaborn only
- Include: import matplotlib.pyplot as plt, import seaborn as sns
- End with: plt.tight_layout()
- NO plt.show()
- NO explanations

Request: {prompt}"""

        response = ollama.chat(
            model="llama3",
            messages=[{"role": "user", "content": system_message}],
            options={"temperature": 0.1}  # Lower temperature for more consistent code
        )

        code = response["message"]["content"].strip()

        # Aggressive code cleaning
        if "```" in code:
            code_blocks = code.split("```")
            for block in code_blocks:
                if any(keyword in block for keyword in ["plt.", "sns.", "df["]):
                    code = block.strip()
                    break
        
        # Remove markdown artifacts
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
        if fig.get_axes():  # Check if chart was actually created
            st.pyplot(fig)
        else:
            raise Exception("No chart was generated")
            
    except Exception as e:
        # Quick fallback chart
        st.warning("⚠️ Creating a simple fallback chart...")
        try:
            fig, ax = plt.subplots(figsize=(8, 5))
            if 'Stage' in df.columns:
                df['Stage'].value_counts().head(10).plot(kind='bar', ax=ax)
                ax.set_title('Top 10 Lead Stages')
            elif len(df.select_dtypes(include=['object']).columns) > 0:
                first_cat_col = df.select_dtypes(include=['object']).columns[0]
                df[first_cat_col].value_counts().head(10).plot(kind='bar', ax=ax)
                ax.set_title(f'Top 10 {first_cat_col} Values')
            else:
                ax.text(0.5, 0.5, 'No suitable data for visualization', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Data Overview')
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            
        except:
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
        # Get comprehensive data analysis
        data_analysis = analyze_data_comprehensively(df)
        
        # Get sample data for context
        sample_data = df.head(10).to_dict('records')
        
        system_message = f"""You are an expert data analyst with deep knowledge of business intelligence, customer analytics, and lead management systems. 

COMPREHENSIVE DATA ANALYSIS:
{json.dumps(data_analysis, indent=2, default=str)}

SAMPLE DATA:
{json.dumps(sample_data, indent=2, default=str)}

INSTRUCTIONS:
1. Provide detailed, accurate analysis based ONLY on the actual data provided
2. Include specific numbers, percentages, and insights
3. Explain the business logic and reasoning behind your analysis
4. If the user asks about trends, patterns, or comparisons, calculate them from the data
5. If you need to make calculations, show your working
6. Never hallucinate or make up data that doesn't exist
7. If you cannot answer something definitively from the data, clearly state that
8. Provide actionable insights and recommendations when relevant
9. Use business terminology appropriate for sales/marketing teams
10. Structure your response with clear headings and bullet points for readability

USER QUESTION: {user_prompt}

Provide a comprehensive, data-driven analysis:"""

        response = ollama.chat(
            model="llama3",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt}
            ]
        )

        answer = response["message"]["content"]
        st.success("🧠 **Comprehensive Data Analysis:**")
        st.markdown(answer)
        
        # Add data summary if relevant
        if any(word in user_prompt.lower() for word in ['summary', 'overview', 'total', 'count']):
            st.markdown("---")
            st.markdown("### 📊 **Quick Data Summary:**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Records", f"{len(df):,}")
            with col2:
                st.metric("Total Columns", len(df.columns))
            with col3:
                if 'Stage' in df.columns:
                    st.metric("Unique Stages", df['Stage'].nunique())
            with col4:
                if 'Source' in df.columns:
                    st.metric("Unique Sources", df['Source'].nunique())

    except Exception as e:
        st.error(f"❌ Could not analyze the data. Please try rephrasing your question.")

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