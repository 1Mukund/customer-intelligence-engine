
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ollama
import re

def extract_code_block(text):
    # Extracts Python code between triple backticks or cleans up plain responses
    code_block = re.findall(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if code_block:
        return code_block[0].strip()
    return text.strip()

def generate_kpi_charts(df, user_instruction):
    try:
        # Smart fallback for specific requests
        if "source" in user_instruction.lower() and "Source" in df.columns:
            st.info("📊 Creating source analysis...")
            fig, ax = plt.subplots(figsize=(12, 6))
            
            if "conversion" in user_instruction.lower() and "Stage" in df.columns:
                # Create conversion analysis by source
                source_stage_counts = df.groupby(['Source', 'Stage']).size().unstack(fill_value=0)
                if 'Sales Closure' in source_stage_counts.columns:
                    conversion_rates = (source_stage_counts['Sales Closure'] / source_stage_counts.sum(axis=1) * 100).sort_values(ascending=False)
                    conversion_rates.head(10).plot(kind='bar', ax=ax, color='green')
                    ax.set_title('Conversion Rates by Source (% Sales Closure)')
                    ax.set_ylabel('Conversion Rate (%)')
                else:
                    source_counts = df['Source'].value_counts().head(10)
                    source_counts.plot(kind='bar', ax=ax, color='lightblue')
                    ax.set_title('Lead Distribution by Source')
                    ax.set_ylabel('Count')
            else:
                source_counts = df['Source'].value_counts().head(10)
                source_counts.plot(kind='bar', ax=ax, color='lightblue')
                ax.set_title('Lead Distribution by Source')
                ax.set_ylabel('Count')
            
            ax.set_xlabel('Source')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            return
            
        elif "stage" in user_instruction.lower() and "Stage" in df.columns:
            st.info("📊 Creating stage analysis...")
            fig, ax = plt.subplots(figsize=(10, 6))
            stage_counts = df['Stage'].value_counts()
            stage_counts.plot(kind='bar', ax=ax, color='lightcoral')
            ax.set_title('Lead Distribution by Stage')
            ax.set_xlabel('Stage')
            ax.set_ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            return

        # Get available columns for context
        columns_list = list(df.columns)
        
        # Enhanced prompt with better context
        prompt = f"""Generate Python code for KPI analysis. DataFrame is 'df'.

Available columns: {columns_list}

IMPORTANT: Pay attention to the specific column mentioned in the user request.
- If user asks about "source", use the 'Source' column
- If user asks about "stage", use the 'Stage' column  
- If user asks about conversion rates, calculate percentages properly

Rules:
- Use matplotlib/seaborn only
- Include: import matplotlib.pyplot as plt, import seaborn as sns
- End with: plt.tight_layout()
- NO plt.show()
- NO explanations
- Create meaningful KPI visualizations
- Use the EXACT column names from the available columns list

User request: {user_instruction}

Generate Python code:"""

        response = ollama.chat(
            model="llama3",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1}
        )
        
        llm_output = response["message"]["content"].strip()
        generated_code = extract_code_block(llm_output)
        
        # Clean the code
        if "```" in generated_code:
            code_blocks = generated_code.split("```")
            for block in code_blocks:
                if any(keyword in block for keyword in ["plt.", "sns.", "df["]):
                    generated_code = block.strip()
                    break
        
        # Remove markdown artifacts
        lines = [line.strip() for line in generated_code.split('\n') 
                if line.strip() and not line.strip().startswith('#') 
                and not line.strip().startswith('```')
                and not line.strip().startswith('python')]
        
        generated_code = '\n'.join(lines)
        
        # Add required imports if missing
        if 'import matplotlib.pyplot as plt' not in generated_code:
            generated_code = 'import matplotlib.pyplot as plt\nimport seaborn as sns\n' + generated_code

        # Show the generated code for transparency
        with st.expander("📝 Generated Code"):
            st.code(generated_code, language='python')

        # Validate string quotes
        if ('"' in generated_code and generated_code.count('"') % 2 != 0) or \
           ("'" in generated_code and generated_code.count("'") % 2 != 0):
            st.error("🛑 Detected unclosed string in code. Please rephrase your instruction.")
            return

        # Execute the code
        local_vars = {'df': df, 'plt': plt, 'sns': sns, 'st': st, 'pd': pd, 'np': np}
        exec(generated_code, {}, local_vars)
        
        # Display the chart
        fig = plt.gcf()
        if fig.get_axes():
            st.pyplot(fig)
        else:
            raise Exception("No chart was generated")

    except Exception as e:
        st.error(f"❌ Could not generate KPI chart: {e}")
        
        # Fallback KPI chart
        st.warning("⚠️ Creating a fallback KPI chart...")
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if 'Stage' in df.columns:
                stage_counts = df['Stage'].value_counts().head(10)
                stage_counts.plot(kind='bar', ax=ax, color='steelblue')
                ax.set_title('KPI: Lead Distribution by Stage')
                ax.set_xlabel('Stage')
                ax.set_ylabel('Count')
                plt.xticks(rotation=45)
            elif len(df.select_dtypes(include=['object']).columns) > 0:
                first_cat_col = df.select_dtypes(include=['object']).columns[0]
                df[first_cat_col].value_counts().head(10).plot(kind='bar', ax=ax, color='orange')
                ax.set_title(f'KPI: Top 10 {first_cat_col} Values')
                ax.set_xlabel(first_cat_col)
                ax.set_ylabel('Count')
                plt.xticks(rotation=45)
            else:
                ax.text(0.5, 0.5, 'No suitable data for KPI visualization', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('KPI Dashboard - No Data Available')
            
            plt.tight_layout()
            st.pyplot(fig)
            
        except Exception as fallback_error:
            st.error(f"❌ Could not create fallback chart: {fallback_error}")
