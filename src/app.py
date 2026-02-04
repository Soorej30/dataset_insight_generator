import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import requests
import json

def ollama_chat(prompt, schema):
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.1:8b",
            "prompt": prompt,
            "stream": False,
            "format": schema
        }
    )
    return r.json()["response"]

def get_main_columns(cols, schema_file):
    if schema_file is not None:
        try:
            schema_file.seek(0)
        except Exception:
            pass

        name = getattr(schema_file, "name", "").lower()
        if name.endswith(('.xls', '.xlsx')):
            data_dict = pd.read_excel(schema_file)
        else:
            data_dict = pd.read_csv(schema_file)

        data_dict = data_dict[data_dict['Column_name'].isin(cols)]
        # print(data_dict.head())
        prompt = f"""
                You are a data analyst analysing a dataset. You should try and understand the data from the given instructions.
                Consider this list of columns in the data file I have, and consider this data dictionary explaining the columns in the data
                columns: {cols}
                data dictionary: {data_dict}

                Considering this data description, give me the list of the main 5 columns for which creating a categorical visualization analysis will help me understand the data.
                Also tell me why you chose these. 
                DOUBLE CHECK THAT YOU SELECT COLUMNS ONLY FROM THE columns LIST
            """
        # print(prompt)
        schema = {
            "type": "object",
            "properties": {
                "columns": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "description": "Unique column name"
                    },
                },
                "reason":{
                    "type": "string",
                    "description": "Why these columns where chosen over the others."
                }
            },
            "required": ["columns", "reason"]
        }

        optimized_resume_text = ollama_chat(prompt, schema)

        return json.loads(optimized_resume_text)['columns']
    else:
        prompt = f"""
                You are a data analyst analysing a dataset. You should try and understand the data from the given instructions.
                Consider this list of columns in the data file I have
                columns: {cols}

                Considering this column names and try to understand what these columns will mean.
                Then give me the list of 5 categorical columns which will give me the most information about the data.
                DO NOT CHOOSE FIELDS THAT COULD HAVE MORE THEN 30 CATEGORIES.
                Also tell me why you chose these.

                DOUBLE CHECK THAT YOU SELECT COLUMNS ONLY FROM THE columns LIST
                RETURN ONLY THE 5 COLUMN NAMES AND THE REASON
            """
        # print(prompt)
        schema = {
            "type": "object",
            "properties": {
                "columns": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "description": "Unique column name"
                    },
                    "description": "The list of 5 columns that provides most information about the data.",
                    "max_length": "Maximum length of 5 elements"
                },
                "reason":{
                    "type": "string",
                    "description": "Why these columns where chosen over the others."
                }
            },
            "required": ["columns", "reason"]
        }

        optimized_resume_text = ollama_chat(prompt, schema)

        return json.loads(optimized_resume_text)['columns']

def generate_ai_insights(df, schema_file=None, max_rows=5):
    """
    Build a compact dataset summary and ask Ollama to produce structured insights.
    Returns parsed JSON (dict) with keys per the schema below.
    """
    # small samples and summaries
    sample_csv = df.head(max_rows).to_csv(index=False)
    info_df = pd.DataFrame({
        "Type": df.dtypes.astype(str),
        "Missing Values": df.isnull().sum(),
        "% Missing": (df.isnull().sum() / len(df) * 100).round(2)
    }).reset_index().rename(columns={'index': 'column'})
    info_snippet = info_df.head(10).to_dict(orient='records')

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # compute top correlations (absolute), limit to top 3 non-self pairs
    corr_pairs = []
    if len(num_cols) > 1:
        corr = df[num_cols].corr().abs().where(~np.eye(len(num_cols), dtype=bool))
        corr_unstack = corr.unstack().dropna().sort_values(ascending=False)
        seen = set()
        for (a, b), val in corr_unstack.items():
            key = tuple(sorted((a, b)))
            if key in seen:
                continue
            seen.add(key)
            corr_pairs.append({"pair": f"{key[0]} & {key[1]}", "corr": float(val)})
            if len(corr_pairs) >= 3:
                break

    # include basic stats for numeric columns (small)
    numeric_stats = df[num_cols].describe().T[['mean', '50%', 'std']].round(3).fillna("").to_dict(orient='index')

    # Construct prompt (concise)
    prompt = f"""
    You are a data analyst. Given the short dataset summary below, provide:
    1) A JSON object with keys: 'insights' (array of short insight strings, max 6),
    'top_columns' (array of up to 5 column names to focus categorical exploration on),
    and 'recommendations' (array of actionable steps).
    2) If specific_analysis is provided, include a focused insight/recommendation about it.

    Dataset summary:
    - rows: {len(df)}, columns: {len(df.columns)}
    - numeric columns: {num_cols}
    - categorical columns: {cat_cols}
    - sample rows (csv, up to {max_rows}): 
    {sample_csv}

    - column dtypes / missing (up to 10):
    {json.dumps(info_snippet, ensure_ascii=False)}

    - top numeric correlations (abs, top 3):
    {json.dumps(corr_pairs, ensure_ascii=False)}

    - small numeric stats:
    {json.dumps(numeric_stats, ensure_ascii=False)}

    Return only valid JSON matching the schema.
    """

    schema = {
        "type": "object",
        "properties": {
            "insights": {
                "type": "array",
                "items": {"type": "string"}
            },
            "top_columns": {
                "type": "array",
                "items": {"type": "string"}
            },
            "recommendations": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["insights", "top_columns", "recommendations"]
    }

    try:
        raw = ollama_chat(prompt, schema)
        parsed = json.loads(raw)
    except Exception as e:
        # fallback: return a minimal structure on failure
        parsed = {
            "insights": [f"AI call failed: {str(e)}"],
            "top_columns": (cat_cols[:5] if cat_cols else num_cols[:5]),
            "recommendations": ["Retry AI call or inspect logs."]
        }
    return parsed

# Page Configuration
st.set_page_config(page_title="Auto-Insight EDA Tool", layout="wide")

st.title("📊 Auto-Insight EDA System")
st.markdown("Upload a CSV to get automated statistics, visualizations, and data quality checks.")

# --- Sidebar: File Uploads ---
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
schema_file = st.sidebar.file_uploader("Upload Data Dictionary (Optional)", type=["csv", "txt", "xlsx"])
# specific_analysis = st.sidebar.text_area("Specific Analysis Request (Optional)", 
#                                         placeholder="e.g., 'Focus on the relationship between Price and Sales'")

# Add run / reset controls using session_state
if "run_analysis" not in st.session_state:
    st.session_state["run_analysis"] = False

if st.sidebar.button("Run Analysis"):
    st.session_state["run_analysis"] = True

if st.sidebar.button("Reset"):
    st.session_state["run_analysis"] = False

# Only execute the heavy analysis when a file is uploaded AND the Run button was pressed
if uploaded_file is not None and st.session_state["run_analysis"]:
    df = pd.read_csv(uploaded_file)
    
    # --- 1. Dataset Overview ---
    st.header("📋 Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Duplicates", df.duplicated().sum())


    tabs = st.tabs(["Data Preview", "Data Types & Missing", "Quality Checks"])
    
    with tabs[0]:
        st.dataframe(df.head(10))
    
    with tabs[1]:
        info_df = pd.DataFrame({
            "Type": df.dtypes.astype(str),
            "Missing Values": df.isnull().sum(),
            "% Missing": (df.isnull().sum() / len(df) * 100).round(2)
        })
        st.table(info_df)

    with tabs[2]:
        st.subheader("Quality Red Flags")
        # Check for columns with only one value
        single_val_cols = [col for col in df.columns if df[col].nunique() <= 1]
        high_missing_cols = [col for col in df.columns if (df[col].isnull().sum() / len(df)) > 0.5]
        
        if single_val_cols: st.warning(f"Columns with only one value: {single_val_cols}")
        if high_missing_cols: st.error(f"Columns with >50% missing data: {high_missing_cols}")
        if not single_val_cols and not high_missing_cols: st.success("No major quality red flags detected!")

    # --- 2. Descriptive Statistics ---
    st.header("📈 Descriptive Statistics")
    
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if num_cols:
        num_main_5_cols = get_main_columns(num_cols, schema_file)
    if cat_cols:
        cat_main_5_cols = get_main_columns(cat_cols, schema_file)
    # print("**************************************************************")
    # print(num_cols)
    # print(num_main_5_cols)
    # print("**************************************************************")
    # print(cat_cols)
    # print(cat_main_5_cols)
    # print("**************************************************************")

    # Numeric Stats
    if len(num_cols) >= 2:
        st.subheader("Numeric Analysis (Top 2+ Columns)")
        stats = df[num_cols].describe().T
        # Adding IQR and Outliers
        stats['IQR'] = stats['75%'] - stats['25%']
        stats['Lower Bound'] = stats['25%'] - 1.5 * stats['IQR']
        stats['Upper Bound'] = stats['75%'] + 1.5 * stats['IQR']
        
        # Outlier flagging logic
        outlier_counts = {}
        for col in num_cols:
            outliers = df[(df[col] < stats.loc[col, 'Lower Bound']) | (df[col] > stats.loc[col, 'Upper Bound'])]
            outlier_counts[col] = len(outliers)
        stats['Outlier Count'] = pd.Series(outlier_counts)
        
        st.dataframe(stats[['mean', '50%', 'std', 'IQR', 'Outlier Count']].rename(columns={'50%': 'median'}))

    # Categorical Stats
    if cat_cols:
        st.subheader("Categorical Analysis")
        selected_cat = st.selectbox("Select a categorical column to inspect:", cat_cols)
        counts = df[selected_cat].value_counts()
        percent = df[selected_cat].value_counts(normalize=True) * 100
        cat_summary = pd.DataFrame({'Counts': counts, 'Percentage (%)': percent.round(2)})
        st.dataframe(cat_summary)

    # --- 3. Visualizations ---
    st.header("🎨 Visualizations")
    fig_cols = st.columns(2)

    # Plot 1: Correlation Heatmap
    if len(num_cols) > 1:
        with fig_cols[0]:
            st.write("**Correlation Heatmap**")
            corr = df[num_cols].corr()
            fig = px.imshow(
                corr, 
                text_auto=True, 
                aspect="auto", 
                color_continuous_scale='RdBu_r',
                labels=dict(color="Correlation")
            )
            st.plotly_chart(fig, use_container_width=True)

    # Plot 2: Distribution (Histogram)

    if num_cols:
        with fig_cols[1]:
            target_num = st.selectbox("Select column for Histogram:",num_cols, index = num_cols.index(num_main_5_cols[0]), key="hist_box")
            st.write(f"**Distribution of {target_num}**")
            fig = px.histogram(
                df, 
                x=target_num, 
                marginal="rug", # Adds a small distribution rug at the bottom
                color_discrete_sequence=['#636EFA']
            )
            st.plotly_chart(fig, use_container_width=True)

    # Plot 3: Outlier Check (Boxplot)

    if num_cols:
        st.header("📦 Distribution Analysis (Box Plot)")
        
        col_x, col_y = st.columns(2)
        
        with col_x:
            # For a proper box plot, X is usually categorical
            x_axis_box = st.selectbox("Group By (Categorical)", [cat_main_5_cols[0]], key="box_x")
        with col_y:
            # Y must be numerical
            y_axis_box = st.selectbox("Numerical Value", num_cols, index = num_cols.index(num_main_5_cols[0]), key="box_y")

        fig = px.box(
            df, 
            x=x_axis_box, 
            y=y_axis_box, 
            points="outliers", # Options: "all" (every dot), "outliers", or False
            notched=True,      # Adds a "notch" to show the confidence interval of the median
            color=x_axis_box if x_axis_box else None,
            title=f"Box Plot of {y_axis_box}" + (f" by {x_axis_box}" if x_axis_box else ""),
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)

    # Plot 4: Categorical Breakdown (Bar Chart)

    if cat_cols:
        # with fig_cols[1]:
        target_cat = st.selectbox("Select column for Bar Chart:", cat_cols, index = cat_cols.index(cat_main_5_cols[1]), key="bar_box")
        st.write(f"**Frequency of {target_cat}**")
        # Creating a frequency dataframe for Plotly
        counts = df[target_cat].value_counts().reset_index().head(10)
        counts.columns = [target_cat, 'count']
        
        fig = px.bar(
            counts, 
            x=target_cat, 
            y='count', 
            color='count',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)

    # Plot 5: Relationship (Scatter Plot)
    if len(num_cols) >= 2:
        st.write(f"**Interactive Relationship: {num_cols[0]} vs {num_cols[1]}**")
        fig = px.scatter(df, x=num_cols[0], y=num_cols[1], 
                        trendline="ols", # Adds a trend line!
                        hover_data=df.columns) # Shows all data on hover
        st.plotly_chart(fig, use_container_width=True)

    # --- 4. Narrative Insights ---
    st.header("💡 Automated Narrative Insights")
    
    # PROMPT CONSTRUCTION (For GenAI)
    # Note: In a real app, you'd send this to an API like Gemini or OpenAI.
    # Here, we'll provide the 'Narrative' based on the computed stats.
    
    prompt = """
        Consider this list of 
    """

    schema = {
        "type": "object",
        "properties": {
            "bullets": {
                "type": "array",
                "items": {
                    "type": "string",
                    "description": "A single rewritten resume bullet point"
                }
            }
        },
        "required": ["bullets"]
    }

    optimized_resume_text = ollama_chat(prompt, schema)

    new_points = json.loads(optimized_resume_text)


    # with st.expander("Click to generate AI Analysis"):
    #     st.write("### Key Patterns & Insights")
    #     st.write(f"1. **Missing Data:** The dataset is {100 - info_df['% Missing'].mean():.1f}% complete.")
    #     if len(num_cols) > 0:
    #         top_num = num_cols[0]
    #         st.write(f"2. **Numerical Center:** The average {top_num} is {df[top_num].mean():.2f}, which is {'higher' if df[top_num].mean() > df[top_num].median() else 'lower'} than the median, suggesting a possible skew.")
        
    #     if len(num_cols) > 1:
    #         corr_matrix = df[num_cols].corr()
    #         strongest = corr_matrix.unstack().sort_values(ascending=False).drop_duplicates()
    #         # Logic to find strongest non-1.0 correlation
    #         st.write(f"3. **Strongest Relationship:** Found between numeric variables (check the heatmap for correlation coefficients).")
            
    #     st.write("4. **Anomalies:** Outliers were detected in several columns, particularly in numeric distributions that show a long tail.")
    #     st.write("5. **Data Quality:** The system flagged duplicates and/or empty values that should be cleaned before modeling.")
        
    #     if specific_analysis:
    #         st.info(f"**Custom Request Analysis:** {specific_analysis}")
    #         st.write("- Analyzing your custom request... (In a live GenAI integration, this section would be populated by the LLM response).")
    
    ai_output = generate_ai_insights(df, schema_file)
    # show results
    with st.expander("Click to generate AI Analysis"):
        st.write("### Key Patterns & Insights (from Ollama)")
        for i, ins in enumerate(ai_output.get("insights", []), 1):
            st.write(f"{i}. {ins}")
        st.write("### Recommended Columns to Inspect")
        st.write(ai_output.get("top_columns", []))
        st.write("### Actionable Recommendations")
        for rec in ai_output.get("recommendations", []):
            st.write(f"- {rec}")

    st.subheader("⚠️ Limitations & Bias Note")
    st.write("> **Note:** This analysis is automated. Results may be biased if the sampling method of the CSV was non-random. Missingness in key columns might lead to 'Informer Bias' where only successful or visible cases are recorded.")

else:
    st.info("Please upload a CSV file from the sidebar to begin.")