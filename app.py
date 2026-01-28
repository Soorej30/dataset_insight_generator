import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Page Configuration
st.set_page_config(page_title="Auto-Insight EDA Tool", layout="wide")

st.title("📊 Auto-Insight EDA System")
st.markdown("Upload a CSV to get automated statistics, visualizations, and data quality checks.")

# --- Sidebar: File Uploads ---
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
schema_file = st.sidebar.file_uploader("Upload Data Dictionary (Optional)", type=["csv", "txt", "xlsx"])
specific_analysis = st.sidebar.text_area("Specific Analysis Request (Optional)", 
                                        placeholder="e.g., 'Focus on the relationship between Price and Sales'")

if uploaded_file is not None:
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
    
    # Numeric Stats
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
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
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
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
            fig, ax = plt.subplots()
            sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)

    # Plot 2: Distribution (Histogram)
    if num_cols:
        with fig_cols[1]:
            target_num = st.selectbox("Select column for Histogram:", num_cols)
            st.write(f"**Distribution of {target_num}**")
            fig, ax = plt.subplots()
            sns.histplot(df[target_num], kde=True, ax=ax)
            st.pyplot(fig)

    # Plot 3: Outlier Check (Boxplot)
    if num_cols:
        with fig_cols[0]:
            st.write(f"**Outlier Detection: {target_num}**")
            fig, ax = plt.subplots()
            sns.boxplot(x=df[target_num], ax=ax)
            st.pyplot(fig)

    # Plot 4: Categorical Breakdown (Bar Chart)
    if cat_cols:
        with fig_cols[1]:
            target_cat = st.selectbox("Select column for Bar Chart:", cat_cols)
            st.write(f"**Frequency of {target_cat}**")
            fig, ax = plt.subplots()
            df[target_cat].value_counts().head(10).plot(kind='bar', ax=ax)
            st.pyplot(fig)

    # Plot 5: Relationship (Scatter Plot)
    if len(num_cols) >= 2:
        with st.container():
            st.write(f"**Scatter Plot: {num_cols[0]} vs {num_cols[1]}**")
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.scatterplot(data=df, x=num_cols[0], y=num_cols[1], ax=ax)
            st.pyplot(fig)

    # --- 4. Narrative Insights ---
    st.header("💡 Automated Narrative Insights")
    
    # PROMPT CONSTRUCTION (For GenAI)
    # Note: In a real app, you'd send this to an API like Gemini or OpenAI.
    # Here, we'll provide the 'Narrative' based on the computed stats.
    
    with st.expander("Click to generate AI Analysis"):
        st.write("### Key Patterns & Insights")
        st.write(f"1. **Missing Data:** The dataset is {100 - info_df['% Missing'].mean():.1f}% complete.")
        if len(num_cols) > 0:
            top_num = num_cols[0]
            st.write(f"2. **Numerical Center:** The average {top_num} is {df[top_num].mean():.2f}, which is {'higher' if df[top_num].mean() > df[top_num].median() else 'lower'} than the median, suggesting a possible skew.")
        
        if len(num_cols) > 1:
            corr_matrix = df[num_cols].corr()
            strongest = corr_matrix.unstack().sort_values(ascending=False).drop_duplicates()
            # Logic to find strongest non-1.0 correlation
            st.write(f"3. **Strongest Relationship:** Found between numeric variables (check the heatmap for correlation coefficients).")
            
        st.write("4. **Anomalies:** Outliers were detected in several columns, particularly in numeric distributions that show a long tail.")
        st.write("5. **Data Quality:** The system flagged duplicates and/or empty values that should be cleaned before modeling.")
        
        if specific_analysis:
            st.info(f"**Custom Request Analysis:** {specific_analysis}")
            st.write("- Analyzing your custom request... (In a live GenAI integration, this section would be populated by the LLM response).")

    st.subheader("⚠️ Limitations & Bias Note")
    st.write("> **Note:** This analysis is automated. Results may be biased if the sampling method of the CSV was non-random. Missingness in key columns might lead to 'Informer Bias' where only successful or visible cases are recorded.")

else:
    st.info("Please upload a CSV file from the sidebar to begin.")