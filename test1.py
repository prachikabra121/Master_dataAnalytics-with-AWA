import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm.openai import OpenAI
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set up OpenAI API Key
os.environ["OPENAI_API_KEY"] = ''

# Initialize LLMstr
llm = OpenAI(api_token=os.getenv("OPENAI_API_KEY"))

# Streamlit App
st.title("AI-Powered Data Analysis Tool")
st.sidebar.header("Upload Your Dataset")

# Upload file
uploaded_file = st.sidebar.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Load dataset
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)

        st.write("### Dataset Preview")
        st.dataframe(df.head())

        # Convert to SmartDataframe
        sdf = SmartDataframe(df, config={"llm": llm})

        # Generate Insights
        st.write("### Summary Statistics")
        st.write(df.describe())

        # Check for missing values
        st.write("### Missing Values")
        st.write(df.isnull().sum())

        # Natural Language Query
        st.write("### Ask a Question")
        query = st.text_input("Enter your question about the data:")

        if query:
            try:
                response = sdf.chat(query)
                st.write("#### AI Response:")
                st.write(response)
            except Exception as e:
                st.error(f"Error: {e}")

        # Visualization
        st.write("### Generate Visualization")
        chart_type = st.selectbox("Select Chart Type", ["Bar Chart", "Line Chart", "Scatter Plot"])
        x_col = st.selectbox("Select X-axis Column", df.columns)
        y_col = st.selectbox("Select Y-axis Column", df.columns)

        if st.button("Generate Chart"):
            try:
                plt.figure(figsize=(10, 6))
                if chart_type == "Bar Chart":
                    sns.barplot(data=df, x=x_col, y=y_col)
                elif chart_type == "Line Chart":
                    sns.lineplot(data=df, x=x_col, y=y_col)
                elif chart_type == "Scatter Plot":
                    sns.scatterplot(data=df, x=x_col, y=y_col)
                plt.title(f"{chart_type} of {y_col} vs {x_col}")
                plt.xticks(rotation=45)
                st.pyplot(plt)
            except Exception as e:
                st.error(f"Error: {e}")

    except Exception as e:
        st.error(f"Error reading file: {e}")
else:
    st.info("Upload a dataset to start.")