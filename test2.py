import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm.openai import OpenAI
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set up OpenAI API Key
os.environ["OPENAI_API_KEY"] = ''
llm = OpenAI(api_token=os.getenv("OPENAI_API_KEY"))

# Streamlit UI Enhancements
st.set_page_config(page_title="AI-Powered Data Analysis", layout="wide")
st.title("📊 AI-Powered Data Analysis Tool")
st.sidebar.header("Upload Your Dataset")

# Upload file
uploaded_file = st.sidebar.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        # Load dataset
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        # Convert to SmartDataframe
        sdf = SmartDataframe(df, config={"llm": llm})

        # Generate AI Insights
        st.subheader("📈 AI-Generated Insights")
        ai_summary = sdf.chat("Summarize the dataset in a few sentences")
        st.write(ai_summary)

        # Summary Statistics
        st.subheader("📊 Summary Statistics")
        st.write(df.describe())

        # Missing Values
        st.subheader("❗ Missing Values")
        st.write(df.isnull().sum())

        # Data Cleaning Options
        st.sidebar.subheader("Data Preprocessing")
        if st.sidebar.button("Drop Missing Values"):
            df.dropna(inplace=True)
            st.sidebar.success("Missing values dropped!")

        # Natural Language Query
        st.subheader("💡 Ask a Question About Your Data")
        query = st.text_input("Enter your question:")
        if query:
            try:
                response = sdf.chat(query)
                st.write("### AI Response:")
                st.write(response)
            except Exception as e:
                st.error(f"Error: {e}")

        # Visualization Options
        st.subheader("📊 Generate Visualizations")
        chart_type = st.selectbox("Select Chart Type",
                                  ["Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart", "Histogram", "Box Plot",
                                   "Heatmap"])
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
                elif chart_type == "Pie Chart":
                    plt.pie(df[y_col].value_counts(), labels=df[y_col].unique(), autopct="%1.1f%%")
                elif chart_type == "Histogram":
                    sns.histplot(df[y_col], kde=True, bins=30)
                elif chart_type == "Box Plot":
                    sns.boxplot(data=df, x=x_col, y=y_col)
                elif chart_type == "Heatmap":
                    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')

                plt.title(f"{chart_type} of {y_col} vs {x_col}")
                plt.xticks(rotation=45)
                st.pyplot(plt)
            except Exception as e:
                st.error(f"Error: {e}")

        # Download Processed Data
        st.sidebar.subheader("Download Processed Data")
        csv = df.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button("Download CSV", csv, "processed_data.csv", "text/csv")
    except Exception as e:
        st.error(f"Error reading file: {e}")
else:
    st.info("Upload a dataset to start.")
