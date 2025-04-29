import streamlit as st
import pandas as pd
from controllers.serviceController import ServiceController
from streamlit_tags import st_tags
from utils.constants import DATA_ANALYSYS_RESPONSES
import json
import re
import os
# Streamlit configuration
st.set_page_config(page_title="ML Model Advisor", layout="wide")
st.title("Analyzer Agent")

def clear_chat():
    st.session_state.dataAnalysis["analyzer"]["message"] = ""
    st.session_state.dataAnalysis["analyzer"]["variables_list"] = []
    st.session_state.dataAnalysis["analyzer"]["target_variable"] = ""
    st.session_state.dataAnalysis["analyzer"]["ml_model"] = ""

# Initialize chat history
if "dataAnalysis" not in st.session_state or st.session_state.dataAnalysis is None:
    st.session_state.dataAnalysis = DATA_ANALYSYS_RESPONSES.copy()

# File upload section (moved to main body)
uploaded_file = st.file_uploader(
    "Upload CSV file", 
    type=["csv"],
    on_change=clear_chat
)

# Fix: Only use file_path if it is a valid file and not empty, and uploaded_file is None
file_path = st.session_state["dataAnalysis"]["analyzer"].get("file_path", "")
if not uploaded_file and file_path and os.path.isfile(file_path):
    uploaded_file = open(file_path, 'rb') if file_path else None

if uploaded_file:
    try:
        # If uploaded_file is a string (from file_path), read as path; else, as file-like object
        if isinstance(uploaded_file, str):
            df = pd.read_csv(uploaded_file)
            uploaded_file_name = os.path.basename(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
            uploaded_file_name = uploaded_file.name if hasattr(uploaded_file, "name") else "analysis_data.csv"

        # Ensure nested structure exists
        if "dataAnalysis" not in st.session_state:
            st.session_state.dataAnalysis = {}
        if "analyzer" not in st.session_state.dataAnalysis:
            st.session_state.dataAnalysis["analyzer"] = {}

        # Fix duplicated folder in file_path
        default_sample_path = os.getenv("DEFAULT_SAMPLE_PATH", "")
        file_path = os.path.join(default_sample_path, uploaded_file_name)
        st.session_state["dataAnalysis"]["analyzer"]["file_path"] = file_path
        df.to_csv(file_path, index=False)
        st.session_state.dataAnalysis["analyzer"]["df"] = df.head(5)
        st.success("File successfully loaded!")
        st.write("Data Preview:")
        st.dataframe(df.head(5))
        # Display all available variables
        columns = list(df.columns)
        st_tags(
            label='#### Available variables:',
            text='',
            value=columns,
            suggestions=columns,
            maxtags = len(columns),
            key=None
            )

        # Select prediction variable
        target_variable = st.selectbox(
            "Select a prediction (target) variable",
            options=[st.session_state.dataAnalysis["analyzer"].get("target_variable", "")] + columns,
            key="target_variable"
        )
        st.session_state.dataAnalysis["analyzer"]["variables_list"] = [col for col in columns if col != target_variable]
        st.session_state.dataAnalysis["analyzer"]["target_variable"] = target_variable

    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
else:
    st.info("👆 Upload a .csv file first.")
    st.stop()

# Chat interface
if "df" in st.session_state.dataAnalysis["analyzer"]:

    prompt = st.chat_input("Ask about your data or model selection")
    target_variable = st.session_state.dataAnalysis["analyzer"].get("target_variable", None)
    if prompt is not None:
        if not target_variable or target_variable == "":
            st.warning("Please select a prediction (target) variable before asking a question.")
        else:
            st.session_state.dataAnalysis["analyzer"]["message"] = {"user": prompt}
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Analyzing data..."):
                    try:
                        agent = ServiceController.create_agent(st.session_state.dataAnalysis["analyzer"]["df"].head(5), st.session_state.dataAnalysis["analyzer"]["variables_list"], st.session_state.dataAnalysis["analyzer"]["target_variable"])

                        df = st.session_state.dataAnalysis["analyzer"]["df"]
                        response = ServiceController.get_analyzer_response(agent, df.head(5), target_variable, prompt)

                        analysis = response['output']
                        analysis = re.sub(r"<think>.*?</think>", "", analysis, flags=re.DOTALL).strip()
                        code_block = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", analysis)
                        if code_block:
                            json_text = code_block.group(1)
                        else:
                            json_match = re.search(r"(\[.*\])", analysis, re.DOTALL)
                            if json_match:
                                json_text = json_match.group(1)
                            else:
                                raise ValueError("No valid JSON found in the output.")
                        try:
                            parsed_analysis = json.loads(json_text)
                            st.session_state.dataAnalysis["analyzer"]["models"] = parsed_analysis
                            st.session_state.dataAnalysis["analyzer"]["message"]["assistant"] = parsed_analysis
                        except json.JSONDecodeError as e:
                            raise ValueError(f"JSON decoding failed: {str(e)}")
                    except Exception as e:
                        st.error(f"Analysis failed: {str(e)}")

    if st.session_state.dataAnalysis["analyzer"]["models"]:
        # st.json(st.session_state.dataAnalysis["analyzer"]["models"])  # Display the JSON in a code block
        for model in st.session_state.dataAnalysis["analyzer"]["models"]:
            model_name = model.get("ml_model", "Unknown Model")
            pros = model.get("pros", "No pros provided.")
            cons = model.get("cons", "No cons provided.")
            explanation = model.get("explanation", "No explanation provided.")

            st.markdown(f"### {model_name}")
            st.markdown(f"**Pros:** {pros}")
            st.markdown(f"**Cons:** {cons}")
            st.markdown(f"**Explanation:** {explanation}")
            st.markdown("---")  # Add a horizontal line for separation