import streamlit as st
import pandas as pd
from langchain_community.llms import Ollama
from streamlit_tags import st_tags, st_tags_sidebar
from utils.constants import DATA_ANALYSYS_RESPONSES, SAMPLE_ANALYSYS_RESPONSES
import re
import os
import tempfile

from controllers.serviceController import ServiceController

# Streamlit configuration
st.set_page_config(page_title="ML Model Advisor", layout="wide")
st.title("Insight Agent")

llm = Ollama(model="phi3:latest", base_url="http://host.docker.internal:39870", verbose=True)

if "dataAnalysis" not in st.session_state or st.session_state.dataAnalysis is None:
    st.session_state.dataAnalysis = SAMPLE_ANALYSYS_RESPONSES

# Save the code to a temporary file
# Run the extracted code and display output/errors
if st.button("Run Code"):
    stdout, stderr = ServiceController.run_python_script(st.session_state.dataAnalysis["coder"]["code"])
    st.session_state.dataAnalysis["insight"]["script_output"] = stdout
    st.session_state.dataAnalysis["insight"]["script_errors"] = stderr

# Display the output and errors if they exist in session state
if "script_output" in st.session_state.dataAnalysis["insight"] and st.session_state.dataAnalysis["insight"]["script_output"]:
    st.subheader("Script Output")
    st.text(st.session_state.dataAnalysis["insight"]["script_output"])
if "script_errors" in st.session_state.dataAnalysis["insight"] and st.session_state.dataAnalysis["insight"]["script_errors"]:
    st.subheader("Script Errors")
    st.text(st.session_state.dataAnalysis["insight"]["script_errors"])


# Chat interface only if a model is selected
if (
    "script_output" in st.session_state.dataAnalysis["insight"] and st.session_state.dataAnalysis["insight"]["script_output"]
    and st.session_state.dataAnalysis["analyzer"].get("models")
    and st.session_state.dataAnalysis["analyzer"].get("models") != []
    and st.session_state.dataAnalysis["analyzer"].get("file_path")
    and st.session_state.dataAnalysis["analyzer"].get("file_path") != ""
):
    # Only use chat_input for user prompt
    insight_prompt = st.chat_input("Ask about insight or best practice")
    if insight_prompt:
        with st.spinner("Getting industry insight..."):
            analyzer = st.session_state.dataAnalysis["analyzer"]
            selected_model = analyzer.get("models", [{}])[0].get("ml_model", "")
            prediction_variable = analyzer.get("target_variable", "")
            feature_variables = [col for col in analyzer.get("columns", []) if col != prediction_variable]
            extracted_code = st.session_state.dataAnalysis["coder"]["code"]
            insight_llm_prompt = f"""
            You are an industry expert. Given the following:
            - Model: {selected_model}
            - Target variable: {prediction_variable}
            - Feature variables: {feature_variables}
            - Python code used for modeling:
            ```python
            {extracted_code}
            ```
            - Output of the code:
            ```python
            {st.session_state.dataAnalysis["insight"]["script_output"]}
            ```
            Provide an industry-specific insight or best practice in response to this user request:
            '{insight_prompt}'
            """
            insight_response = ""
            for chunk in llm.stream(insight_llm_prompt):
                insight_response += chunk
            st.markdown(insight_response.strip())





