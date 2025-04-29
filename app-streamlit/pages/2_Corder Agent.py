import streamlit as st
import pandas as pd
from langchain_community.llms import Ollama
from streamlit_tags import st_tags, st_tags_sidebar
from utils.constants import DATA_ANALYSYS_RESPONSES, SAMPLE_ANALYSYS_RESPONSES
import re
import os
from controllers.agentController import AgentController


# Streamlit configuration
st.set_page_config(page_title="ML Model Advisor", layout="wide")
st.title("Coder Agent")

st.session_state.dataAnalysis = SAMPLE_ANALYSYS_RESPONSES if st.session_state.dataAnalysis is None else st.session_state.dataAnalysis

# Remove file uploader, assume df is already in session state
models = st.session_state.dataAnalysis["analyzer"].get("models", [])
if len(models) > 0:
    # Select prediction variable
    selected_model = st.selectbox(
        "Select a model to get code",
        options=[""] + [x["ml_model"] for x in models],
        key="prediction_variable"
    )

else:
    st.info("No model available. Please back to Analyzer agent to get the model list.")
    st.stop()

# Chat interface only if a model is selected
if (
    st.session_state.dataAnalysis["analyzer"].get("models")
    and st.session_state.dataAnalysis["analyzer"].get("models") != []
    and st.session_state.dataAnalysis["analyzer"].get("file_path")
    and st.session_state.dataAnalysis["analyzer"].get("file_path") != ""
):
    prompt = st.chat_input("Ask about your data or model selection")
    prediction_variable = st.session_state.dataAnalysis["analyzer"].get("target_variable", None)
    if prompt is not None:
        if not prediction_variable or prediction_variable == "":
            st.warning("Please select a prediction (target) variable before asking a question.")
        elif not selected_model or selected_model == "":
            st.warning("Please select a model before asking a question.")
        else:
            st.session_state.dataAnalysis["coder"]["message"] = {"user": prompt}
            st.session_state.dataAnalysis["coder"]["selected_model"] = selected_model

            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Generating code..."):

                    # Instantiate CodeAgent
                    agent = AgentController.getCoderAgent()

                    # Call generate_code with all required arguments
                    response = agent.generate_code(
                        selected_model=selected_model,
                        file_path=st.session_state.dataAnalysis["analyzer"].get("file_path"),
                        prediction_variable=prediction_variable,
                        user_prompt=prompt
                    )

                    code_output = ""
                    for chunk in response:
                        code_output += chunk
                    code_output = re.sub(r"<think>.*?</think>", "", code_output, flags=re.DOTALL).strip()
                    code_block_match = re.search(r"```(?:python)?\s*([\s\S]*?)\s*```", code_output, re.DOTALL)
                    if code_block_match:
                        extracted_code = code_block_match.group(1).strip()
                    else:
                        extracted_code = code_output.strip()
                    st.code(extracted_code, language="python")
                    st.session_state.dataAnalysis["coder"]["message"]["assistant"] = extracted_code
                    st.session_state.dataAnalysis["coder"]["code"] = extracted_code

