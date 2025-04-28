import streamlit as st
import pandas as pd
from langchain_community.llms import Ollama
from streamlit_tags import st_tags, st_tags_sidebar
from utils.constants import DATA_ANALYSYS_RESPONSES, SAMPLE_ANALYSYS_RESPONSES
import re
import os
# Streamlit configuration
st.set_page_config(page_title="ML Model Advisor", layout="wide")
st.title("Coder Agent")

llm = Ollama(model="qwen2.5-coder:3b", base_url="http://host.docker.internal:39870", verbose=True)

st.session_state.dataAnalysis = SAMPLE_ANALYSYS_RESPONSES


prompt = st.chat_input("Ask about your data or model selection")
prediction_variable = st.session_state.dataAnalysis["analyzer"].get("target_variable", None)
if prompt is not None:

    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Generating code..."):
       
            llm_prompt = prompt
            # Stream the response from the LLM
            response = llm.stream(llm_prompt)
            code_output = ""
            # Collect the streamed content
            for chunk in response:
                code_output += chunk
            # Render the final accumulated response
            st.markdown(code_output)
            # Remove <think> tags and extract code block
            code_output = re.sub(r"<think>.*?</think>", "", code_output, flags=re.DOTALL).strip()
            code_block_match = re.search(r"```(?:python)?\s*([\s\S]*?)\s*```", code_output, re.DOTALL)
            if code_block_match:
                extracted_code = code_block_match.group(1).strip()
            else:
                extracted_code = code_output.strip()
            st.code(extracted_code, language="python")
            st.session_state.dataAnalysis["coder"]["message"]["assistant"] = extracted_code
            st.session_state.dataAnalysis["coder"]["code"] = extracted_code

