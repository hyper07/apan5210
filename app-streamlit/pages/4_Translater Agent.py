import streamlit as st
import pandas as pd
from langchain_community.llms import Ollama
from streamlit_tags import st_tags, st_tags_sidebar
from utils.constants import DATA_ANALYSYS_RESPONSES, SAMPLE_ANALYSYS_RESPONSES
import re

from controllers.agentController import AgentController
# Streamlit configuration
st.set_page_config(page_title="ML Model Advisor", layout="wide")
st.title("Translater Agent")


# Ensure session state is initialized
if "dataAnalysis" not in st.session_state or st.session_state.dataAnalysis is None:
    st.session_state.dataAnalysis = SAMPLE_ANALYSYS_RESPONSES.copy()

if "coder" not in st.session_state.dataAnalysis:
    st.session_state.dataAnalysis["coder"] = {"message": {}, "code": ""}

# Replace chat_input with text_area for user input
prompt = st.text_area("Please enter the text to translate about your data or model selection")
# Add translation buttons
col1, col2 = st.columns(2)
translate_to_chinese = col1.button("Translate to Chinese")
translate_to_korean = col2.button("Translate to Korean")

# Only process if a prompt is entered and a translation button is pressed
if prompt and (translate_to_chinese or translate_to_korean):
    with st.chat_message("assistant"):
        with st.spinner("Translating..."):
            target_language = "Chinese" if translate_to_chinese else "Korean"
            translate_prompt = (
                f"Translate the following text to {target_language}. "
                f"{prompt}"
            )

            agent = AgentController.getTranslateAgent()
            response = agent.stream(translate_prompt)
            translated_text = ""
            for chunk in response:
                translated_text += chunk
            translated_text = re.sub(r"<think>.*?</think>", "", translated_text, flags=re.DOTALL).strip()
            st.markdown(translated_text)
            st.session_state.dataAnalysis["translator"]["cn" if translate_to_chinese else "kr"] = translated_text


# Optionally, show the original prompt if entered but no translation button pressed
elif prompt:
    with st.chat_message("user"):
        st.markdown(prompt)




