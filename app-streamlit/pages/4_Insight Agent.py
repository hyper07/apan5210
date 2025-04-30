import streamlit as st
import pandas as pd
from langchain_community.llms import Ollama
from streamlit_tags import st_tags, st_tags_sidebar
from utils.constants import DATA_ANALYSYS_RESPONSES, SAMPLE_ANALYSYS_RESPONSES
import re
from controllers.serviceController import ServiceController
from controllers.agentController import AgentController
from streamlit_ace import st_ace

# Streamlit configuration
st.set_page_config(page_title="ML Model Advisor", layout="wide")
st.title("Insight Agent")

if "dataAnalysis" not in st.session_state or st.session_state.dataAnalysis is None:
    st.session_state.dataAnalysis = DATA_ANALYSYS_RESPONSES.copy()

# Add a code editor for editing the code before running
code_text = st_ace(
    value=st.session_state.dataAnalysis["coder"]["code"],
    language="python",
    theme="monokai",
    key="code_editor",
    height=300,
    font_size=14,
    tab_size=4,
    show_gutter=True,
    show_print_margin=False,
    wrap=True,
    auto_update=True
)
st.session_state.dataAnalysis["coder"]["code"] = code_text

# Save the code to a temporary file
# Run the extracted code and display output/errors
if st.button("Run Code"):
    # Use the updated code from the text area
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

            agent = AgentController.getInsightAgent()

            # Pass all required arguments to critique
            insight_llm = agent.critique(
                selected_model=selected_model,
                prediction_variable=prediction_variable,
                feature_variables=feature_variables,
                extracted_code=extracted_code,
                insight_prompt=insight_prompt
            )
            insight_response = ""
            for chunk in insight_llm:
                insight_response += chunk

            response_text = re.sub(r"<think>.*?</think>", "", insight_response, flags=re.DOTALL).strip()
            st.session_state.dataAnalysis["insight"]["message"] = response_text
            st.session_state.dataAnalysis["insight"]["en"] = response_text

# --- Language selection and translation ---
if "message" in st.session_state.dataAnalysis["insight"] and st.session_state.dataAnalysis["insight"]["message"]:
    lang = st.radio("Select language", ["English", "Chinese", "Korean"], horizontal=True)
    
    translater = AgentController.getTranslateAgent()

    # Automatic translation when language is changed and translation is empty
    if lang == "Chinese" and st.session_state.dataAnalysis["insight"].get("message"):
        with st.spinner("Translating to Chinese..."):

            translated_text = ""
            for chunk in translater.stream(lang, st.session_state.dataAnalysis["insight"].get("message")):
                translated_text += chunk
            translated_text = re.sub(r"<think>.*?</think>", "", translated_text, flags=re.DOTALL).strip()
            st.session_state.dataAnalysis["insight"]["cn"] = translated_text

    if lang == "Korean" and st.session_state.dataAnalysis["insight"].get("message"):
        with st.spinner("Translating to Korean..."):

            translated_text = ""
            for chunk in translater.stream(lang, st.session_state.dataAnalysis["insight"].get("message")):
                translated_text += chunk
            translated_text = re.sub(r"<think>.*?</think>", "", translated_text, flags=re.DOTALL).strip()
            st.session_state.dataAnalysis["insight"]["kr"] = translated_text

    # Display the insight in the selected language
    if lang == "English" and st.session_state.dataAnalysis["insight"].get("en"):
        st.markdown(st.session_state.dataAnalysis["insight"]["en"])
    elif lang == "Chinese" and st.session_state.dataAnalysis["insight"].get("cn"):
        st.markdown(st.session_state.dataAnalysis["insight"]["cn"])
    elif lang == "Korean" and st.session_state.dataAnalysis["insight"].get("kr"):
        st.markdown(st.session_state.dataAnalysis["insight"]["kr"])




