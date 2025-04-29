import streamlit as st
import pandas as pd
from langchain_community.llms import Ollama
from streamlit_tags import st_tags, st_tags_sidebar
from utils.constants import DATA_ANALYSYS_RESPONSES, SAMPLE_ANALYSYS_RESPONSES
import re
import os
import io
from fpdf import FPDF
from controllers.serviceController import ServiceController
from controllers.agentController import AgentController

# Streamlit configuration
st.set_page_config(page_title="ML Model Advisor", layout="wide")
st.title("Reporter Agent")

pdf_dir = os.getenv("DEFAULT_PDF_PATH", "/tmp/files/pdf/")

# Ensure session state is initialized
if "dataAnalysis" not in st.session_state or st.session_state.dataAnalysis is None:
    st.session_state.dataAnalysis = DATA_ANALYSYS_RESPONSES.copy()

# Extract relevant variables
analyzer = st.session_state.dataAnalysis["analyzer"]
selected_model = analyzer.get("models", [])[0].get("ml_model", "") if len(analyzer.get("models", [])) > 0 else ""
target_variable = analyzer.get("target_variable", "")
feature_variables = [col for col in analyzer.get("columns", []) if col != target_variable]
code = st.session_state.dataAnalysis["coder"]["code"]
code_result = st.session_state.dataAnalysis["insight"]["script_output"] if "script_output" in st.session_state.dataAnalysis["insight"] else ""
insight = st.session_state.dataAnalysis["insight"]["message"]

# Button to generate report
col1, col2 = st.columns([2, 1])
with col1:
    generate_report = st.button("Generate Report")

if generate_report:
    # Construct the prompt for the LLM
    report_prompt = (
        f"Generate a comprehensive report based on the following information:\n\n"
        f"Selected Model: {selected_model}\n"
        f"Target Variable: {target_variable}\n"
        f"Feature Variables: {', '.join(feature_variables)}\n"
        f"Code Used:\n{code}\n"
        f"Code Result:\n{code_result}\n"
        f"Insight:\n{insight}\n"
        f"Please provide a detailed, human-readable report summarizing the analysis, results, and any recommendations."
    )

    agent = AgentController.getReportAgent()

    report_response = ""
    for chunk in agent.stream(report_prompt):
        report_response += chunk
    result_text = re.sub(r"<think>.*?</think>", "", report_response, flags=re.DOTALL).strip()
    st.session_state.dataAnalysis["reporter"]["message"] = result_text
    st.session_state.dataAnalysis["reporter"]["en"] = result_text


if st.session_state.dataAnalysis["reporter"]["message"] and st.session_state.dataAnalysis["reporter"]["message"] != "":
    # Language selection

    lang = st.radio("Select language", ["English", "Chinese", "Korean"], horizontal=True)
    translater = AgentController.getTranslateAgent()
    if lang == "Chinese" and st.session_state.dataAnalysis["reporter"].get("message"):
        with st.spinner("Translating to Chinese..."):
            translated_text = ""
            st.markdown(st.session_state.dataAnalysis['reporter']['en'])
            for chunk in translater.stream(lang, st.session_state.dataAnalysis['reporter']['message']):
                translated_text += chunk
            translated_text = re.sub(r"<think>.*?</think>", "", translated_text, flags=re.DOTALL).strip()
            st.session_state.dataAnalysis["reporter"]["cn"] = translated_text

    if lang == "Korean" and st.session_state.dataAnalysis["reporter"].get("message"):
        with st.spinner("Translating to Korean..."):
            translated_text = ""
            for chunk in translater.stream(lang, st.session_state.dataAnalysis['reporter']['message']):
                translated_text += chunk
            translated_text = re.sub(r"<think>.*?</think>", "", translated_text, flags=re.DOTALL).strip()
            st.session_state.dataAnalysis["reporter"]["kr"] = translated_text

    # Display the report in the selected language
    if lang == "English" and st.session_state.dataAnalysis["reporter"]["en"]:
        st.markdown(st.session_state.dataAnalysis["reporter"]["en"])
    elif lang == "Chinese" and st.session_state.dataAnalysis["reporter"].get("cn"):
        st.markdown(st.session_state.dataAnalysis["reporter"]["cn"])
    elif lang == "Korean" and st.session_state.dataAnalysis["reporter"].get("kr"):
        st.markdown(st.session_state.dataAnalysis["reporter"]["kr"])

    # --- PDF Export and Download Section ---

    os.makedirs(pdf_dir, exist_ok=True)
    pdf_filename = f"report_{lang.lower()}.pdf"
    pdf_path = f"{pdf_dir}/{pdf_filename}"

    # Get the report text for the selected language
    report_text = ""
    if lang == "English":
        report_text = st.session_state.dataAnalysis["reporter"].get("en", "")
    elif lang == "Chinese":
        report_text = st.session_state.dataAnalysis["reporter"].get("cn", "")
    elif lang == "Korean":
        report_text = st.session_state.dataAnalysis["reporter"].get("kr", "")

    col_pdf1, col_pdf2 = st.columns([1, 1])
    with col_pdf1:
        if st.button("Save as PDF"):
            if report_text:
                # Pass all relevant context to save_pdf
                result = ServiceController.save_pdf(
                    st.session_state.dataAnalysis["reporter"].get("kr", "") if lang == "Korean" else st.session_state.dataAnalysis["reporter"].get("cn", "") if lang == "Chinese" else st.session_state.dataAnalysis["reporter"].get("en", ""),
                    lang,
                    selected_model=selected_model,
                    target_variable=target_variable,
                    feature_variables=feature_variables,
                    code=code,
                    code_result=code_result,
                    insight=insight
                )
                if result:
                    st.session_state.dataAnalysis["reporter"]["file_path"] = result
                    st.success(f"PDF saved to {pdf_path}")
                else:
                    st.error("Failed to save PDF.")
            else:
                st.warning("No report to save for the selected language.")

    with col_pdf2:
        if os.path.exists(pdf_path):
            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="Download PDF",
                    data=f,
                    file_name=pdf_filename,
                    mime="application/pdf"
                )





