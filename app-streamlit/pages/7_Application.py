
import pandas as pd
import os
import json
import re

import streamlit as st
from streamlit_tags import st_tags
from langchain.document_loaders import PDFMinerLoader 
from streamlit_ace import st_ace
from controllers.agentController import AgentController
from controllers.serviceController import ServiceController
from utils.constants import DATA_ANALYSYS_RESPONSES

ServiceController.initialize_llms_session_state()

def clear_chat():
    st.session_state.dataAnalysis["analyzer"]["message"] = ""
    st.session_state.dataAnalysis["analyzer"]["variables_list"] = []
    st.session_state.dataAnalysis["analyzer"]["target_variable"] = ""
    st.session_state.dataAnalysis["analyzer"]["ml_model"] = ""

st.title("APPLICATION")

pdf_dir = os.getenv("DEFAULT_PDF_PATH", "/tmp/files/pdf/")
if "dataAnalysis" not in st.session_state or st.session_state.dataAnalysis is None:
    st.session_state.dataAnalysis = DATA_ANALYSYS_RESPONSES.copy()

# Upload a PDF file
uploaded_file = st.file_uploader(
    "Upload CSV file", 
    type=["csv"],
    on_change=clear_chat
)

if uploaded_file is not None:
    try:
        # If uploaded_file is a string (from file_path), read as path; else, as file-like object
        if isinstance(uploaded_file, str):
            df = pd.read_csv(uploaded_file)
            uploaded_file_name = os.path.basename(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
            uploaded_file_name = uploaded_file.name if hasattr(uploaded_file, "name") else "analysis_data.csv"

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


# Inject CSS to make tab font bigger
st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size:1.6rem;
    }
</style>
""", unsafe_allow_html=True)


#TABS CONTAINERS:
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Analyzer", "Code", "Insight", "Report", "Review"])
with tab1:
    # add button "Analyze" to get run the code below
    if st.button("Analyze", key="analyze_tab1"):
        analyzerAgent = AgentController.getAnalyzerAgent()
        df = st.session_state.dataAnalysis["analyzer"]["df"]
        stream_generator = analyzerAgent.stream(df.head(5), target_variable, "Give me top 3 ML models")
        analysis = st.write_stream(stream_generator)

        # Extract JSON (ensure analysis is treated as string)
        json_text = None
        analysis_str = str(analysis) # Ensure it's a string
        code_block = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", analysis_str, re.DOTALL)
        if code_block:
            json_text = code_block.group(1).strip()
        else:
            # Fallback checks (similar to analyzerAgent logic)
            cleaned_response = analysis_str.strip()
            if (cleaned_response.startswith('{') and cleaned_response.endswith('}')) or \
                (cleaned_response.startswith('[') and cleaned_response.endswith(']')):
                json_text = cleaned_response
            else:
                json_match = re.search(r"(\[.*\])", analysis_str, re.DOTALL)
                if json_match:
                    json_text = json_match.group(1).strip()

        if json_text is None:
                st.error("Could not extract valid JSON from the streamed response.")
                raise ValueError("No valid JSON found in the output after streaming.")

        try:
            # Clean potential artifacts before parsing
            json_text = json_text.replace('\\n', '\n').replace('\\"', '"') # Basic cleaning
            parsed_analysis = json.loads(json_text)
            st.session_state.dataAnalysis["analyzer"]["models"] = parsed_analysis
            st.session_state.dataAnalysis["analyzer"]["message"] = analysis_str

        except json.JSONDecodeError as e:
            st.error(f"JSON decoding failed after streaming: {str(e)}")
            # st.text_area("Failed JSON Text:", json_text, height=150)
            raise ValueError(f"JSON decoding failed: {str(e)}")

with tab2:
    
    models = st.session_state.dataAnalysis["analyzer"].get("models", [])
    if len(models) > 0:
        # Select prediction variable
        selected_model = st.selectbox(
            "Select a model to get code",
            options=[""] + [x["ml_model"] for x in models],
            key="selected_model"
        )

    else:
        st.info("No model available. Please back to Analyzer agent to get the model list.")
        st.stop()

    if (
        st.session_state.dataAnalysis["analyzer"].get("models")
        and st.session_state.dataAnalysis["analyzer"].get("models") != []
        and st.session_state.dataAnalysis["analyzer"].get("file_path")
        and st.session_state.dataAnalysis["analyzer"].get("file_path") != ""
    ):
        if st.button("Generate Code", key="coder_tab2"):
            prediction_variable = st.session_state.dataAnalysis["analyzer"].get("target_variable", None)
 
            if not prediction_variable or prediction_variable == "":
                st.warning("Please select a prediction (target) variable before asking a question.")
            elif not selected_model or selected_model == "":
                st.warning("Please select a model before asking a question.")
            else:
                user_prompt = f"Give me code for {selected_model} model"
                st.session_state.dataAnalysis["coder"]["message"] = {"user": user_prompt}
                st.session_state.dataAnalysis["coder"]["selected_model"] = selected_model


                with st.chat_message("assistant"):
                    with st.spinner("Generating code..."):

                        # Instantiate CodeAgent
                        agent = AgentController.getCoderAgent()

                        # Call generate_code with all required arguments
                        code_generator = agent.generate_code(
                            selected_model=st.session_state.dataAnalysis["coder"]["selected_model"],
                            file_path=st.session_state.dataAnalysis["analyzer"].get("file_path"),
                            prediction_variable=prediction_variable,
                            user_prompt=user_prompt
                        )

                        response = st.write_stream(code_generator)

                        code_output = ""
                        for chunk in response:
                            code_output += chunk
                        code_output = re.sub(r"<think>.*?</think>", "", code_output, flags=re.DOTALL).strip()
                        code_block_match = re.search(r"```(?:python)?\s*([\s\S]*?)\s*```", code_output, re.DOTALL)
                        if code_block_match:
                            extracted_code = code_block_match.group(1).strip()
                        else:
                            extracted_code = code_output.strip()
                        st.session_state.dataAnalysis["coder"]["code"] = extracted_code


with tab3:

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

    if st.button("What Insight?", key="insight_tab3"):
        

        with st.spinner("Getting industry insight..."):
            analyzer = st.session_state.dataAnalysis["analyzer"]
            selected_model = st.session_state.dataAnalysis["coder"].get("selected_model","")
            prediction_variable = analyzer.get("target_variable", "")
            feature_variables = [col for col in analyzer.get("variables_list", []) if col != prediction_variable]
            extracted_code = st.session_state.dataAnalysis["coder"]["code"]

            agent = AgentController.getInsightAgent()

            # Pass all required arguments to critique
            insight_generator = agent.critique(
                selected_model=selected_model,
                prediction_variable=prediction_variable,
                feature_variables=feature_variables,
                extracted_code=extracted_code,
                insight_prompt="insight"
            )

            response = st.write_stream(insight_generator)

            insight_response = ""
            for chunk in response:
                insight_response += chunk

            response_text = re.sub(r"<think>.*?</think>", "", insight_response, flags=re.DOTALL).strip()
            st.session_state.dataAnalysis["insight"]["message"] = response_text
            st.session_state.dataAnalysis["insight"]["en"] = response_text
            st.rerun()

    # --- Translation and language selection ---
    if "message" in st.session_state.dataAnalysis["insight"] and st.session_state.dataAnalysis["insight"]["message"]:
        lang = st.radio("Select language", ["English", "Chinese", "Korean"], horizontal=True, key="insight_lang")
        
        translater = AgentController.getTranslateAgent()

        # Automatic translation when language is changed and translation is empty
        if lang == "Chinese" and st.session_state.dataAnalysis["insight"].get("message"):
            if not st.session_state.dataAnalysis["insight"].get("cn"):
                with st.spinner("Translating to Chinese..."):
                    
                    response = st.write_stream(translater.stream(lang, st.session_state.dataAnalysis["insight"].get("message")))
                    translated_text = ""
                    for chunk in response:
                        translated_text += chunk
                    translated_text = re.sub(r"<think>.*?</think>", "", translated_text, flags=re.DOTALL).strip()
                    st.session_state.dataAnalysis["insight"]["cn"] = translated_text

        if lang == "Korean" and st.session_state.dataAnalysis["insight"].get("message"):
            if not st.session_state.dataAnalysis["insight"].get("kr"):
                with st.spinner("Translating to Korean..."):

                    response = st.write_stream(translater.stream(lang, st.session_state.dataAnalysis["insight"].get("message")))
                    translated_text = ""
                    for chunk in response:
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

with tab4:
    if st.button("Generate Report", key="report_tab4"):

        report_prompt = (
            f"Generate a comprehensive report based on the following information:\n\n"
            f"Selected Model: {st.session_state.dataAnalysis['coder'].get('selected_model','')}\n"
            f"Target Variable: {st.session_state.dataAnalysis['analyzer']['target_variable']}\n"
            f"Feature Variables: {', '.join(st.session_state.dataAnalysis['analyzer'].get('variables_list',[]))}\n"
            f"Code Used:\n{st.session_state.dataAnalysis['coder']['code']}\n"
            f"Code Result:\n{st.session_state.dataAnalysis['insight']['script_output'] if 'script_output' in st.session_state.dataAnalysis['insight'] else ''}\n"
            f"Insight:\n{st.session_state.dataAnalysis['insight']['message']}\n"
            f"Please provide a detailed, human-readable report summarizing the analysis, results, and any recommendations."
        )

        agent = AgentController.getReportAgent()
        response = st.write_stream(agent.stream(report_prompt))

        report_response = ""
        for chunk in response:
            report_response += chunk
        result_text = re.sub(r"<think>.*?</think>", "", report_response, flags=re.DOTALL).strip()
        st.session_state.dataAnalysis["reporter"]["message"] = result_text
        st.session_state.dataAnalysis["reporter"]["en"] = result_text
        st.rerun()
        
        # --- Translation and language selection ---
    if "message" in st.session_state.dataAnalysis["reporter"] and st.session_state.dataAnalysis["reporter"]["message"]:
        lang = st.radio("Select language", ["English", "Chinese", "Korean"], horizontal=True, key="reporter_lang")
        
        translater = AgentController.getTranslateAgent()

        # Automatic translation when language is changed and translation is empty
        if lang == "Chinese" and st.session_state.dataAnalysis["reporter"].get("message"):
            if not st.session_state.dataAnalysis["reporter"].get("cn"):
                with st.spinner("Translating to Chinese..."):
                    
                    response = st.write_stream(translater.stream(lang, st.session_state.dataAnalysis["reporter"].get("message")))
                    translated_text = ""
                    for chunk in response:
                        translated_text += chunk
                    translated_text = re.sub(r"<think>.*?</think>", "", translated_text, flags=re.DOTALL).strip()
                    st.session_state.dataAnalysis["reporter"]["cn"] = translated_text

        if lang == "Korean" and st.session_state.dataAnalysis["reporter"].get("message"):
            if not st.session_state.dataAnalysis["reporter"].get("kr"):
                with st.spinner("Translating to Korean..."):

                    response = st.write_stream(translater.stream(lang, st.session_state.dataAnalysis["reporter"].get("message")))
                    translated_text = ""
                    for chunk in response:
                        translated_text += chunk
                    translated_text = re.sub(r"<think>.*?</think>", "", translated_text, flags=re.DOTALL).strip()
                    st.session_state.dataAnalysis["reporter"]["kr"] = translated_text

        # Display the reporter in the selected language
        if lang == "English" and st.session_state.dataAnalysis["reporter"].get("en"):
            st.markdown(st.session_state.dataAnalysis["reporter"]["en"])
        elif lang == "Chinese" and st.session_state.dataAnalysis["reporter"].get("cn"):
            st.markdown(st.session_state.dataAnalysis["reporter"]["cn"])
        elif lang == "Korean" and st.session_state.dataAnalysis["reporter"].get("kr"):
            st.markdown(st.session_state.dataAnalysis["reporter"]["kr"])
    
        os.makedirs(pdf_dir, exist_ok=True)
        pdf_filename = f"report_{lang.lower()}.pdf"
        pdf_path = f"{pdf_dir}{pdf_filename}"

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

                    feature_variables = [col for col in st.session_state.dataAnalysis["analyzer"].get("variables_list", []) if col != st.session_state.dataAnalysis["analyzer"].get("target_variable", "")]

                    # Pass all relevant context to save_pdf
                    result = ServiceController.save_pdf(
                        report_text,
                        lang,
                        selected_model=selected_model,
                        target_variable=target_variable,
                        feature_variables=feature_variables,
                        code=st.session_state.dataAnalysis['coder']['code'],
                        code_result=st.session_state.dataAnalysis['insight']['script_output'],
                        insight=st.session_state.dataAnalysis['insight']['message']
                    )
                    if result:
                        st.session_state.dataAnalysis["reporter"]["file_path"] = pdf_path
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
with tab5:
    if st.button("Final Review", key="review_tab5"):
        with st.spinner("Reviewing report..."):
            report_path = st.session_state.dataAnalysis["reporter"].get("file_path")
            loader = PDFMinerLoader(report_path)
            documents = loader.load()
            full_text = " ".join([doc.page_content for doc in documents])
            review_agent = AgentController.getReviewAgent() 
            response = st.write_stream(review_agent.stream(full_text))

            reviewer_response = ""
            for chunk in response:
                if isinstance(chunk, dict):
                    reviewer_response += chunk.get("generated_text", "")
                else:
                    reviewer_response += str(chunk)
            result_text = re.sub(r"<think>.*?</think>", "", reviewer_response, flags=re.DOTALL).strip()
            st.session_state.dataAnalysis["reviewer"]["message"] = result_text
            st.session_state.dataAnalysis["reviewer"]["en"] = result_text

        st.success("Review completed!")
        st.markdown("#### Review Output:")
        st.rerun()
    
    # --- Language selection and translation ---
    if "reviewer" in st.session_state.dataAnalysis and st.session_state.dataAnalysis["reviewer"].get("message"):
        if "last_review_lang" not in st.session_state:
            st.session_state["last_review_lang"] = "English"

        lang = st.radio(
            "Select language",
            ["English", "Chinese", "Korean"],
            horizontal=True,
            index=["English", "Chinese", "Korean"].index(st.session_state["last_review_lang"]),
            key="review_lang"
        )

        # Only translate if language changed
        if lang != st.session_state["last_review_lang"]:
            translater = AgentController.getTranslateAgent()
            if lang == "Chinese" and st.session_state.dataAnalysis["reviewer"].get("message"):
                with st.spinner("Translating to Chinese..."):

                    response = st.write_stream(translater.stream(lang, st.session_state.dataAnalysis['reviewer']['message']))

                    translated_text = ""
                    for chunk in response:
                        if isinstance(chunk, dict):
                            translated_text += chunk.get("text", str(chunk))
                        else:
                            translated_text += chunk
                    translated_text = re.sub(r"<think>.*?</think>", "", translated_text, flags=re.DOTALL).strip()
                    st.session_state.dataAnalysis["reviewer"]["cn"] = translated_text

            if lang == "Korean" and st.session_state.dataAnalysis["reviewer"].get("message"):
                with st.spinner("Translating to Korean..."):

                    response = st.write_stream(translater.stream(lang, st.session_state.dataAnalysis['reviewer']['message']))

                    translated_text = ""
                    for chunk in response:
                        if isinstance(chunk, dict):
                            translated_text += chunk.get("text", str(chunk))
                        else:
                            translated_text += chunk
                    translated_text = re.sub(r"<think>.*?</think>", "", translated_text, flags=re.DOTALL).strip()
                    st.session_state.dataAnalysis["reviewer"]["kr"] = translated_text

            st.session_state["last_review_lang"] = lang

        # Display the review in the selected language
        if lang == "English" and st.session_state.dataAnalysis["reviewer"].get("en"):
            st.markdown(st.session_state.dataAnalysis["reviewer"]["en"])
        elif lang == "Chinese" and st.session_state.dataAnalysis["reviewer"].get("cn"):
            st.markdown(st.session_state.dataAnalysis["reviewer"]["cn"])
        elif lang == "Korean" and st.session_state.dataAnalysis["reviewer"].get("kr"):
            st.markdown(st.session_state.dataAnalysis["reviewer"]["kr"])