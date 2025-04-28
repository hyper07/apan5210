import streamlit as st
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_community.llms import Ollama
from streamlit_tags import st_tags
from utils.constants import DATA_ANALYSYS_RESPONSES
import json
import re
import os
# Streamlit configuration
st.set_page_config(page_title="ML Model Advisor", layout="wide")
st.title("Analyzer Agent")

llm = Ollama(model="deepseek-r1:1.5b", base_url="http://host.docker.internal:39870", verbose=True)

def clear_chat():
    st.session_state.dataAnalysis["analyzer"]["message"] = ""
    st.session_state.dataAnalysis["analyzer"]["variables_list"] = []
    st.session_state.dataAnalysis["analyzer"]["target_variable"] = ""
    st.session_state.dataAnalysis["analyzer"]["ml_model"] = ""

# Initialize chat history
if "dataAnalysis" not in st.session_state:
    st.session_state.dataAnalysis = DATA_ANALYSYS_RESPONSES


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
        prediction_variable = st.selectbox(
            "Select a prediction (target) variable",
            options=[""] + columns,
            key="prediction_variable"
        )
        st.session_state.dataAnalysis["analyzer"]["variables_list"] = [col for col in columns if col != prediction_variable]
        st.session_state.dataAnalysis["analyzer"]["target_variable"] = prediction_variable

    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
else:
    st.info("👆 Upload a .csv file first.")
    st.stop()

# Chat interface
if "df" in st.session_state.dataAnalysis["analyzer"]:
    try:
        agent = create_pandas_dataframe_agent(
            llm=llm,
            df=st.session_state.dataAnalysis["analyzer"]["df"],
            verbose=True,
            max_iterations=1000,
            agent_type="zero-shot-react-description",
            extra_tools=[],
            handle_parsing_errors=True,
            allow_dangerous_code=True
        )
    except ValueError as e:
        st.error(f"Agent creation failed: {str(e)}")
        st.stop()

    prompt = st.chat_input("Ask about your data or model selection")
    prediction_variable = st.session_state.get("prediction_variable", None)
    if prompt is not None:
        if not prediction_variable or prediction_variable == "":
            st.warning("Please select a prediction (target) variable before asking a question.")
        else:
            # Always overwrite to keep only the latest exchange
            st.session_state.dataAnalysis["analyzer"]["message"] = {"user": prompt}
            # Re-render only the latest user prompt
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Analyzing data..."):
                    max_retries = 60
                    retry_count = 0

                    try:
                        df = st.session_state.dataAnalysis["analyzer"]["df"]
                      

                        while retry_count <= max_retries:
                            try:
                                response = agent.invoke({
                                    "input": f"""For analyzing this dataset and recommend machine learning models considering:
                                    1. Data types: {dict(df.dtypes.apply(lambda x: str(x)))}
                                    2. Sample Data: {df.head(5).to_dict()}
                                    3. Available Variables: {list(df.columns)}
                                    4. Target Variable: {prediction_variable}
                                    {prompt}
                                    Output ONLY in JSON format, inside triple backticks like this:
                                    ```
                                    [{{
                                        "ml_model": "Model name",
                                        "pros": "Pros of model",
                                        "cons": "Cons of model",
                                        "explanation": "Explanation why model fits"
                                    }}]
                                    ```
                                    DO NOT include any other text outside the triple backticks."""
                                })
                                break  # Success
                            except Exception as e:
                                retry_count += 1
                                if retry_count > max_retries:
                                    raise e  # If still broken after retries, raise error
                                # else:
                                #     st.warning(f"Parsing failed, retrying ({retry_count}/{max_retries})...")

                        # Then process response as normal
                        analysis = response['output']

                         # First, remove any <think>...</think> tags
                        analysis = re.sub(r"<think>.*?</think>", "", analysis, flags=re.DOTALL).strip()

                        # Try to find JSON inside triple backticks
                        code_block = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", analysis)

                        if code_block:
                            json_text = code_block.group(1)
                        else:
                            # No triple backticks, fallback to extract JSON manually
                            json_match = re.search(r"(\[.*\])", analysis, re.DOTALL)
                            if json_match:
                                json_text = json_match.group(1)
                            else:
                                raise ValueError("No valid JSON found in the output.")

                        # Now safely parse the JSON
                        try:
                            parsed_analysis = json.loads(json_text)

                            st.session_state.dataAnalysis["analyzer"]["models"] = parsed_analysis
                            st.session_state.dataAnalysis["analyzer"]["message"]["assistant"] = parsed_analysis
                        except json.JSONDecodeError as e:
                            raise ValueError(f"JSON decoding failed: {str(e)}")

                        # Use parsed_analysis as your final output
                        # Display the parsed analysis in a more readable markdown format
   
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