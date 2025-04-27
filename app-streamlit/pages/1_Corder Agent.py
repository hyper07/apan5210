import streamlit as st
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_community.llms import Ollama
from streamlit_tags import st_tags, st_tags_sidebar
from utils.constants import DATA_ANALYSYS_RESPONSES, SAMPLE_ANALYSYS_RESPONSES
import re
import os
# Streamlit configuration
st.set_page_config(page_title="ML Model Advisor", layout="wide")
st.title("Coder Agent")

llm = Ollama(model="deepseek-r1:14b", base_url="http://host.docker.internal:39870", verbose=True)

st.session_state.dataAnalysis = SAMPLE_ANALYSYS_RESPONSES

# def clear_chat():
#     st.session_state.dataAnalysis["analyzer"]["message"] = ""
#     st.session_state.dataAnalysis["analyzer"]["variables_list"] = []
#     st.session_state.dataAnalysis["analyzer"]["target_variable"] = ""
#     st.session_state.dataAnalysis["analyzer"]["ml_model"] = ""


# Remove file uploader, assume df is already in session state
models = st.session_state.dataAnalysis["analyzer"].get("models", [])
if len(models) > 0:
    # Select prediction variable
    prediction_variable = st.selectbox(
        "Select a model to get code",
        options=[""] + [x["ml_model"] for x in models],
        key="prediction_variable"
    )

else:
    st.info("No model available. Please back to Analyzer agent to get the model list.")
    st.stop()

# Chat interface only if a model is selected
if (
    "df" in st.session_state.dataAnalysis["analyzer"]
    and st.session_state.dataAnalysis["analyzer"].get("models")
    and st.session_state.get("selected_model", "")
):
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
    selected_model = st.session_state.get("selected_model", None)
    if prompt is not None:
        if not prediction_variable or prediction_variable == "":
            st.warning("Please select a prediction (target) variable before asking a question.")
        elif not selected_model or selected_model == "":
            st.warning("Please select a model before asking a question.")
        else:
            # Always overwrite to keep only the latest exchange
            st.session_state.dataAnalysis["analyzer"]["message"] = {"user": prompt}
            # Re-render only the latest user prompt
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Analyzing data..."):
                    try:
                        df = st.session_state.dataAnalysis["analyzer"]["df"]
                        response = agent.invoke({
                            "input": f"""Analyze this dataset and recommend machine learning models considering:
                            1. Data types: {dict(df.dtypes.apply(lambda x: str(x)))}
                            2. Sample Data: {df.head(5).to_dict()}
                            3. Available Variables: {list(df.columns)}
                            4. Target Variable: {prediction_variable}
                            5. Selected Model: {selected_model}
                            6. Problem type detection
                            {prompt}
                            And output in json format like this, surrounded by triple backticks:
                            ```
                            {[{
                                "ml_model_1": "",
                                "pros": "",
                                "cons": "",
                                "explanation": "",
                            },{
                                "ml_model_2": "",
                                "pros": "",
                                "cons": "",
                                "explanation": "",
                            },]}
                            ```
                            """,
                        })
                        analysis = response['output']
                        # Remove <think>...</think> and extract JSON inside triple backticks if present
                        analysis = re.sub(r"<think>.*?</think>", "", analysis, flags=re.DOTALL)
                        code_block = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", analysis)
                        if code_block:
                            analysis = code_block.group(1)
                        else:
                            json_start = analysis.find('{')
                            if json_start != -1:
                                analysis = analysis[json_start:]
                        st.markdown(analysis)
                        st.session_state.dataAnalysis["analyzer"]
                        st.session_state.dataAnalysis["analyzer"]["message"]["assistant"] = analysis
                    except Exception as e:
                        st.error(f"Analysis failed: {str(e)}")

    # Always display only the latest chat exchange
    last_message = st.session_state.dataAnalysis["analyzer"].get("message", {})
    if last_message:
        if "user" in last_message:
            with st.chat_message("user"):
                st.markdown(last_message["user"])
        if "assistant" in last_message:
            with st.chat_message("assistant"):
                st.markdown(last_message["assistant"])
