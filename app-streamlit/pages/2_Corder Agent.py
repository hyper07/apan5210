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
    # st.code(st.session_state.dataAnalysis["coder"]["code"], language="python")


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
    try:
        df = pd.read_csv(st.session_state.dataAnalysis["analyzer"]["file_path"], encoding='utf-8', header=0)
    except ValueError as e:
        st.error(f"Agent creation failed: {str(e)}")
        st.stop()

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
                    available_variables = list(df.columns)
                    feature_variables = [col for col in available_variables if col != prediction_variable]
                    llm_prompt = f"""Generate Python code to train a '{selected_model}' model using the pandas DataFrame 'df'.
                        The file path is '{st.session_state.dataAnalysis["analyzer"].get("file_path")} for df'.
                        The target variable is '{prediction_variable}'.
                        The available feature variables are: {feature_variables}.
                        Include steps for:
                        1. Importing necessary libraries (like pandas, scikit-learn).
                        2. Convert data to correct format for ML model.
                        3. Defining features (X) and target (y).
                        4. Splitting the data into training and testing sets.
                        5. Initializing and training the '{selected_model}' model.
                        6. Making predictions on the test set (if applicable).
                        7. Evaluating the model (e.g., accuracy, MSE, R2 score, depending on the problem type).

                        Consider the data types: {dict(df.dtypes.apply(lambda x: str(x)))}
                        Here's a sample of the data: {df.head(3).to_dict()}

                        User request: {prompt}

                        Output only the Python code block, enclosed in triple backticks like this:
                        ```python
                        # Your Python code here
                        ```
                        """
                    # Stream the response from the LLM
                    response = llm.stream(llm_prompt)
                    code_output = ""
                    # Collect the streamed content
                    for chunk in response:
                        code_output += chunk
                    # Render the final accumulated response
                    # st.markdown(code_output)
                    code_output = re.sub(r"<think>.*?</think>", "", code_output, flags=re.DOTALL).strip()
                    code_block_match = re.search(r"```(?:python)?\s*([\s\S]*?)\s*```", code_output, re.DOTALL)
                    if code_block_match:
                        extracted_code = code_block_match.group(1).strip()
                    else:
                        extracted_code = code_output.strip()
                    st.code(extracted_code, language="python")
                    st.session_state.dataAnalysis["coder"]["message"]["assistant"] = extracted_code
                    st.session_state.dataAnalysis["coder"]["code"] = extracted_code


    

