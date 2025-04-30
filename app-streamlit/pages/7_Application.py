import streamlit as st
from streamlit_tags import st_tags
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain_community.llms import Ollama
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import numpy as np
import pandas as pd
import os
import time
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

lb, lmb, mb, mrb, rb = st.columns(5)
if lb.button("Analyzer", use_container_width=True):
    customAgent = AgentController.getAnalyzerAgent()
    lb.markdown(customAgent.test())
if lmb.button("Coder", use_container_width=True):
    customAgent = AgentController.getCoderAgent()
    lmb.markdown("You clicked the emoji button.")
if mb.button("Reporter", use_container_width=True):
    customAgent = AgentController.getReportAgent()
    mb.markdown(customAgent.test())
if mrb.button("Insight", use_container_width=True):
    customAgent = AgentController.getInsightAgent()
    mrb.markdown("You clicked the Material button.")    
if rb.button("Reviewer", use_container_width=True):
    customAgent = AgentController.getReviewAgent()
    rb.markdown(customAgent.test())


st.header('Overview')
# Charger les données


#TABS CONTAINERS:
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Analyzer", "Code", "Insight", "Report", "Review"])
with tab1:
    st.code('df.head(10)')
    st.markdown("Overview of the first 10 lines")
with tab2:
    st.markdown("The dimensions of the dataset: 17 variables and 11,162 rows.")
    st.code('df.shape')
with tab3:
    st.code('df.describe()')
    st.subheader("Observation")
    st.markdown("""
    - age: 50% of the values ​​are between 32 and 49 years old. Many extreme values: max 95.
    - balance: 50% of the values ​​are between 122 and 1708. Presence of negative values ​​and extreme values: min -6,847, max 81,204.
    - duration: 50% of the values ​​are between 138 sec (2min) and 496 (8min). Presence of extreme values: max 3,881.
    - campaign: 50% of the values ​​are between 1 and 3 contacts. Presence of extreme values: max 63.
    - pdays: 50% of the values ​​are between - 1 and 20. The median is -1 which means that half of the customers have never been contacted before this campaign. Presence of extreme values: max 854.
    - previous: 50% of the values ​​are between 0 and 1. Presence of extreme values: max 58.
    """)
with tab4:
    st.markdown("Data types: ")
    st.code('df.dtypes')
with tab5:
    st.markdown("No missing values: ")
    st.code('df.isna().sum()')
