from pathlib import Path
import os
import re

from langchain_community.llms import Ollama
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.document_loaders import UnstructuredExcelLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.document_loaders import CSVLoader
from langchain.embeddings import OllamaEmbeddings
from streamlit_tags import st_tags, st_tags_sidebar

import streamlit as st
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import pandas as pd
from streamlit_modal import Modal  # Ensure you have the correct package installed

from contollers.serviceController import ServiceController
from pandasai import SmartDataframe
from pandasai.llm.local_llm import LocalLLM

ServiceController.initialize_llms_session_state()

file_path = "/tmp/files"


llm = Ollama(model="deepseek-r1:14b", base_url="http://host.docker.internal:39870", verbose=True)

model = LocalLLM(
    api_base="http://host.docker.internal:39870/v1",
    model="deepseek-r1:14b"
)

sample_file_path = ''
columns = []


@st.dialog("CSV Error")
def dialogBox():
    st.write("Please upload a valid utf-8 csv file")
    if st.button("Ok"):
        st.session_state["uploader_key"] += 1
        st.session_state["message"] += ""
        
        st.rerun()

def predictionVariable():
    st.session_state["dataAnalysis"]["analyzer"]["predict_variable"] = st.session_state.predictionVariable
    # st.write(st.session_state)
    #  "analyzer": {
    #     "file_path": "",
    #     "predict_variable" : "",
    #     "variables_list" : [],
    #     "ml_model" : "",
    #     "message" : "",
    # },

st.write("# Analyzer Agent")


if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 1

uploaded_file = st.file_uploader(
    "Please upload a utf-8 csv file",
    type='csv',
    key=st.session_state["uploader_key"]
)


if uploaded_file is not None:
    file_container = st.expander("Check your uploaded .csv")
    try:
        shows = pd.read_csv(uploaded_file, encoding='utf-8', header=0)
        file_path = os.getenv("DEFAULT_SAMPLE_PATH", "")+uploaded_file.name
        sample = shows.head()
        uploaded_file.seek(0)
        columns = list(sample.columns.values)
        st.session_state["dataAnalysis"]["analyzer"]["variables_list"] = columns
        keywords = st_tags(
            label='# Available variables:',
            text='',
            value=columns,
            suggestions=columns,
            maxtags = len(columns),
            key=None
            )

        file_container.write(shows)

        option = st.selectbox(
            label="Select a variable for prediction",
            options=list(sample.columns.values),
            on_change=predictionVariable,
            key="predictionVariable"
        )

        st.session_state["dataAnalysis"]["analyzer"]["file_path"] = file_path
        sample.to_csv(file_path, index=False)

        data = pd.read_csv(st.session_state["dataAnalysis"]["analyzer"]["file_path"])
        st.dataframe(data.head(5))
        df = SmartDataframe(data,{"enable_cache": False},config={"llm": model})
        # st.write(df)
        prompt = st.text_area("What do you want to ask?")

        if st.button("Ask"):
            if prompt:
                prompt = "This is the data variable descrition. "+ data.describe().to_string() +"." + prompt
                with st.spinner("Generating Request..."):
                    st.write(df.chat(prompt))

    except UnicodeDecodeError:
        dialogBox()

else:
    st.session_state.messages = []
    st.session_state["dataAnalysis"]["analyzer"]["variables_list"] = []
    st.session_state["dataAnalysis"]["analyzer"]["predict_variable"] = ""
    st.info(
        f"""
            👆 Upload a .csv file first.
            """
    )

    st.stop()

if "messages" not in st.session_state.keys(): 
        st.session_state.messages = [
            # {"role": "assistant", "content": "Please type short prompts (example: relathiship between {column name 1} and {column name 2})"}
        ]
    
if prompt := st.chat_input("Your prompt"): 
        # st.session_state.messages.append({"role": "user", "content": uploaded_file.seek(0)})
        st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: 
    with st.chat_message(message["role"]):
        st.write(message["content"])

if len(st.session_state.messages) > 0 and (st.session_state.messages[0]["role"] != "assistant" or st.session_state.messages[-1]["role"]) != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Analyzing ..."):
            df = pd.read_csv(st.session_state["dataAnalysis"]["analyzer"]["file_path"])

            loader = CSVLoader(file_path=st.session_state["dataAnalysis"]["analyzer"]["file_path"],
                csv_args={
                    'delimiter': ',',
                    'quotechar': '"',
                    'fieldnames': columns
                })
            data = loader.load()
            # st.write(data)
            embeddings = HuggingFaceEmbeddings()
            index_creator = VectorstoreIndexCreator(embedding=embeddings)
            docsearch = index_creator.from_loaders([loader])
            chain=RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=docsearch.vectorstore.as_retriever(),
                input_key="question"
                )

            query= "This is the variables description. " + df.describe().to_string() +"." + prompt
            # query="what type of machine learning model do you recommend to analyze relationship between joined date and inducted date from attached data? Can you give me the machine learning model list only?"
            response=chain({"question":query})
            cleaned_result = re.sub(r'<think>.*?</think>', '', response['result'], flags=re.DOTALL)
            # cleaned_result = cleaned_result.replace("content includes", "")
        
            message = {"role": "assistant", "content": cleaned_result}
            # st.write(response['result'])
            
            st.session_state.messages.append(message) 
