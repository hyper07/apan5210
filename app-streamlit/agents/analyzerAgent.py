from pathlib import Path
import os

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.document_loaders import UnstructuredExcelLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_community.llms import Ollama

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.document_loaders import CSVLoader
from langchain.embeddings import OllamaEmbeddings
from streamlit_tags import st_tags, st_tags_sidebar

import streamlit as st
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import pandas as pd

class AnalyzerAgent:
    def __init__(self, var1=os.getenv("DEFAULT_ANALYZER_LLM_MODEL", "") , var2=os.getenv("DEFAULT_API_URL", "")):
        self.llmModel = var1
        self.llmUrl = var2

    def getModel(self):  

        return self.llmModel
    
    def setModel(self, model):  

        self.llmModel = model

        return self
    
    def getUrl(self):  

        return self.llmUrl
    
    def setUrl(self, url):
        self.llmUrl = url

        return self

    def create_agent(self, df):
        llm = Ollama(model=self.llmModel, base_url=self.llmUrl, verbose=True)
        agent = create_pandas_dataframe_agent(
            llm=llm,
            df=df,
            verbose=True,
            max_iterations=1000,
            agent_type="zero-shot-react-description",
            extra_tools=[],
            handle_parsing_errors=True,
            allow_dangerous_code=True
        )
        return agent

    def analyze_data(self, agent, df, prediction_variable, prompt):
        response = agent.invoke({
            "input": f"""For analyzing this dataset and recommend machine learning models considering:
            1. Data types: {dict(df.dtypes.apply(lambda x: str(x)))}
            2. Sample Data: {df.to_dict()}
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
        return response
