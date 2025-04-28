
from pathlib import Path
import os

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


## presetting by Jing
class CodeAgent:
    def __init__(self, var1=os.getenv("DEFAULT_LLM_MODEL", "") , var2=os.getenv("DEFAULT_API_URL", "")):
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
    
    def getAPIKey(self):  

        return self.apikey
    
    def setAPIKey(self, apikey):
        self.apikey = apikey

        return self
    
    def generate(self, analysis: str) -> str:
        presetting ="You are an AI coding assistant. \
            You will receive recommended machine learning model(s) for a given dataset and analysis task. \
            Your job is to generate Python code for each recommended model. \
            If multiple models are provided, generate separate, clearly labeled Python code blocks for each one. \
            Ensure the code includes all essential steps for model training and evaluation, such as data splitting, fitting, and prediction. \
            Do not explain or justify the model choices—focus only on clean, executable Python code for each model. \
            Respond in English."

        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "You are a data analysis expert skilled in critiquing results."},
                {"role": "user",   "content": prompt}
            ],
            "temperature": 0.0
        }

        response = requests.post(self.endpoint, json=payload, headers=self.headers)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
        # st.session_state.messages.append({"role":"system", "content":presetting})