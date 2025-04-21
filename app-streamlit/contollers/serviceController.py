import streamlit as st
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
from agents.analyzerAgent import AnalyzerAgent
from agents.codeAgent import CodeAgent
from agents.insightAgent import InsightAgent
from agents.reportAgent import ReportAgent
from agents.reviewAgent import ReviewAgent
from agents.translateAgent import TranslateAgent
import requests

class ServiceController:

    def __init__(self):
        self.api_url = os.getenv("DEFAULT_API_URL", "")  # Replace with the correct hostname or IP if different
        self.model = os.getenv("DEFAULT_LLM_MODEL", "") 

    def setAPIUrl(self, url):
        self.api_url = url
        return self

    def getAPIUrl(self):
       return self.api_url
    
    def setModel(self, model):
        self.model = model
        return self 
    
    def getModel(self):
        return self.model
    
    def pullModelFromSite(self, model):
        try:
            response = requests.post(self.api_url+"/api/pull", json={"model": model, "stream": False})  # Replace with the correct hostname or IP if different
            return response.json()["status"]
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching Ollama model list: {e}")
            return None


    def getLLMlist(self):
        try:
            response = requests.get(self.api_url+"/api/tags")  # Replace with the correct hostname or IP if different
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching Ollama model list: {e}")
            return None

    def getModelListOnly(self):
        models = []
        model_list = self.getLLMlist()
        if model_list is not None:
            for model in model_list.get("models", []):
                models.append(model.get("name"))
            return models
        
        return None

    def initialize_llms_session_state():
        if 'llms' not in st.session_state:
            nav_script = """
                <meta http-equiv="refresh" content="0; url='/'">
            """
            st.write(nav_script, unsafe_allow_html=True)

