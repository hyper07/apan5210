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


class AgentController:
    def __init__(self):
        self.api_url = os.getenv("DEFAULT_API_URL", "")  # Replace with the correct hostname or IP if different
        self.model = os.getenv("DEFAULT_LLM_MODEL", "") 
        
        # Ollama(model="deepseek-r1:1.5b", base_url="http://host.docker.internal:39870", verbose=True)
        # self.sample_file_path = ''
        # self.columns = []
        # self.file_path = "/tmp/files/"
        # self.data_path = "/tmp/files/data/"
        # self.folder = '/tmp/files/sample/'

    def setAPIUrl(self, url):
        self.api_url = url
        return self

    def getAPIUrl(self):
       return self.api_url

    def getAnalyzerAgent(self, ):
        return AnalyzerAgent(self.model, self.api_url)

    def getCoderAgent(self):
        return CodeAgent(self.model, self.api_url)

    def getInsightAgent(self):
        return InsightAgent(self.model, self.api_url)

    def getReportAgent(self):
        return ReportAgent(self.model, self.api_url)

    def getReviewAgent(self):
        return ReviewAgent(self.model, self.api_url)

    def getTranslateAgent(self):
        return TranslateAgent(self.model, self.api_url)
    

