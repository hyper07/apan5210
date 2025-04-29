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

    def setAPIUrl(self, url):
        self.api_url = url
        return self

    def getAPIUrl(self):
       return self.api_url

    def getAnalyzerAgent():
        return AnalyzerAgent()

    def getCoderAgent():
        return CodeAgent()

    def getInsightAgent():
        return InsightAgent()

    def getReportAgent():
        return ReportAgent()

    def getReviewAgent():
        return ReviewAgent()

    def getTranslateAgent():
        return TranslateAgent()
    

