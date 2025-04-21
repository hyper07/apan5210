
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


class ReviewAgent:
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
    
    def sendPrompt(self, prompt):
        
    # Sze Ning 's edit
    # MODIFY HERE: Treat 'prompt' as the full narrative text from Agents 1–5
        file_path = os.getcwd()
        llm = Ollama(model="llama3.2:1b", base_url="http://host.docker.internal:39870", verbose=True)

    # MODIFY HERE: Construct a review prompt that references outputs from Agents 1–5
        review_prompt = (
            "You are Agent 6, the Reviewer. You receive the complete pipeline outputs from Agents 1–5: "
            "the data summary, model selection rationale, execution results, insights, and translated text. "
            "Polish and refine the following narrative for clarity, logical flow, and professional tone. "
            "Ensure consistent terminology, add smooth transitions between sections, and label sections appropriately.\n\n"
            + prompt
        )
        response = llm.invoke(review_prompt)
        polished_narrative = response.strip()

  # MODIFY HERE: Display only the refined narrative in the Streamlit UI
        st.title("Agent 6: Narrative Reviewer")
        st.subheader("Refined Narrative Output")
        st.write(polished_narrative)

        # MODIFY HERE: End of Agent 6 scope – downstream PDF builder will use this polished text

# End of Reviewer class

        
        