
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

class TranslateAgent:
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
        file_path = os.getcwd()
        llm = Ollama(model="llama3.2:1b", base_url="http://host.docker.internal:39870", verbose=True)

        sample_file_path = ''
        columns = []

        def sendPrompt(prompt):
            global llm
            response = llm.invoke(prompt)
            return response


        st.title("Download LLM Models")
        st.write("Check the LLM models from the following links:")
        st.write("[OPEN SOURCE LLMs](https://ollama.com/library)")

        st.image(file_path+"/images/select_llm.png", width=700)
        st.write("Run 'docker exec -ti apan-ollama ollama pull llama3.2:1b' on the command to download the model")
        st.image(file_path+"/images/download_llm.png", width=700)


        st.title("Sample Chat UI")

        if "messages" not in st.session_state.keys(): 
                st.session_state.messages = [
                    # {"role": "assistant", "content": "Please type short prompts (example: relathiship between {column name 1} and {column name 2})"}
                ]
        
        ## by Jerry
        presetting = "You are an AI language translator. \
            Your primary task is to translate any input text into the target language. \
            By default, translate all content into English unless the user specifies a different target language. \
            \
            The input may consist of one or multiple paragraphs, possibly drawn from various sources. \
            Read the entire text carefully, understand the context, and ensure your translation \
            preserves the original meaning, tone, and style wherever appropriate."
                            
        st.session_state.messages.append({"role":"system", "content":presetting})


        if prompt := st.chat_input("Your prompt"): 
            st.session_state.messages.append({"role": "user", "content": prompt})

        for message in st.session_state.messages: 
            with st.chat_message(message["role"]):
                st.write(message["content"])
                
        if len(st.session_state.messages) > 0 and (st.session_state.messages[0]["role"] != "assistant" or st.session_state.messages[-1]["role"]) != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Checking ..."):
                    # loader = CSVLoader(file_path=sample_file_path,
                    #     csv_args={
                    #         'delimiter': ',',
                    #         'quotechar': '"',
                    #         'fieldnames': columns
                    #     })
                    # data = loader.load()
                    # st.write(data)
                    embeddings = HuggingFaceEmbeddings()
                    index_creator = VectorstoreIndexCreator(embedding=embeddings)
                    docsearch = index_creator.from_loaders([])
                    # st.write(docsearch)
                    # st.write(docsearch.vectorstore.as_retriever())
                    chain=RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=docsearch.vectorstore.as_retriever(),
                        input_key="question")

                    query=prompt
                    # query="what type of machine learning model do you recommend to analyze relationship between joined date and inducted date from attached data? Can you give me the machine learning model list only?"
                    if query:
                        response=chain({"question":query})
                        message = {"role": "assistant", "content": response['result']}
                        st.write(response['result'])
                        st.session_state.messages.append(message) 