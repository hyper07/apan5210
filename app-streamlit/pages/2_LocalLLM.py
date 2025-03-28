
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

import ollama


file_path = os.getcwd()
llm = Ollama(model="llama3.2:1b", base_url="http://host.docker.internal:37869", verbose=True)

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