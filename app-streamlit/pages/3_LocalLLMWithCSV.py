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



file_path = os.getcwd()
llm = Ollama(model="deepseek-r1:1.5b", base_url="http://host.docker.internal:37869", verbose=True)

sample_file_path = ''
columns = []

    
CSS = """
.stChatMessage:has([data-testid="stChatMessageAvatarUser"]) {
    display: flex;
    flex-direction: row-reverse;
    align-itmes: end;
}

[data-testid="stChatMessageAvatarUser"] + [data-testid="stChatMessageContent"] {
    text-align: right;
}
"""
st.html(f"<style>{CSS}</style>")

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

uploaded_file = st.file_uploader(
            "",
            type='csv',
            key="1",
            help="To activate 'wide mode', go to the hamburger menu > Settings > turn on 'wide mode'",
        )   

if uploaded_file is not None:
    file_container = st.expander("Check your uploaded .csv")
    shows = pd.read_csv(uploaded_file, index_col=0)
    sample = shows.head()
    uploaded_file.seek(0)
    columns = list(sample.columns.values)
    keywords = st_tags(
        label='# Available variables:',
        text='',
        value=columns,
        suggestions=columns,
        maxtags = len(columns),
        key='tags')

    file_container.write(shows)

    sample_file_path = '/tmp/files/sample_'+uploaded_file.name
    sample.to_csv(sample_file_path)
    if "messages" not in st.session_state.keys(): 
        st.session_state.messages = []
    # if "messages" in st.session_state.keys(): 
    #     st.session_state.messages = [
    #         {"role": "assistant", "content": "Please type short prompts (example: relathiship between {column name 1} and {column name 2})"}
    #     ]
else:
    st.info(
        f"""
            👆 Upload a .csv file first.
            """
    )

    st.stop()
    
if prompt := st.chat_input("Your prompt"): 
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: 
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
if len(st.session_state.messages) > 0 and (st.session_state.messages[0]["role"] != "assistant" or st.session_state.messages[-1]["role"]) != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking ..."):
            loader = CSVLoader(file_path=sample_file_path,
                csv_args={
                    'delimiter': ',',
                    'quotechar': '"',
                    'fieldnames': columns
                })
            data = loader.load()
            embeddings = HuggingFaceEmbeddings()
            index_creator = VectorstoreIndexCreator(embedding=embeddings)
            docsearch = index_creator.from_loaders([loader])
            # st.write(data)
            # st.write(docsearch)
            # st.write(docsearch.vectorstore.as_retriever())
            chain=RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=docsearch.vectorstore.as_retriever(),
                verbose=True,
                input_key="question")

            query = prompt
            message_placeholder = st.empty()
            response = ""
            if query:
                response_generator = chain.stream(
                    {"question": query},
                    callbacks=[StreamingStdOutCallbackHandler()]
                )
                response = ""
                for chunk in response_generator:
                    message_placeholder.write(chunk)
                    # Remove <think> tags from the chunk
                    if chunk['result'] is not None:
                        chunk['result'] = chunk['result'].replace("<think>", "").replace("</think>", "")
                        response += chunk['result']
                        message_placeholder.write(response)  # Update the assistant's message progressively

                st.session_state.messages.append({"role": "assistant", "content": response})