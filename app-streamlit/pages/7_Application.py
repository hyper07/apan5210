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
from agents.analyzer import Analyzer
from agents.insightGenerator import InsightGenerator
from agents.reportGenerator import ReportGenerator
from agents.codeGenerator import CodeGenerator
from agents.reviewer import Reviewer
from agents.translater import Translater

folder = '/tmp/files/sample/'

@st.cache_data
def load_data():
    return pd.read_csv(folder+'bank.csv')

# FONCTIONS PAGE2 JEU DE DONNEES
@st.cache_data
def get_data_summary(df):
    return {
        "head": df.head(10),
        "shape": df.shape,
        "description": df.describe(),
        "dtypes": df.dtypes,
        "missing_values": df.isna().sum(),
        "duplicates": df[df.duplicated()],
    }
# Create directories if they don't exist
if not os.path.exists('files'):
    os.mkdir('files')

if not os.path.exists('db'):
    os.mkdir('db')


# Initialize template as a session state 
if 'template' not in st.session_state:

    # Set value of template key
    st.session_state.template = """
    
    You are a knowledgeable chatbot, here to help with questions of the user. 
    Your tone should be polite, professional and informative.

    Context: {context}
    History: {history}

    User: {question}
    Chatbot:

    """

# Initialize prompt as a session state
if 'prompt' not in st.session_state:

    # Set value of prompt key to PromptTemplate from langchain.prompts
    st.session_state.prompt = PromptTemplate(
        
        # Set input variables 
        input_variables=["history", "context", "question"],
        
        # Set template to the session state, template 
        template=st.session_state.template,
    )

# Initialize memory as a session state
if 'memory' not in st.session_state:

    # Set value of memory key to ConversationBufferMemory from langchain.memory
    st.session_state.memory = ConversationBufferMemory(

        # Set params from input variables list
        memory_key="history",
        return_messages=True,
        input_key="question")
    

# Initialize vectorstore
if 'vectorstore' not in st.session_state:

    # Set value of vectorstore key to Chroma 
    st.session_state.vectorstore = Chroma(persist_directory='db',
                                          embedding_function=OllamaEmbeddings(base_url='http://host.docker.internal:37869',
                                                                              model="deepseek-r1:1.5b")
                                          )
if 'llm' not in st.session_state:
    st.session_state.llm = Ollama(base_url="http://host.docker.internal:37869",
                                  model="deepseek-r1:1.5b",
                                  verbose=True,
                                  callback_manager=CallbackManager(
                                      [StreamingStdOutCallbackHandler()]),
                                  )

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

st.title("Chat with your PDFs")

# Upload a PDF file
uploaded_file = st.file_uploader("Upload your PDF", type='pdf')

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["message"])

if uploaded_file is not None:
    if not os.path.isfile("files/"+uploaded_file.name+".pdf"):
        with st.status("Analyzing your document..."):
            bytes_data = uploaded_file.read()
            f = open("files/"+uploaded_file.name+".pdf", "wb")
            f.write(bytes_data)
            f.close()
            loader = PyPDFLoader("files/"+uploaded_file.name+".pdf")
            data = loader.load()

            # Initialize text splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=200,
                length_function=len
            )
            all_splits = text_splitter.split_documents(data)

            # Create and persist the vector store
            st.session_state.vectorstore = Chroma.from_documents(
                documents=all_splits,
                embedding=OllamaEmbeddings(model="llama3")
            )
            st.session_state.vectorstore.persist()

    st.session_state.retriever = st.session_state.vectorstore.as_retriever()
    # Initialize the QA chain
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=st.session_state.llm,
            chain_type='stuff',
            retriever=st.session_state.retriever,
            verbose=True,
            chain_type_kwargs={
                "verbose": True,
                "prompt": st.session_state.prompt,
                "memory": st.session_state.memory,
            }
        )

    # Chat input
    if user_input := st.chat_input("You:", key="user_input"):
        user_message = {"role": "user", "message": user_input}
        st.session_state.chat_history.append(user_message)
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            with st.spinner("Assistant is typing..."):
                response = st.session_state.qa_chain(user_input)
            message_placeholder = st.empty()
            full_response = ""
            for chunk in response['result'].split():
                full_response += chunk + " "
                time.sleep(0.05)
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        chatbot_message = {"role": "assistant", "message": response['result']}
        st.session_state.chat_history.append(chatbot_message)


else:
    st.write("Please upload a PDF file.")


# from agents.analyzer import Analyzer
# from agents.reportGenerator import ReportGenerator
# from agents.reviewer import Reviewer
# from agents.translater import Translater


left, middle, right = st.columns(3)
if left.button("Analyzer button", use_container_width=True):
    customAgent = Analyzer()
    left.markdown(customAgent.test())
if left.button("ReportGenerator button", icon="😃", use_container_width=True):
    customAgent = ReportGenerator()
    left.markdown(customAgent.test())
if middle.button("Reviewer button", icon=":material/mood:", use_container_width=True):
    customAgent = Reviewer()
    middle.markdown(customAgent.test())
if middle.button("Translater button", use_container_width=True):
    customAgent = Translater()
    middle.markdown(customAgent.test())
if right.button("Emoji button", icon="😃", use_container_width=True):
    right.markdown("You clicked the emoji button.")
if right.button("Material button", icon=":material/mood:", use_container_width=True):
    right.markdown("You clicked the Material button.")    

st.divider()
st.header('Data Overview')
# Charger les données
df = load_data()
# Obtenir le résumé des données
data_summary = get_data_summary(df)
#TABS CONTAINERS:
tab1, tab2, tab3, tab4, tab5,tab6= st.tabs(["Overview", "Dimensions", "Statistics", "Types", "Nulls", "Duplicates"])
with tab1:
    st.code('df.head(10)')
    st.markdown("Overview of the first 10 lines")
    st.dataframe(data_summary["head"])
with tab2:
    st.markdown("The dimensions of the dataset: 17 variables and 11,162 rows.")
    st.code('df.shape')
    st.write(data_summary["shape"])
with tab3:
    st.code('df.describe()')
    st.write(data_summary["description"])
    st.subheader("Observation")
    st.markdown("""
    - age: 50% of the values ​​are between 32 and 49 years old. Many extreme values: max 95.
    - balance: 50% of the values ​​are between 122 and 1708. Presence of negative values ​​and extreme values: min -6,847, max 81,204.
    - duration: 50% of the values ​​are between 138 sec (2min) and 496 (8min). Presence of extreme values: max 3,881.
    - campaign: 50% of the values ​​are between 1 and 3 contacts. Presence of extreme values: max 63.
    - pdays: 50% of the values ​​are between - 1 and 20. The median is -1 which means that half of the customers have never been contacted before this campaign. Presence of extreme values: max 854.
    - previous: 50% of the values ​​are between 0 and 1. Presence of extreme values: max 58.
    """)
with tab4:
    st.markdown("Data types: ")
    st.code('df.dtypes')
    st.write(data_summary["dtypes"])
with tab5:
    st.markdown("No missing values: ")
    st.code('df.isna().sum()')
    st.write(data_summary["missing_values"])
with tab6:
    st.markdown("No duplicates: ")
    st.code('df[df.duplicated()]')
    st.write(data_summary["duplicates"])
    st.divider()