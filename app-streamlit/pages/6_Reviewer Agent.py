import streamlit as st 
import os
import base64
import torch 
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM 
from transformers import pipeline
from langchain_community.llms import Ollama
from langchain.document_loaders import PDFMinerLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.embeddings import SentenceTransformerEmbeddings 
from langchain.vectorstores import Chroma 
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA 
from streamlit_chat import message
from utils.constants import DATA_ANALYSYS_RESPONSES
from utils.constants import CHROMA_SETTINGS

from agents.reviewAgent import ReviewAgent  # <-- Add this import

persist_directory = "db"

st.set_page_config(page_title="ML Model Advisor", layout="wide")
st.title("Review Agent")
if "dataAnalysis" not in st.session_state or st.session_state.dataAnalysis is None:
    st.session_state.dataAnalysis = DATA_ANALYSYS_RESPONSES.copy()

review_agent = ReviewAgent(var1="llama3.2:1b", var2="http://host.docker.internal:39870")  # <-- Instantiate ReviewAgent
llm = review_agent.get_llm()  # <-- Use ReviewAgent to get the LLM

device = torch.device('cpu')

checkpoint = "MBZUAI/LaMini-T5-738M"
# print(f"Checkpoint path: {checkpoint}")  # Add this line for debugging
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    checkpoint,
    torch_dtype=torch.float32
)

@st.cache_resource
def llm_pipeline():
    pipe = pipeline(
        'text2text-generation',
        model = base_model,
        tokenizer = tokenizer,
        max_length = 256,
        do_sample = True,
        temperature = 0.3,
        top_p= 0.95,
        device=device
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm

@st.cache_resource
def qa_llm():
    llm = llm_pipeline()
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="db", embedding_function = embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever = retriever,
        return_source_documents=True
    )
    return qa

def process_answer(instruction):
    qa = qa_llm()
    generated_text = qa(instruction)
    answer = generated_text['result']
    return answer

def get_file_size(file):
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    return file_size

@st.cache_data
def displayPDF(file):
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def display_conversation(history):
    for i in range(len(history["generated"])):
        message(history["past"][i], is_user=True, key=str(i) + "_user")
        message(history["generated"][i],key=str(i))

def review_report_with_llm(pdf_path):
    # You only need to load the PDF and concatenate its text for the LLM prompt.
    loader = PDFMinerLoader(pdf_path)
    documents = loader.load()
    # If the PDF is not too large, you can skip splitting and just join all page contents.
    full_text = " ".join([doc.page_content for doc in documents])
    review_prompt = (
        "Please review the following report and provide feedback on its clarity, completeness, and overall quality:\n\n"
        + full_text[:4000]  # Truncate if needed for context length
    )
    review = llm(review_prompt)
    if isinstance(review, list):
        return review[0]['generated_text']
    elif isinstance(review, dict) and 'generated_text' in review:
        return review['generated_text']
    else:
        return str(review)

if "dataAnalysis" in st.session_state and "reporter" in st.session_state.dataAnalysis:
    report_path = st.session_state.dataAnalysis["reporter"].get("file_path")
    if report_path and os.path.exists(report_path):
        if st.button("Review Report with LLM"):
            with st.spinner("Reviewing report..."):
                review_result = review_report_with_llm(report_path)
            st.success("Review completed!")
            st.markdown("#### LLM Review Output:")
            st.write(review_result)
