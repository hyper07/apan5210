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
import subprocess
import os
import time
from agents.analyzerAgent import AnalyzerAgent
from agents.codeAgent import CodeAgent
from agents.insightAgent import InsightAgent
from agents.reportAgent import ReportAgent
from agents.reviewAgent import ReviewAgent
from agents.translateAgent import TranslateAgent
import requests
from fpdf import FPDF

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
            response = requests.post(
                f"{self.api_url}/api/pull", 
                json={"model": model, "stream": False}
            )
            if response.status_code == 200:
                return response.json().get("status", "Unknown status")
            else:
                error_message = response.json().get("message", "Unknown error")
                st.error(f"Error pulling model '{model}': {error_message}")
                return None
        except requests.exceptions.RequestException as e:
            st.error(f"Request exception while pulling model '{model}': {e}")
            return None

    def getLLMlist(self):
        try:
            response = requests.get(f"{self.api_url}/api/tags")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            st.error(f"HTTP error while fetching model list: {e}")
            return None
        except requests.exceptions.RequestException as e:
            st.error(f"Request exception while fetching model list: {e}")
            return None

    def getModelListOnly(self):
        models = []
        model_list = self.getLLMlist()
        if model_list is not None:
            for model in model_list.get("models", []):
                models.append(model.get("name"))
            return models
        else:
            st.warning("Model list is empty or could not be fetched.")
            return []

    def initialize_llms_session_state():
        if 'llms' not in st.session_state:
            nav_script = """
                <meta http-equiv="refresh" content="0; url='/'">
            """
            st.write(nav_script, unsafe_allow_html=True)

    @staticmethod
    def save_pdf(
        text,
        lang,
        selected_model=None,
        target_variable=None,
        feature_variables=None,
        code=None,
        code_result=None,
        insight=None
    ):
        """Save the given text and analysis context as a PDF in /tmp/files/pdf/report_{lang}.pdf."""
        pdf_dir = '/tmp/files/pdf'
        os.makedirs(pdf_dir, exist_ok=True)
        pdf_filename = f"report_{lang.lower()}.pdf"
        pdf_path = os.path.join(pdf_dir, pdf_filename)
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)

        # Use a Unicode font (DejaVu) for full Unicode support
        font_dir = "/tmp/files/fonts"
        os.makedirs(font_dir, exist_ok=True)
        font_path = os.path.join(font_dir, "DejaVuSans.ttf")
        font_bold_path = os.path.join(font_dir, "DejaVuSans-Bold.ttf")
        font_other_language_path = os.path.join(font_dir, "NotoSansCJK-Regular.ttc")
        # Download fonts if not present
        if not os.path.exists(font_path):
            import urllib.request
            url = "https://github.com/dejavu-fonts/dejavu-fonts/raw/master/ttf/DejaVuSans.ttf"
            urllib.request.urlretrieve(url, font_path)
        if not os.path.exists(font_bold_path):
            import urllib.request
            url_bold = "https://github.com/dejavu-fonts/dejavu-fonts/raw/master/ttf/DejaVuSans-Bold.ttf"
            urllib.request.urlretrieve(url_bold, font_bold_path)
        if not os.path.exists(font_other_language_path):
            import urllib.request
            url_bold = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/TraditionalChinese/NotoSansCJKtc-Regular.otf"
            urllib.request.urlretrieve(url_bold, font_bold_path)
        pdf.add_font("DejaVu", "", font_path, uni=True)
        pdf.add_font("DejaVu", "B", font_bold_path, uni=True)
        pdf.add_font("NotoCJK", "", font_other_language_path, uni=True)

        pdf.set_font("DejaVu", size=12)
        page_width = pdf.w - 2 * pdf.l_margin  # Usable page width

        # Add analysis context if provided
        if selected_model is not None:
            pdf.set_font("DejaVu", "B", 12)
            pdf.cell(0, 10, "Selected Model:", ln=True)
            pdf.set_font("DejaVu", "", 12)
            pdf.multi_cell(page_width, 10, str(selected_model))
            pdf.ln(2)
        if target_variable is not None:
            pdf.set_font("DejaVu", "B", 12)
            pdf.cell(0, 10, "Target Variable:", ln=True)
            pdf.set_font("DejaVu", "", 12)
            pdf.multi_cell(page_width, 10, str(target_variable))
            pdf.ln(2)
        if feature_variables is not None:
            pdf.set_font("DejaVu", "B", 12)
            pdf.cell(0, 10, "Feature Variables:", ln=True)
            pdf.set_font("DejaVu", "", 12)
            pdf.multi_cell(page_width, 10, ", ".join(map(str, feature_variables)))
            pdf.ln(2)
        if code is not None:
            pdf.set_font("DejaVu", "B", 12)
            pdf.cell(0, 10, "Code Used:", ln=True)
            pdf.set_font("DejaVu", "", 12)
            pdf.multi_cell(page_width, 10, str(code))
            pdf.ln(2)
        if code_result is not None:
            pdf.set_font("DejaVu", "B", 12)
            pdf.cell(0, 10, "Code Result:", ln=True)
            pdf.set_font("DejaVu", "", 12)
            pdf.multi_cell(page_width, 10, str(code_result))
            pdf.ln(2)
        if insight is not None:
            pdf.set_font("DejaVu", "B", 12)
            pdf.cell(0, 10, "Insight:", ln=True)
            pdf.set_font("DejaVu", "", 12)
            pdf.multi_cell(page_width, 10, str(insight))
            pdf.ln(2)

        if text is not None:
            pdf.set_font("DejaVu", "B", 12)
            pdf.cell(0, 10, "Report:", ln=True)
            pdf.set_font("NotoCJK", "", 12)
            pdf.multi_cell(page_width, 10, str(text))
            pdf.ln(2)
 
        pdf.output(pdf_path)
        return pdf_path

    def run_python_script(code: str):
        """Save code to /tmp/files/script/run.py and execute it, returning stdout and stderr."""
        script_dir = "/tmp/files/script"
        script_path = os.path.join(script_dir, "run.py")
        os.makedirs(script_dir, exist_ok=True)
        with open(script_path, "w") as f:
            f.write(code)
        try:
            result = subprocess.run(
                ["python", script_path],
                capture_output=True,
                text=True,
                timeout=60,
                env={**os.environ, "PYTHONUNBUFFERED": "1"}  # Ensure real-time output
            )
            return result.stdout, result.stderr
        except Exception as e:
            return "", f"Execution failed: {e}"

    @staticmethod
    def get_analyzer_agent(df):
        agent = AnalyzerAgent()
        return agent.create_agent(df)

    @staticmethod
    def get_analyzer_response(agent, df, prediction_variable, prompt):
        return AnalyzerAgent().analyze_data(agent, df, prediction_variable, prompt)
