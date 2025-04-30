
from pathlib import Path
import os

from langchain_community.llms import Ollama

class ReportAgent:

    def __init__(self, var1=os.getenv("DEFAULT_REPORT_LLM_MODEL", "") , var2=os.getenv("DEFAULT_API_URL", "")):
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
    
    def get_llm(self):

        return Ollama(model=self.llmModel, base_url=self.llmUrl, verbose=True)
    
    def stream(self, prompt):

        llm = self.get_llm()
        return llm.stream(prompt)
    
    def invoke(self, prompt):

        llm = self.get_llm()
        return llm.invoke(prompt)

