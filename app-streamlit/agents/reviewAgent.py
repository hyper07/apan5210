from pathlib import Path
import os

from langchain_community.llms import Ollama

class ReviewAgent:
    def __init__(self, var1=os.getenv("DEFAULT_REVIEWER_LLM_MODEL", "") , var2=os.getenv("DEFAULT_API_URL", "")):
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

