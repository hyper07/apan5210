
from pathlib import Path
import os
from langchain_community.llms import Ollama

class TranslateAgent:
    def __init__(self, var1=os.getenv("DEFAULT_TRANSLATER_LLM_MODEL", "") , var2=os.getenv("DEFAULT_API_URL", "")):
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
    
    def getAPIKey(self):  

        return self.apikey
    
    def setAPIKey(self, apikey):
        self.apikey = apikey

        return self

    def get_llm(self):

        return Ollama(model=self.llmModel, base_url=self.llmUrl, verbose=True)

    def invoke(self, language, prompt):
        llm = self.get_llm()

        translate_prompt = (
            f"Translate the following text to {language}:\n\n"
            f"{prompt}"
        )
        
        return llm.invoke(translate_prompt)

    def stream(self, language, prompt):
        llm = self.get_llm()

        translate_prompt = (
            f"Translate the following text to {language}:\n\n"
            f"{prompt}"
        )

        return llm.stream(translate_prompt)
