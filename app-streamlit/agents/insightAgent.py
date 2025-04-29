import os
from langchain_community.llms import Ollama


class InsightAgent:

    def __init__(self, var1=os.getenv("DEFAULT_INSIGHT_LLM_MODEL", "") , var2=os.getenv("DEFAULT_API_URL", "")):
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

        return  Ollama(model=self.llmModel, base_url=self.llmUrl, verbose=True)
        

    def critique(
        self,
        selected_model: str,
        prediction_variable: str,
        feature_variables: list,
        extracted_code: str,
        script_output: str,
        insight_prompt: str
    ):
        llm = self.get_llm()

        insight_llm_prompt = f"""
        You are an industry expert. Given the following:
        - Model: {selected_model}
        - Target variable: {prediction_variable}
        - Feature variables: {feature_variables}
        - Python code used for modeling:
        ```python
        {extracted_code}
        ```
        - Output of the code:
        ```python
        {script_output}
        ```
        Provide an industry-specific insight or best practice in response to this user request:
        '{insight_prompt}'
        """

        return llm.stream(insight_llm_prompt)
