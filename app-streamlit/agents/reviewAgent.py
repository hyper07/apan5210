import os
from langchain_community.llms import Ollama


class ReviewAgent:
    def __init__(self, var1=os.getenv("DEFAULT_REVIEWER_LLM_MODEL", ""), var2=os.getenv("DEFAULT_API_URL", "")):
        self.llmModel = var1
        self.llmUrl = var2
        self.MAX_TOKENS = 4000

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

    def setMaxTokens(self, max_tokens):
        self.MAX_TOKENS = max_tokens
        return self
    
    def get_llm(self):
        return Ollama(model=self.llmModel, base_url=self.llmUrl, verbose=True)

    def invoke(self, prompt):
        llm = self.get_llm()
        return llm.invoke(prompt)
    
    def stream(self, full_text):
        llm = self.get_llm()
        review_prompt = (
            "Please review the following report and provide feedback on its clarity, completeness, and overall quality:\n\n"
            + full_text[:self.MAX_TOKENS]
        )
        return llm.stream(review_prompt)