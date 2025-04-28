import os
import requests
from dotenv import load_dotenv

load_dotenv()

os.environ["DEEPSEEK_API_KEY"] = "sk-91eb1dbe41a34242a86cd3d4f4786fc9"

API_KEY  = os.getenv("DEEPSEEK_API_KEY")
BASE_URL = "https://api.deepseek.com"

class InsightAgent:

    def __init__(self, var1=os.getenv("DEFAULT_LLM_MODEL", "") , var2=os.getenv("DEFAULT_API_URL", "")):
        self.api_key  = os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("Please set DEEPSEEK_API_KEY in your .env file")
        self.endpoint = "https://api.deepseek.com/chat/completions"
        self.headers  = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json"
        }

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
    
    def critique(self, analysis: str) -> str:
        """
        Send the analysis text to DeepSeek and return a point-by-point critique:
        1. Main strengths
        2. Weaknesses or potential risks
        3. Suggestions for improvement
        4. Deeper insights or trends reflected
        """

        ## by shuteng
        prompt = (
            "Below is the result of a Python analysis:\n"
            f"{analysis}\n\n"
            "Please provide a professional critique including:\n"
            "1. What are the main strengths of these results?\n"
            "2. What weaknesses or potential risks exist?\n"
            "3. Are there areas for improvement? How would you improve?\n"
            "4. What deeper insights or trends do these results reflect?\n\n"
            "Respond in clear bullet points."
        )

        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "You are a data analysis expert skilled in critiquing results."},
                {"role": "user",   "content": prompt}
            ],
            "temperature": 0.0
        }

        response = requests.post(self.endpoint, json=payload, headers=self.headers)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()