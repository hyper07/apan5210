import os
import requests
from dotenv import load_dotenv

load_dotenv()

########
## presetting by Jing
class CodeAgent:
    def __init__(self, var1=os.getenv("DEFAULT_LLM_MODEL", "") , var2=os.getenv("DEFAULT_API_URL", "")):
        self.api_key  = os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("Please set DEEPSEEK_API_KEY in your .env file")
        self.endpoint = "https://api.deepseek.com/chat/completions"
        self.headers  = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json"
        }

    def generate(self, analysis: str) -> str:
        presetting ="You are an AI coding assistant. \
            You will receive recommended machine learning model(s) for a given dataset and analysis task. \
            Your job is to generate Python code for each recommended model. \
            If multiple models are provided, generate separate, clearly labeled Python code blocks for each one. \
            Ensure the code includes all essential steps for model training and evaluation, such as data splitting, fitting, and prediction. \
            Do not explain or justify the model choices—focus only on clean, executable Python code for each model. \
            Respond in English."

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
        # st.session_state.messages.append({"role":"system", "content":presetting})