from pathlib import Path
import os

from langchain_community.llms import Ollama
import pandas as pd


## presetting by Jing
class CodeAgent:
    def __init__(self, var1=os.getenv("DEFAULT_CODER_LLM_MODEL", "") , var2=os.getenv("DEFAULT_API_URL", "")):
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
    
    
    def generate_code(self, selected_model, file_path, prediction_variable, user_prompt):
        
        agent = self.get_llm()

        df = pd.read_csv(file_path, encoding='utf-8', header=0)
        available_variables = list(df.columns)
        feature_variables = [col for col in available_variables if col != prediction_variable]

        llm_prompt = f"""Generate Python code to train a '{selected_model}' model using the pandas DataFrame 'df'.
            The file path is '{file_path}' for df.
            The target variable is '{prediction_variable}'.
            The available feature variables are: {feature_variables}.
            Include steps for:
            1. Importing necessary libraries (like pandas, scikit-learn).
            2. Convert data to correct format for ML model.
            3. Defining features (X) and target (y).
            4. Splitting the data into training and testing sets.
            5. Initializing and training the '{selected_model}' model.
            6. Making predictions on the test set (if applicable).
            7. Evaluating the model (e.g., accuracy, MSE, R2 score, depending on the problem type).

            Consider the data types: {dict(df.dtypes.apply(lambda x: str(x)))}
            Here's a sample of the data: {df.head(5).to_dict()}

            User request: {user_prompt}

            Output only the Python code block, enclosed in triple backticks like this:
            ```python
            # Your Python code here
            ```
            """
        # Stream the response from the LLM
        response = agent.stream(llm_prompt)
        return response

