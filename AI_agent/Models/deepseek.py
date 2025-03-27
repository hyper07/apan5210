from openai import OpenAI
import os

from dotenv import load_dotenv
load_dotenv()

client = OpenAI(
    api_key=os.environ.get("Deepseek_API_KEY"), 
    base_url="https://api.deepseek.com")



response = client.chat.completions.create(
    # model = "deepseek-reasoner"
    # model = "deepseek-chat"
    model="deepseek-chat",


    messages=[
        # The system message initializes the AI's behavior.
        # Can also set the context

        ## here I set the output in Chinese
        {"role": "system", "content": "Response in Chinses"},


        {"role": "user", "content": "Hello, how are you"},
    ],

    # If stream = True, like the output style u see in deepseek website
    stream=False
)

print(response.choices[0].message.content)