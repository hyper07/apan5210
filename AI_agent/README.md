# Work with AI Agent
 Demo for works

### To set .env

Please create your own dot-env file named ".env" (without a file extension) in the same directory

Apply and set your own API Key in the .env file

    OPENAI_API_KEY = "sk-xxx"  # (paste your own API key to replace the sk-xxx)

You can apply the API key through:
openai: <https://platform.openai.com/api-keys>
deepseek: <https://platform.deepseek.com/api_keys>
etc..

Finally, to load your .env file
```python
# pip install os
# pip install dotenv
import os

from dotenv import load_dotenv
load_dotenv()

api_key=os.environ.get("OPENAI_API_KEY")
```
And 

Enjoy your AI journey. :)
