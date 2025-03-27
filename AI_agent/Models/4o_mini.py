from openai import OpenAI
import os

# load the env file
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(
   # This is the default and can be omitted

    # By the way, remember to creat a dotenv file, please
    api_key=os.environ.get("OPENAI_API_KEY"),
)

prompt = input("Please write down your mission in text")

completion = client.chat.completions.create(
  model="gpt-4o-mini",
  store=True,
  messages=[
    {"role": "user", "content": prompt}
  ]
)

print(completion.choices[0].message.content);


# we can add the loop to end the input/ quota (i++)
