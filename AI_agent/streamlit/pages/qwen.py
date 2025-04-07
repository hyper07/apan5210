import os
from openai import OpenAI
import streamlit as st

from dotenv import load_dotenv
load_dotenv()

try:
    client = OpenAI(
    
        api_key= os.getenv("Qwen_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",

        # Tips:
        ## 1. remember authorize it on ali cloud-model console (Chinese: "Bailian" or “百炼”)
        # <https://www.alibabacloud.com/help/en/model-studio/developer-reference/use-workspace?spm=a2c63.p38356.0.i3#f2e68d7ba7ubk>

        ## 2. the international verion and localized version are different
        ## use the link where you registered your account 

        # e.g.: China mainland vs international
        # https://dashscope.aliyuncs.com/compatible-mode/v1
        # "https://dashscope-intl.aliyuncs.com/compatible-mode/v1" (international)

    )

    completion = client.chat.completions.create(
        model="qwen-plus",  # different models：https://www.alibabacloud.com/help/zh/model-studio/getting-started/models
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': 'Who are you？'}
            ]
    )
    st.write(completion.choices[0].message.content)
except Exception as e:
    st.write(f"Error Information：{e}")
    st.write("Please refer to ：https://www.alibabacloud.com/help/zh/model-studio/developer-reference/error-code")