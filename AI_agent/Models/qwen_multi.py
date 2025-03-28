import os
from openai import OpenAI

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
    
    # to realize the multiple round chat, set quota = 10, set End_sign as "q"
    quota = 10
    pre_setting = 'You are a helpful assistant.'
    Input = [
        {'role': 'system', 'content': pre_setting}, 
                ]  
    End_sign = "q"
    
    # loop
    while quota>0:
        
        Input_message = input(f"Please write down your question (Enter {End_sign} to quit, remain quota:{quota}):  ")
        quota -= 1
        
        # End the chat
        if Input_message == End_sign:
            print("Thank you.\nSee you next time~")
            break

        Input.append({"role":"user", "content": Input_message})

        completion = client.chat.completions.create(
            model="qwen-plus",  # different models：https://www.alibabacloud.com/help/zh/model-studio/getting-started/models
            messages= Input
            
        )
        Input.append({"role":"assistant", "content": completion.choices[0].message.content})
        
        print(completion.choices[0].message.content + f"\n remain quota:{quota}")

        # print(f"Input is: {Input}\n")
        


except Exception as e:
    print(f"Error Information：{e}")
    print("Please refer to ：https://www.alibabacloud.com/help/zh/model-studio/developer-reference/error-code")