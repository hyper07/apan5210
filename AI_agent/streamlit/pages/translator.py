# Translation by deepseek

### settings (do not change them if possible)
from openai import OpenAI
from dotenv import load_dotenv
import os
import streamlit as st
import time

load_dotenv()
client = OpenAI(
    api_key=os.environ.get("Deepseek_API_KEY"), 
    base_url="https://api.deepseek.com")

def stream_data(texts):
    for text in texts.split(" "):
        yield text+" "
        time.sleep(0.05)


### setting of webpage and AI bot
st.title("Translator")
st.subheader("I am a translator, feel free to input your text. I will auto save it and you can use it in other sections")


presetting = "Generally, you are a translator. By default, translate all things to English. \
    User can set the language, but should express explicitly. Otherwise, use English as the general language \
    If user ask something in english without transaltion demand, you can also answer it"


# initialize the dialoge content
if "translator_messages" not in st.session_state:
    st.session_state.translator_messages = []
    st.session_state.translator_messages.append({"role": "system", "content": presetting})


# Display chat messages from history on app rerun
for message in st.session_state.translator_messages:
    if message['role'] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])



if prompt := st.chat_input("Say something"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.translator_messages.append({"role": "user", "content": prompt})


    # Use ai here
    response = client.chat.completions.create(

        model="deepseek-chat",
        ## Can add a select box (later)
        # model = "deepseek-reasoner"
        # model = "deepseek-chat"

        messages = st.session_state.translator_messages,
        stream=False
    )
    ai_response = response.choices[0].message.content
    st.session_state.translator_messages.append({"role": "assistant", "content": ai_response})

    with st.chat_message('assistant'):
        # st.write(ai_response)
        
        ## Make it like output by llm
        st.write_stream(stream_data(ai_response))


if st.button('summary', icon = "✍"):
    mg = st.session_state.translator_messages
    mg.append({"role": "user", "content": "create a summary of the dialog for user"})
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages = mg,
        stream=False
        )
    with st.chat_message('assistant'):
        st.write(response.choices[0].message.content)
