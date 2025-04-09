import streamlit as st
from langchain.llms import Ollama
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks import StreamlitCallbackHandler
from langchain.callbacks.streaming_stdout_final_only import FinalStreamingStdOutCallbackHandler

search_internet = st.checkbox("check internet?", value=False, key="internet")
prompt = st.text_input("prompt", value="", key="prompt")

if prompt!="":
    response = ""
    if not search_internet:
        llm = Ollama(model="deepseek-r1:1.5b"
                     ,base_url="http://host.docker.internal:37869"
                     ,verbose=True
                     ) # 👈 stef default
        response = llm.predict(prompt)
    else:
        llm = Ollama(
            model="deepseek-r1:1.5b"
            ,base_url="http://host.docker.internal:37869"
            ,verbose=True
            ,callback_manager=CallbackManager([FinalStreamingStdOutCallbackHandler()])
        )
        agent = initialize_agent(
            load_tools(["ddg-search"])
            ,llm 
            ,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION
            ,verbose=True
            ,handle_parsing_errors=True
        )
        response = agent.run(prompt, callbacks=[StreamlitCallbackHandler(st.container())])

        
    st.markdown(response)