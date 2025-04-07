import time
import numpy as np
import pandas as pd
import streamlit as st


Welcome  = "Welcome to Multi bot👋"
st.subheader(Welcome)

Words = "Nice to meet you here, Multi bot is a powerful text processing LLM. Feel free to use it. \
 You can use it with one function. OR work with a workflow"

def stream_data_hp():
    """For homepage, welcome words"""

    for word in Words.split(" "):
        yield word + " "
        time.sleep(0.02)

st.write_stream(stream_data_hp)

st.text("Have a nice try")