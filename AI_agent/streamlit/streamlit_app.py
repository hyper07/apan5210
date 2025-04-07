import time
import numpy as np
import pandas as pd
import streamlit as st

## doc: https://docs.streamlit.io/develop
## Add emoji: https://emojidb.org/



# with out processing
## url is based on the file name here
homepage = st.Page('homepage.py', title = 'homepage')

translation = st.Page("pages/translator.py", title="translation", icon="🔠")
summary = st.Page("pages/qwen.py", title="summary", icon="📋") # summary icon




pg = st.navigation({
    "homepage":[homepage], 
    "multi":[translation, summary]})
st.set_page_config(page_title="Multi bot", page_icon="🤖") # 𖠌
pg.run() 