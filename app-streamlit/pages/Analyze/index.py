import os 
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(page_title="Mapping Demo", page_icon="🌍")


def main():
    # st.write(f'# {Path(__file__).parent.name} - {Path(__file__).name}')

    st.title("Hi from streamlit inside docker")
    st.write("looks like this works properly.")
    st.write("What happens if you change the file?")


    st.write("Hello World")
    st.write("## This is a H2 Title!")
    x = st.text_input("Movie", "Star Wars")

    if st.button("Click Me"):
        st.write(f"Your favorite movie is `{x}`")

    cwd = os.getcwd()
    st.write(cwd)
    files = os.listdir(cwd)  # Get all the files in that directory
    print("Files in %r: %s" % (cwd, files))
    st.write("Files in %r: %s" % (cwd, files))
    data = pd.read_csv(cwd+"/movies.csv")
    st.write(data)



    chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])

    st.bar_chart(chart_data)