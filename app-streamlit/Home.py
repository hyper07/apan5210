import streamlit as st
import pandas as pd
import numpy as np
import os

st.set_page_config(layout="wide")

st.write("Hello World")
st.write("## This is a H2 Title!1")
x = st.text_input("Movie", "Star Wars")

if st.button("Click Me"):
    st.write(f"Your favorite movie is `{x}`")


file_path = "/tmp/files/sample/movies.csv"  # Updated file path
if os.path.exists(file_path):
    try:
        data = pd.read_csv(file_path)
        st.write(data)
    except Exception as e:
        st.error(f"Error reading the file: {e}")



chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])

st.bar_chart(chart_data)
