import os 
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(page_title="Plotting Demo", page_icon="📈")
st.markdown("# Mapping Demo")

def main():
    st.write(f'# {Path(__file__).parent.name} - {Path(__file__).name}')