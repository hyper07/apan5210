import os 
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

def main():
    st.write(f'# {Path(__file__).parent.name} - {Path(__file__).name}')