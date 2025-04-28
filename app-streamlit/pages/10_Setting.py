import streamlit as st
import pandas as pd
import numpy as np
import os
from controllers.serviceController import ServiceController
from controllers.agentController import AgentController
# from urllib.parse import urlparse
from utils.constants import DATA_ANALYSYS_RESPONSES, REQUIRED_MODELS

st.set_page_config(layout="wide")

st.write("# Settings")

# Function to refresh model list
def refresh_models():
    st.session_state.llms = ServiceController().getModelListOnly()

llms = ServiceController().getModelListOnly()
st.session_state.llms = llms

with st.status("Initializing required models ...", expanded=True) as status:
    common_elements = list(set(llms if llms is not None else []) & set(REQUIRED_MODELS))

    if llms is None:
        st.error("Can't get models. Please check the API URL.")
        status.update(label="Initialization failed: Cannot connect to API.", expanded=True, state="error")
    elif len(common_elements) < len(REQUIRED_MODELS):
        missing_models = [model for model in REQUIRED_MODELS if model not in common_elements]
        remains = len(missing_models)
        count = 1
        for model in missing_models:
            status.update(
                label=f"({count}/{remains}) Downloading required model: {model} ...", expanded=True, state="running"
            )
            result = ServiceController().pullModelFromSite(model)
            # Optionally check result and handle errors
            count += 1

        refresh_models() # Refresh list after downloads
        llms = st.session_state.llms # Update local variable
        status.update(label="Required models downloaded. Initialization complete.", expanded=True, state="complete")
    else:
        status.update(label="All required models are available. Initialization complete.", expanded=True, state="complete")


st.session_state.currentPage = "Home"
st.session_state.dataAnalysis = DATA_ANALYSYS_RESPONSES

st.divider()

# Section to download a specific model
st.write("### Download a Model")
model_to_download = st.text_input("Enter the model name to download (e.g., 'llama3:latest'):")
if st.button("Download Model"):
    if model_to_download:
        with st.spinner(f"Downloading {model_to_download}..."):
            try:
                result = ServiceController().pullModelFromSite(model_to_download)
                # Assuming result indicates success/failure, you might want to check it
                st.success(f"Model '{model_to_download}' downloaded successfully (or already exists).")
                refresh_models() # Refresh the list after download
                llms = st.session_state.llms # Update local variable
                st.rerun() # Rerun the script to update the selectbox
            except Exception as e:
                st.error(f"Failed to download model '{model_to_download}': {e}")
    else:
        st.warning("Please enter a model name.")


st.divider()

if llms:
    st.write("### Available Models:")
    # Use the refreshed list from session state
    selected_model = st.selectbox("", st.session_state.llms, key="model_selector")

else:
    st.warning("No models available. Please check the API connection or download a model.")
