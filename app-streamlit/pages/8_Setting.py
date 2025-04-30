import streamlit as st
from controllers.serviceController import ServiceController
# from urllib.parse import urlparse
from utils.constants import REQUIRED_MODELS, SAMPLE_ANALYSYS_RESPONSES

st.set_page_config(layout="wide")

st.write("# Settings")
st.session_state.currentPage = "Home"
# Function to refresh model list
def refresh_models():
    st.session_state.llms = ServiceController().getModelListOnly()

def setAPIKey(api_type, api_key):
    # if api_type is not None and api_type is not None:
        st.session_state.dataAnalysis[api_type] = api_key

st.session_state.dataAnalysis = SAMPLE_ANALYSYS_RESPONSES if st.session_state.dataAnalysis is None else st.session_state.dataAnalysis

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


st.divider()
openai_api_key = st.text_input("OPENAI API KEY")
if st.button("Save OPENAI API Key"):
    setAPIKey("openai_api_key", openai_api_key)
    st.success("OPENAI API Key saved to session.")

deepseek_api_key = st.text_input("DEEPSEEK API KEY")
if st.button("Save DEEPSEEK API Key"):
    setAPIKey("deepseek_api_key", deepseek_api_key)
    st.success("DEEPSEEK API Key saved to session.")
