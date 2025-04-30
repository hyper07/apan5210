import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from controllers.serviceController import ServiceController
from controllers.agentController import AgentController
# from urllib.parse import urlparse
from utils.constants import DATA_ANALYSYS_RESPONSES, REQUIRED_MODELS, SAMPLE_ANALYSYS_RESPONSES, BACKUP_DIR

st.set_page_config(layout="wide")
st.write("# HOME")

st.session_state.currentPage = "Home"
st.session_state.dataAnalysis = DATA_ANALYSYS_RESPONSES if 'dataAnalysis' not in st.session_state else st.session_state.dataAnalysis


llms = ServiceController().getModelListOnly()
st.session_state.llms = llms


with st.status("Initializing models ...", expanded=True) as status:
    common_elements = list(set(llms if llms is not None else []) & set(REQUIRED_MODELS))    

    if llms is None:
        st.error("Can't get models. Please check the API URL.")
    elif len(common_elements) < 7:
        remains = 7 - len(common_elements)
        count = 1
        for model in REQUIRED_MODELS:
            if model not in llms:
                status.update(
                    label=f"({count}/{remains}) Downloading {model} model ...", expanded=True, state="running"
                )
                result = ServiceController().pullModelFromSite(model)
                count += 1

        llms = ServiceController().getModelListOnly()
        st.session_state.llms = llms
        status.update(label="All models downloaded. Initialization complete.", expanded=True, state="complete")
    else:
        status.update(label="All models downloaded. Initialization complete.", expanded=True, state="complete")


# --- Button Section ---
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Load Sample Data"):
        st.session_state.dataAnalysis = SAMPLE_ANALYSYS_RESPONSES
        st.success("Sample data loaded.")


with col2:
    if st.button("Backup Current Data"):
        if "dataAnalysis" in st.session_state:
            try:
                def make_json_serializable(obj):
                    if isinstance(obj, pd.DataFrame):
                        return obj.to_dict(orient="records")
                    elif isinstance(obj, dict):
                        return {k: make_json_serializable(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [make_json_serializable(i) for i in obj]
                    else:
                        return obj

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_filename = f"dataAnalysis_backup_{timestamp}.json"
                backup_filepath = os.path.join(BACKUP_DIR, backup_filename)
                serializable_data = make_json_serializable(st.session_state.dataAnalysis)
                with open(backup_filepath, 'w') as f:
                    json.dump(serializable_data, f, indent=4)
                st.success(f"Data backed up to {backup_filepath}")
            except Exception as e:
                st.error(f"Error backing up data: {e}")
        else:
            st.warning("No data analysis state found to back up.")

with col3:
    if st.button("Load from Backup"):
        try:
            backup_files = [f for f in os.listdir(BACKUP_DIR) if f.startswith("dataAnalysis_backup_") and f.endswith(".json")]
            if not backup_files:
                st.warning("No backup files found.")
            else:
                # Find the latest backup file
                latest_backup_file = max(backup_files, key=lambda f: os.path.getmtime(os.path.join(BACKUP_DIR, f)))
                latest_backup_filepath = os.path.join(BACKUP_DIR, latest_backup_file)
                
                with open(latest_backup_filepath, 'r') as f:
                    backup_data = json.load(f)
                st.session_state.dataAnalysis = backup_data
                st.success(f"Data loaded from {latest_backup_filepath}")
        except Exception as e:
            st.error(f"Error loading from backup: {e}")


if llms:
    st.write("### Available Models:")
    st.selectbox("", llms, on_change=None)
else:
    st.write("No models available.")

st.write(st.session_state.dataAnalysis)
