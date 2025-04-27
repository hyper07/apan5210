import streamlit as st
import pandas as pd
import numpy as np
import os
from contollers.serviceController import ServiceController
from contollers.agentController import AgentController
# from urllib.parse import urlparse
from utils.constants import DATA_ANALYSYS_RESPONSES, REQUIRED_MODELS

st.set_page_config(layout="wide")

st.write("# HOME")
# st.write("## This is a H2 Title!1")
# x = st.text_input("Movie", "Star Wars")

# if st.button("Click Me"):
#     st.write(f"Your favorite movie is `{x}`")

# file_path = "/tmp/files/sample/movies.csv"  # Updated file path
# if os.path.exists(file_path):
#     try:
#         data = pd.read_csv(file_path)
#         st.write(data)
#     except Exception as e:
#         st.error(f"Error reading the file: {e}")


# chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])
# st.bar_chart(chart_data)


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

st.session_state.currentPage = "Home"
st.session_state.dataAnalysis = DATA_ANALYSYS_RESPONSES

if llms:
    st.write("### Available Models:")
    for model in llms:
        st.write(f"- {model}")
else:
    st.write("No models available.")

st.write(st.session_state.dataAnalysis)

# st.session_state.llms = llms

# api_rul = AgentController().getAPIUrl()
# analyzer = AgentController().getAnalyzerAgent()

# st.write(api_rul)
# st.write(analyzer.getUrl())