import streamlit as st
import pandas as pd
import numpy as np
import os
from contollers.serviceController import ServiceController
from contollers.agentController import AgentController
# from urllib.parse import urlparse


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
    common_elements = list(set(llms if llms is not None else []) & set(["qwen:1.8b", "qwen2.5-coder:3b", "deepseek-r1:1.5b", "phi3:latest", "gemma3:1b", "llama3.2:1b"]))    

    if llms is None:
        st.error("Can't get models. Please check the API URL.")

    elif len(common_elements) < 6:
        # st.write("Initializing models. The time required depends on your internet speed ....")
        remains = 6 - len(common_elements)
        count = 1
        if 'qwen:1.8b' not in llms:
            status.update(
                label="("+str(count)+"/"+str(remains)+") Downloading qwen1.8b model for translater ...", expanded=False, state="running")
            result = ServiceController().pullModelFromSite("qwen:1.8b")
            count = count + 1
        if 'qwen2.5-coder:3b' not in llms:
            status.update(
                label="("+str(count)+"/"+str(remains)+") Downloading qwen2.5-coder:3b model for coder ...")
            result = ServiceController().pullModelFromSite("qwen2.5-coder:3b")
            count = count + 1
        if 'deepseek-r1:1.5b' not in llms:
            status.update(
                label="("+str(count)+"/"+str(remains)+") Downloading deepseek-r1:1.5b model for insight ...")
            result = ServiceController().pullModelFromSite("deepseek-r1:1.5b")
            count = count + 1
        if 'phi3:latest' not in llms:
            status.update(
                label="("+str(count)+"/"+str(remains)+") Downloading phi3:latest model for reporter ...")
            result = ServiceController().pullModelFromSite("phi3:latest")
            count = count + 1
        if 'gemma3:1b' not in llms:
            status.update(
                label="("+str(count)+"/"+str(remains)+") Downloading gemma3:1b model for analyzer ...")
            result = ServiceController().pullModelFromSite("gemma3:1b")
            count = count + 1
        if 'llama3.2:1b' not in llms:
            status.update(
                label="("+str(count)+"/"+str(remains)+") Downloading llama3.2:1b model for review ...")
            result = ServiceController().pullModelFromSite("llama3.2:1b")
            count = count + 1

        st.session_state.llms = ServiceController().getModelListOnly()

st.session_state.currentPage = "Home"
st.session_state.dataAnalysis = {
    "analyzer": {
        "file_path": "",
        "predict_variable" : "",
        "variables_list" : [],
        "ml_model" : "",
        "message" : "",
    },
    "insight": {
        "translation" : "en",
        "message" : "",
        "en" : "",
        "cn" : "",
        "kr" : "",
    },
    "reporter": {
        "translation" : "en",
        "message" : "",
        "en" : "",
        "cn" : "",
        "kr" : "",
    },
    "reviewer": {
        "translation" : "en",
        "message" : "",
        "en" : "",
        "cn" : "",
        "kr" : "",
    },
    "translator": {
        "en" : "",
        "cn" : "",
        "kr" : "",
    },
    "coder":  {
        "message" : "",
        "code" : "",
    },
    "results": {
        "message" : "",
        "file_path" : "",
    }
}

if llms:
    st.write("### Available Models:")
    for model in llms:
        st.write(f"- {model}")
else:
    st.write("No models available.")



# st.session_state.llms = llms

# api_rul = AgentController().getAPIUrl()
# analyzer = AgentController().getAnalyzerAgent()

# st.write(api_rul)
# st.write(analyzer.getUrl())