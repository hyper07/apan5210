import streamlit as st 
import os
import re
from langchain.document_loaders import PDFMinerLoader 
from utils.constants import DATA_ANALYSYS_RESPONSES

from agents.reviewAgent import ReviewAgent
from controllers.agentController import AgentController

persist_directory = "db"

st.set_page_config(page_title="ML Model Advisor", layout="wide")
st.title("Review Agent")
if "dataAnalysis" not in st.session_state or st.session_state.dataAnalysis is None:
    st.session_state.dataAnalysis = DATA_ANALYSYS_RESPONSES.copy()

if "dataAnalysis" in st.session_state and "reporter" in st.session_state.dataAnalysis:
    report_path = st.session_state.dataAnalysis["reporter"].get("file_path")
    if report_path and os.path.exists(report_path):
        if st.button("Review Report with LLM"):
            with st.spinner("Reviewing report..."):
                loader = PDFMinerLoader(report_path)
                documents = loader.load()
                full_text = " ".join([doc.page_content for doc in documents])
                review_agent = AgentController.getReviewAgent() 

                response = st.write_stream(review_agent.stream(full_text))
                reviewer_response = ""
                for chunk in response:
                    if isinstance(chunk, dict):
                        reviewer_response += chunk.get("generated_text", "")
                    else:
                        reviewer_response += str(chunk)
                result_text = re.sub(r"<think>.*?</think>", "", reviewer_response, flags=re.DOTALL).strip()
                st.session_state.dataAnalysis["reviewer"]["message"] = result_text
                st.session_state.dataAnalysis["reviewer"]["en"] = result_text

            st.success("Review completed!")
            st.markdown("#### Review Output:")
            st.rerun()

        # --- Language selection and translation ---
        if "reviewer" in st.session_state.dataAnalysis and st.session_state.dataAnalysis["reviewer"].get("message"):
            if "last_review_lang" not in st.session_state:
                st.session_state["last_review_lang"] = "English"

            lang = st.radio(
                "Select language",
                ["English", "Chinese", "Korean"],
                horizontal=True,
                index=["English", "Chinese", "Korean"].index(st.session_state["last_review_lang"])
            )

            # Only translate if language changed
            if lang != st.session_state["last_review_lang"]:
                translater = AgentController.getTranslateAgent()
                if lang == "Chinese" and st.session_state.dataAnalysis["reviewer"].get("message"):
                    with st.spinner("Translating to Chinese..."):

                        response = st.write_stream(translater.stream(lang, st.session_state.dataAnalysis['reviewer']['message']))
                        translated_text = ""
                        for chunk in response:
                            if isinstance(chunk, dict):
                                translated_text += chunk.get("text", str(chunk))
                            else:
                                translated_text += chunk
                        translated_text = re.sub(r"<think>.*?</think>", "", translated_text, flags=re.DOTALL).strip()
                        st.session_state.dataAnalysis["reviewer"]["cn"] = translated_text

                if lang == "Korean" and st.session_state.dataAnalysis["reviewer"].get("message"):
                    with st.spinner("Translating to Korean..."):

                        response = st.write_stream(translater.stream(lang, st.session_state.dataAnalysis['reviewer']['message']))

                        translated_text = ""
                        for chunk in response:
                            if isinstance(chunk, dict):
                                translated_text += chunk.get("text", str(chunk))
                            else:
                                translated_text += chunk
                        translated_text = re.sub(r"<think>.*?</think>", "", translated_text, flags=re.DOTALL).strip()
                        st.session_state.dataAnalysis["reviewer"]["kr"] = translated_text

                st.session_state["last_review_lang"] = lang

            # Display the review in the selected language
            if lang == "English" and st.session_state.dataAnalysis["reviewer"].get("en"):
                st.markdown(st.session_state.dataAnalysis["reviewer"]["en"])
            elif lang == "Chinese" and st.session_state.dataAnalysis["reviewer"].get("cn"):
                st.markdown(st.session_state.dataAnalysis["reviewer"]["cn"])
            elif lang == "Korean" and st.session_state.dataAnalysis["reviewer"].get("kr"):
                st.markdown(st.session_state.dataAnalysis["reviewer"]["kr"])

    else:
        st.warning("Please save PDF from Reporter.")