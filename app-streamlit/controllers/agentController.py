import os
from agents.analyzerAgent import AnalyzerAgent
from agents.codeAgent import CodeAgent
from agents.insightAgent import InsightAgent
from agents.reportAgent import ReportAgent
from agents.reviewAgent import ReviewAgent
from agents.translateAgent import TranslateAgent


class AgentController:
    def __init__(self):
        self.api_url = os.getenv("DEFAULT_API_URL", "")  # Replace with the correct hostname or IP if different
        self.model = os.getenv("DEFAULT_LLM_MODEL", "") 

    def setAPIUrl(self, url):
        self.api_url = url
        return self

    def getAPIUrl(self):
       return self.api_url

    def getAnalyzerAgent():
        return AnalyzerAgent()

    def getCoderAgent():
        return CodeAgent()

    def getInsightAgent():
        return InsightAgent()

    def getReportAgent():
        return ReportAgent()

    def getReviewAgent():
        return ReviewAgent()

    def getTranslateAgent():
        return TranslateAgent()
    

