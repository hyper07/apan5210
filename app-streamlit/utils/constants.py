import os 
import chromadb
from chromadb.config import Settings 


CHROMA_SETTINGS = Settings(
        chroma_db_impl='duckdb+parquet',
        persist_directory='db',
        anonymized_telemetry=False
)

REQUIRED_MODELS = ["qwen:1.8b", "qwen2.5-coder:3b", "deepseek-r1:1.5b", "phi3:latest", "gemma3:1b", "llama3.2:1b", "all-minilm:l6-v2"]

DATA_ANALYSYS_RESPONSES = {
    "analyzer": {
        "file_path": "",
        "target_variable" : "",
        "variables_list" : [],
        "df": None,
        "message" : {},
        "models" : [],
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

SAMPLE_ANALYSYS_RESPONSES = {
    "analyzer": {
        "file_path": "",
        "target_variable" : "",
        "variables_list" : [],
        "df": None,
        "message" : {},
        "models" : [{
                    "ml_model":"Decision Tree",
                    "pros":"Simple and interpretable. Captures non-linear relationships.",
                    "cons":"",
                    "explanation":"A decision tree can help identify key factors influencing CTR, making it a good starting point."
                    },
                    {
                    "ml_model":"Random Forest",
                    "pros":"High accuracy with reduced variance. Interpretable through feature importance.",
                    "cons":"",
                    "explanation":"Effective at capturing complex patterns and interactions without user data."
                    },
                    {
                    "ml_model":"XGBoost",
                    "pros":"Handles missing values, robust to overfitting.",
                    "cons":"",
                    "explanation":"XGBoost is a powerful algorithm for structured data, often outperforming others."
                    }],
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
        "selected_model" : None,
        "message" : "",
        "code" : "",
    },
    "results": {
        "message" : "",
        "file_path" : "",
    }
}
