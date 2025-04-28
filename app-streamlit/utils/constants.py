import os 
import chromadb
from chromadb.config import Settings 

BACKUP_DIR = "/tmp/files/backup/"

DEFAULT_API_URL="http://host.docker.internal:39870"
DEFAULT_LLM_MODEL="qwen:1.8b"

DEFAULT_ANALYZER_LLM_MODEL="gemma3:1b"
DEFAULT_CODER_LLM_MODEL="qwen2.5-coder:3b"
DEFAULT_INSIGHT_LLM_MODEL="deepseek-r1:1.5b"
DEFAULT_REPORT_LLM_MODEL="phi3:latest"
DEFAULT_REVIEWER_LLM_MODEL="llama3.2:1b"
DEFAULT_TRANSLATER_LLM_MODEL="qwen:1.8b"

DEFAULT_FILE_PATH='/tmp/files/'
DEFAULT_SAMPLE_PATH='/tmp/files/sample/'
DEFAULT_IMAGE_PATH='/tmp/files/images/'
DEFAULT_ML_MODEL_PATH='/tmp/files/models/'
DEFAULT_DATA_PATH='/tmp/files/data/'
DEFAULT_RESULT_PATH='/tmp/files/result/'
DEFAULT_CSV_PATH='/tmp/files/csv/'
DEFAULT_PDF_PATH='/tmp/files/pdf/'
DEFAULT_SCRIPT_PATH='/tmp/files/script/'

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
        "message" : {},
        "en" : "",
        "cn" : "",
        "kr" : "",
    },
    "reporter": {
        "translation" : "en",
        "message" : {},
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
        "message" : {},
        "code" : "",
    },
    "results": {
        "message" : {},
        "file_path" : "",
    }
}

SAMPLE_ANALYSYS_RESPONSES = {
    "analyzer": {
        "file_path": "/tmp/files/sample/analysis_data.csv",
        "target_variable" : "CTR",
        "variables_list" : ["id","targeting_score","visual_appeal","contextual_relevance","headline_length","cta_strength","position_on_page","ad_format","age_group","gender","location","time_of_day","day_of_week","brand_familiarity","device_type","ad_frequency","market_saturation","seasonality","headline_sentiment","headline_word_count","headline_power_words","body_text_length","body_word_count","body_sentiment","headline_question","headline_numbers","body_keyword_density","body_readability_score"],
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
        "script_output" : None,
        "script_errors" :  None,
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
        "code" : """import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the dataset
df = pd.read_csv('/tmp/files/sample/analysis_data.csv')

# Step 2: Define features (X) and target (y)
X = df.drop('CTR', axis=1)
y = df['CTR']

# Encode categorical features
X = pd.get_dummies(X)

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Initialize and train the Decision Tree Regressor
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# Step 5: Make predictions on the test set
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error (MSE): {mse:.4f}')
print(f'R-squared (R2 Score): {r2:.4f}')""",
    },
    "results": {
        "message" : "",
        "file_path" : "",
    }
}
