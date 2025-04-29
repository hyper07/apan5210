import os 
import chromadb
from chromadb.config import Settings 

BACKUP_DIR = "/tmp/files/backup/"

DEFAULT_API_URL="http://host.docker.internal:39870"
DEFAULT_LLM_MODEL="deepseek-r1:1.5b"

DEFAULT_ANALYZER_LLM_MODEL="deepseek-r1:1.5b"
DEFAULT_CODER_LLM_MODEL="qwen2.5-coder:3b"
DEFAULT_INSIGHT_LLM_MODEL="phi3:latest"
DEFAULT_REPORT_LLM_MODEL="llama3.2:1b"
DEFAULT_REVIEWER_LLM_MODEL="gemma3:1b"
DEFAULT_TRANSLATER_LLM_MODEL="phi4-mini"

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

REQUIRED_MODELS = ["phi4-mini", "qwen2.5-coder:3b", "deepseek-r1:1.5b", "phi3:latest", "gemma3:1b", "llama3.2:1b", "all-minilm:l6-v2"]

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
        "message" : "Given the output and context of your Random Forest model for predicting Click Through Rate (CTR), here is a professional assessment along with some insights:\n\nYour results indicate that there is room for improvement in CTR prediction accuracy. Although an MSE value as low as 0.0201 might seem decent, the corresponding R-squared score of just over 46% suggests that approximately half of the variance in your target variable (CTR) isn't being captured by the model alone—which may be attributable to many factors not accounted for or due to inherent noise within CTR data.\n\nHere are some industry-specific insights and best practices:\n\n1. **Feature Engineering** - The effectiveness of Random Forest models, particularly in domains like online advertising where user behavior can greatly influence outcomes such as click rates, hinges on the selection of meaningful features that drive engagement effectively. Consider leveraging domain knowledge to engineer more predictive attributes — for example, time-based decay functions might be used if recent ad exposure is believed to have a stronger impact than older ones.\n\n2. **Temporal Dynamics** - CTR can exhibit strong temporal patterns due to evolving trends and seasonality in the industry. Including features that account for timing, like day of week or hour of day, might capture these dynamics better if they're relevant predictors based on your dataset context.\n\n3. **Evaluate Alternative Models** - While Random Forest is robust to outliers and capable with unstructured data due to its nature as an ensemble method, other models like Gradient Boosting Machines (GBM) might provide better performance in such a specific application by optimizing for the most predictive splits.\n\n4. **Hyperparameter Tuning** - Experimentation with different Random Forest configurations could enhance results; this includes playing around with 'n_estimators', min_samples_split, and max depth to find an optimal trade-off between bias and variance that suits your dataset characteristics better.\n\n5. **Cross-validation Strategy** - Employ a more robust cross-validation strategy than the simple train-test split such as k-fold or time series cross-validation if temporal patterns are relevant, to ensure that model performance is consistent across different data segments and generalizes well for unseen interactions.\n\n6. **Data Quality Insight** - Given only 20% of your dataset was used for testing the Random Forest predictions (assuming this from 'test_size=0.2'), consider increasing the test size or employing techniques like bootstrapping to validate model performance, which could also provide more confidence in its predictive power with real-world data where variance is higher and patterns may change over time due to market dynamics.\n\n7. **Feature Selection Techniques** - Utilize feature selection methods such as recursive feature elimination or mutual information scores within the Random Forest algorithm itself, which could aid by identifying more impactful features for prediction without increasing complexity unnecessarily. \n\n8.0 MSE & R2 Score: Industry standards can vary widely but generally speaking in highly competitive and fast-changing fields like online advertising, aiming to consistently achieve high CTR with lower variance between sessions is key — hence a strong model would ideally result in an MSE below 0.01 and R2 above 50%.\n\nIn summary, while the Random Forest algorithm appears as a suitable starting point for predictive tasks such as forecasting CTRs due to its robustness against overfitting, there's ample scope for performance enhancement by refining features, exploring alternative model architectures and hyperparameters tuning in alignment with industry benchmarks.",
        "script_output" : None,
        "script_errors" :  None,
        "en" : "Given the output and context of your Random Forest model for predicting Click Through Rate (CTR), here is a professional assessment along with some insights:\n\nYour results indicate that there is room for improvement in CTR prediction accuracy. Although an MSE value as low as 0.0201 might seem decent, the corresponding R-squared score of just over 46% suggests that approximately half of the variance in your target variable (CTR) isn't being captured by the model alone—which may be attributable to many factors not accounted for or due to inherent noise within CTR data.\n\nHere are some industry-specific insights and best practices:\n\n1. **Feature Engineering** - The effectiveness of Random Forest models, particularly in domains like online advertising where user behavior can greatly influence outcomes such as click rates, hinges on the selection of meaningful features that drive engagement effectively. Consider leveraging domain knowledge to engineer more predictive attributes — for example, time-based decay functions might be used if recent ad exposure is believed to have a stronger impact than older ones.\n\n2. **Temporal Dynamics** - CTR can exhibit strong temporal patterns due to evolving trends and seasonality in the industry. Including features that account for timing, like day of week or hour of day, might capture these dynamics better if they're relevant predictors based on your dataset context.\n\n3. **Evaluate Alternative Models** - While Random Forest is robust to outliers and capable with unstructured data due to its nature as an ensemble method, other models like Gradient Boosting Machines (GBM) might provide better performance in such a specific application by optimizing for the most predictive splits.\n\n4. **Hyperparameter Tuning** - Experimentation with different Random Forest configurations could enhance results; this includes playing around with 'n_estimators', min_samples_split, and max depth to find an optimal trade-off between bias and variance that suits your dataset characteristics better.\n\n5. **Cross-validation Strategy** - Employ a more robust cross-validation strategy than the simple train-test split such as k-fold or time series cross-validation if temporal patterns are relevant, to ensure that model performance is consistent across different data segments and generalizes well for unseen interactions.\n\n6. **Data Quality Insight** - Given only 20% of your dataset was used for testing the Random Forest predictions (assuming this from 'test_size=0.2'), consider increasing the test size or employing techniques like bootstrapping to validate model performance, which could also provide more confidence in its predictive power with real-world data where variance is higher and patterns may change over time due to market dynamics.\n\n7. **Feature Selection Techniques** - Utilize feature selection methods such as recursive feature elimination or mutual information scores within the Random Forest algorithm itself, which could aid by identifying more impactful features for prediction without increasing complexity unnecessarily. \n\n8.0 MSE & R2 Score: Industry standards can vary widely but generally speaking in highly competitive and fast-changing fields like online advertising, aiming to consistently achieve high CTR with lower variance between sessions is key — hence a strong model would ideally result in an MSE below 0.01 and R2 above 50%.\n\nIn summary, while the Random Forest algorithm appears as a suitable starting point for predictive tasks such as forecasting CTRs due to its robustness against overfitting, there's ample scope for performance enhancement by refining features, exploring alternative model architectures and hyperparameters tuning in alignment with industry benchmarks.",
        "cn" : "",
        "kr" : "",
    },
    "reporter": {
        "translation" : "en",
        "file_path" : "",
        "message" : "",
        "en" : "**Click Through Rate (CTR) Prediction using Decision Tree Regressor**\n\n**Introduction**\n\nThis report evaluates the performance of a Decision Tree Regressor model for predicting Click Through Rate (CTR). Our goal is to identify areas for improvement in CTR prediction accuracy and provide industry-specific insights and best practices.\n\n**Dataset Overview**\n\nWe loaded a sample dataset from '/tmp/files/sample/analysis_data.csv', which consisted of 1000 rows with two features: 'Code' and 'Feature'. The target variable was 'CTR'.\n\n**Feature Engineering and Encoding**\n\nBefore training the model, we encoded categorical features using pandas' `get_dummies` function. We then dropped the original 'CTR' feature.\n\n**Model Training**\n\nWe initialized a Decision Tree Regressor model with a random state of 42 and trained it on the preprocessed data using the `train_test_split` function from scikit-learn.\n\n**Model Evaluation**\n\nAfter training, we made predictions on the test set using the `predict` method. We then evaluated the model's performance using the Mean Squared Error (MSE) and R-squared score metrics.\n\n**Results**\n\nOur results indicated an MSE of 0.0201 and an R2 score of approximately 46%. These values suggest that the model performed reasonably well, but there was room for improvement in terms of accuracy and variance capture.\n\n**Insights and Recommendations**\n\nBased on our analysis and insights, we identified several areas for improvement:\n\n1. **Feature Engineering**: The effectiveness of Random Forest models depends on the selection of meaningful features that drive engagement effectively. We recommend leveraging domain knowledge to engineer more predictive attributes.\n2. **Temporal Dynamics**: CTR can exhibit strong temporal patterns due to evolving trends and seasonality in online advertising. Including features that account for timing, such as day of week or hour of day, might capture these dynamics better.\n3. **Hyperparameter Tuning**: Experimentation with different Random Forest configurations could enhance results by finding an optimal trade-off between bias and variance.\n4. **Cross-validation Strategy**: Employing a more robust cross-validation strategy to ensure consistency across different data segments and generalization to unseen interactions is recommended.\n5. **Data Quality Insight**: Increasing the test size or employing techniques like bootstrapping can validate model performance, providing more confidence in its predictive power with real-world data.\n\n**Additional Recommendations**\n\n* Utilize feature selection methods, such as recursive feature elimination or mutual information scores, within the Random Forest algorithm to identify more impactful features.\n* Consider using Gradient Boosting Machines (GBM) instead of Decision Tree Regressor, which might perform better in specific applications by optimizing for the most predictive splits.\n\n**Industry Standards**\n\nIn highly competitive and fast-changing fields like online advertising, aiming to consistently achieve high CTR with lower variance between sessions is key. A strong model would ideally result in an MSE below 0.01 and R2 above 50%.\n\nBy implementing these recommendations and leveraging industry-specific insights, we believe that our Decision Tree Regressor model can be further improved and become a more effective tool for predicting Click Through Rate.",
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
