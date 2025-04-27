#imports
import streamlit as st
import os
from pathlib import Path
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, RobustScaler, OrdinalEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import joblib
import time
import logging
from contollers.serviceController import ServiceController

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')

ServiceController.initialize_llms_session_state()

#sidebar
#st.sidebar.title("Example")
pages=["Project", "Dataset", "Data Visualization", "Preprocessing", "Modeling", "Conclusion"]
page=st.sidebar.radio("Menu", pages)
sample_folder = '/tmp/files/sample/'
model_folder = '/tmp/files/models/'

# FUNCTIONS FOR DATA AND RESOURCES --------------------------------------------------------------
# LOADING SAVED MODELS
#@st.cache_resource
def load_model(filename):
    """
    Load a model from a file with joblib, ensuring compatibility with scikit-learn versions.
    """
    try:
        # Ensure compatibility by using scikit-learn's joblib utilities
        model = joblib.load(model_folder + filename)
        return model
    except FileNotFoundError:
        st.error(f"The template {filename} could not be loaded.")
        return None

# Load models as needed
def get_model(model_name):
    """
    Returns the requested model, loading it if necessary.
    """

    if model_name == 'rf':
        if 'model_rf' not in st.session_state:
            st.session_state['model_rf'] = load_model('random_forest_model.pkl')

        return st.session_state['model_rf']
    elif model_name == 'xgb':
        if 'model_xgb' not in st.session_state:
            st.session_state['model_xgb'] = load_model('xgboost_model.pkl')
        
        return st.session_state['model_xgb']
    elif model_name == 'lgb':
        if 'model_lgb' not in st.session_state:

            st.session_state['model_lgb'] = load_model('lightgbm_model.pkl')
        return st.session_state['model_lgb']

# Function to initialize results
def initialize_results():
    if 'results_rf' not in st.session_state:
        st.session_state['results_rf'] = None
    if 'results_xgb' not in st.session_state:
        st.session_state['results_xgb'] = None
    if 'results_lgb' not in st.session_state:
        st.session_state['results_lgb'] = None

# Initialize results in session state
initialize_results()

def save_model(model, filename):
    """
    Save a model to a file with joblib.
    """
    joblib.dump(model, filename)

# FUNCTION TO LOAD DATA FOR THE DATAFRAME
@st.cache_data
def load_data():
    return pd.read_csv(sample_folder+'bank.csv')

# FUNCTIONS FOR PAGE 2 DATASET
@st.cache_data
def get_data_summary(df):
    return {
        "head": df.head(10),
        "shape": df.shape,
        "description": df.describe(),
        "dtypes": df.dtypes,
        "missing_values": df.isna().sum(),
        "duplicates": df[df.duplicated()],
    }

@st.cache_data
def calculate_balance_statistics(df):
    mean_balance_all = df['balance'].mean()
    mean_balance_yes = df[df['deposit'] == 'yes']['balance'].mean()
    mean_balance_no = df[df['deposit'] == 'no']['balance'].mean()
    return mean_balance_all, mean_balance_yes, mean_balance_no

@st.cache_data
def calculate_duration_statistics(df):
    df['duration_minutes'] = round((df['duration'] / 60.0), 2)
    median_duration_all = df['duration_minutes'].median()
    median_duration_deposit_yes = df[df['deposit'] == 'yes']['duration_minutes'].median()
    median_duration_deposit_no = df[df['deposit'] == 'no']['duration_minutes'].median()
    return median_duration_all, median_duration_deposit_yes, median_duration_deposit_no

@st.cache_data
def calculate_previous_statistics(df):
    mean_previous_all = np.mean(df['previous'])
    mean_previous_yes = np.mean(df[df['deposit'] == 'yes']['previous'])
    mean_previous_no = np.mean(df[df['deposit'] == 'no']['previous'])
    return mean_previous_all, mean_previous_yes, mean_previous_no

@st.cache_data
def check_pdays_previous_equivalence(df):
    filtered_df = df[df['pdays'] == -1]
    equivalence = (filtered_df['previous'] == 0).all()
    return equivalence

# FUNCTIONS FOR PAGE 3 PREPROCESSING
## STEP-BY-STEP PREPROCESSING
@st.cache_data
def discretize_age(df):
    df['age_cat'] = pd.cut(df.age, bins=[18, 29, 40, 50, 60, np.inf], labels=['18-29', '30-40', '40-50', '50-60', '60+'], right=False)
    df = df.drop('age', axis=1)
    return df

@st.cache_data
def replace_unknown_education(df):
    most_frequent = df[df['education'] != 'unknown']['education'].mode()[0]
    df['education'] = df['education'].replace('unknown', most_frequent)
    return df

@st.cache_data
def transform_pdays(df):
    df['pdays_contact'] = df['pdays'].apply(lambda x: 'no' if x == -1 else 'yes')
    df['pdays_days'] = df['pdays'].apply(lambda x: 0 if x == -1 else x)
    df = df.drop('pdays', axis=1)
    return df

## SPLIT FEATURES AND TARGET
@st.cache_data
def separate_features_target(df, target_column):
    """Separate features and target."""
    features = df.drop(target_column, axis=1)
    target = df[target_column]
    return features, target

@st.cache_data
def split_train_test(features, target, test_size=0.25, random_state=42):
    """Split data into training and test sets."""
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

## PREPROCESSING PIPELINE
### Function to preprocess data
@st.cache_data
def preprocess_data(df):
    # Convert 'month' to int using a mapping
    month_mapping = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12 }
    df['month'] = df['month'].map(month_mapping)

    # Discretize the 'age' variable
    df['age_cat'] = pd.cut(df.age, bins=[18, 29, 40, 50, 60, np.inf], labels=['18-29', '30-40', '40-50', '50-60', '60+'], right=False)

    # Split the 'pdays' column into two columns
    df['pdays_contact'] = df['pdays'].apply(lambda x: 'no' if x == -1 else 'yes')
    df['pdays_days'] = df['pdays'].apply(lambda x: 0 if x == -1 else x)

    # Separate features and target
    features = df.drop(columns=['deposit', 'age', 'pdays'])
    target = df['deposit']

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=42)
    return X_train, X_test, y_train, y_test

### Function to preprocess and transform data
@st.cache_data
def preprocess_and_transform(X_train, X_test, y_train, y_test):
    # Define features
    binary_features = ['default', 'housing', 'loan', 'pdays_contact']
    categorical_features = ['job', 'marital', 'contact', 'poutcome']
    ordinal_features = ['education', 'age_cat']
    numerical_features = ['balance', 'campaign', 'duration', 'previous', 'pdays_days']
    cyclic_features = ['day', 'month']
    
    # Pipeline for binary variables
    binary_pipeline = Pipeline([
        ('binary_encoding', FunctionTransformer(lambda x: x.replace({'yes': 1, 'no': 0}))),
    ])

    # Pipeline for categorical variables
    categorical_pipeline = Pipeline([
        ('onehot_encoding', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Preprocessing for 'education'
    most_frequent_education = X_train['education'].mode()[0]
    X_train['education'] = X_train['education'].replace('unknown', most_frequent_education)
    X_test['education'] = X_test['education'].replace('unknown', most_frequent_education)

    # Pipeline for ordinal encoding of 'education' and 'age_cat'
    education_categories = ['primary', 'secondary', 'tertiary']
    age_cat_categories = ['18-29', '30-40', '40-50', '50-60', '60+']

    ordinal_pipeline = Pipeline([
        ('ordinal_encoding', OrdinalEncoder(categories=[education_categories, age_cat_categories]))
    ])

    # Pipeline for numerical variables
    numerical_pipeline = Pipeline([
        ('duration_minutes', FunctionTransformer(lambda x: x / 60.0 if 'duration' in x else x)),
        ('scaler', RobustScaler())
    ])

    # Cyclic encoding for 'day' and 'month'
    def encode_cyclic(df, column, max_value):
        df[column + '_sin'] = np.sin(2 * np.pi * df[column] / max_value)
        df[column + '_cos'] = np.cos(2 * np.pi * df[column] / max_value)
        return df

    X_train = encode_cyclic(X_train, 'day', 31)
    X_train = encode_cyclic(X_train, 'month', 12)

    X_test = encode_cyclic(X_test, 'day', 31)
    X_test = encode_cyclic(X_test, 'month', 12)

    # Full preprocessing with ColumnTransformer
    preprocessor = ColumnTransformer([
        ('binary', binary_pipeline, binary_features),
        ('categorical', categorical_pipeline, categorical_features),
        ('ordinal', ordinal_pipeline, ordinal_features),
        ('cyclic', 'passthrough', ['day_sin', 'day_cos', 'month_sin', 'month_cos']),
        ('numerical', numerical_pipeline, numerical_features),
    ])

    # Apply preprocessor to X_train and X_test
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Encode the target variable 'deposit'
    label_encoder = LabelEncoder()
    y_train_processed = label_encoder.fit_transform(y_train)
    y_test_processed = label_encoder.transform(y_test)

    # Convert to DataFrame for visualization
    columns = (binary_features + 
           list(preprocessor.named_transformers_['categorical'].named_steps['onehot_encoding'].get_feature_names_out(categorical_features)) +
           ordinal_features + 
           ['day_sin', 'day_cos', 'month_sin', 'month_cos'] + 
           numerical_features)

    X_train_processed_df = pd.DataFrame(X_train_processed, columns=columns, index=X_train.index).sort_index()
    X_test_processed_df = pd.DataFrame(X_test_processed, columns=columns, index=X_test.index).sort_index()
    y_train_processed_df = pd.DataFrame({'Deposit': y_train_processed}, index=y_train.index).sort_index()
    y_test_processed_df = pd.DataFrame({'Deposit': y_test_processed}, index=y_test.index).sort_index()

    return X_train_processed_df, X_test_processed_df, y_train_processed_df, y_test_processed_df

# FUNCTIONS FOR MODELING
@st.cache_data
def remove_duration(X_train_processed_df, X_test_processed_df):
    """
    Removes the 'duration' column from the X_train and X_test datasets.

    Parameters:
    X_train_processed_df (pd.DataFrame): Preprocessed training set.
    X_test_processed_df (pd.DataFrame): Preprocessed test set.

    Returns:
    X_train_processed_df (pd.DataFrame): Training set without the 'duration' column.
    X_test_processed_df (pd.DataFrame): Test set without the 'duration' column.
    """
    # Remove the 'duration' column from the datasets
    if 'duration' in X_train_processed_df.columns:
        X_train_processed_df = X_train_processed_df.drop(columns=['duration'])
    if 'duration' in X_test_processed_df.columns:
        X_test_processed_df = X_test_processed_df.drop(columns=['duration'])
    return X_train_processed_df, X_test_processed_df

    
# Train and evaluate the model
def train_and_evaluate_model(model, X_train_processed, X_test_processed, y_train_processed, y_test_processed):
    model.fit(X_train_processed, y_train_processed)
    y_pred = model.predict(X_test_processed)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_processed, y_pred)
    precision = precision_score(y_test_processed, y_pred, pos_label=1)  # 1 corresponds to 'yes'
    recall = recall_score(y_test_processed, y_pred, pos_label=1)
    f1 = f1_score(y_test_processed, y_pred, pos_label=1)
    
    # Classification report
    report = classification_report(y_test_processed, y_pred, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_test_processed, y_pred)
    
    # Feature importances
    importances = model.feature_importances_ if hasattr(model, 'feature_importances_') else None

    return {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'report': report,
        'confusion_matrix': cm,
        'importances': importances
    }

def save_model(model, filename):
    """
    Save a model to a file with joblib.
    """
    joblib.dump(model, filename)

#NEW TRAIN WITH BEST PARAMS and SAVE
def train_and_evaluate_and_save(model_class, params, model_name, key):
    # Create an instance of the model with the specified parameters
    model = model_class(**params)
    
    # Display model parameters
    with st.expander(f"Show parameters for model {model_name}"):
        st.write(f"Model {model_name} created with the following parameters:")
        st.write(params)
    
    # Load and preprocess data
    df = load_data()
    # df = df.head(100)
    X_train, X_test, y_train, y_test = preprocess_data(df)
    X_train_processed_df, X_test_processed_df, y_train_processed_df, y_test_processed_df = preprocess_and_transform(X_train, X_test, y_train, y_test)

    # Remove the 'duration' column from the processed datasets
    X_train_processed_df, X_test_processed_df = remove_duration(X_train_processed_df, X_test_processed_df)

    # Train and evaluate the model
    results = train_and_evaluate_model(model, X_train_processed_df, X_test_processed_df, y_train_processed_df['Deposit'], y_test_processed_df['Deposit'])
    
    # Save the model
    save_model(model, f'{model_folder+model_name.lower()}_model.pkl')
    
    # Store results in st.session_state
    st.session_state[key] = results
    
    # Display results
    st.write("**Classification report**")
    display_classification_report(results['report'])
    
    st.write("")
    plot_confusion_matrix(results['confusion_matrix'])
    
    st.write("")
    plot_feature_importances(results['importances'], X_train_processed_df.columns)



# Visualization of the confusion matrix
def plot_confusion_matrix(cm):
    """
    Displays the confusion matrix as a heatmap and the raw matrix.

    Parameters:
    cm (np.ndarray): Confusion matrix.
    """
    # Define labels for axes
    x_labels = ['Actual No', 'Actual Yes'] # Columns: Actual classes
    y_labels = ['Prediction No', 'Prediction Yes'] # Rows: Predicted classes
    
    # Custom color palette
    colorscale = [
        [0, '#f0e4a1'], # Pastel yellow
        [0.33, '#f8d7da'], # Very light pastel pink
        [0.67, '#f4a6b0'], # Light pastel pink
        [1, '#f59b4f'] # Pastel orange
    ]
    

    # Creating the annotated heatmap
    fig = ff.create_annotated_heatmap(
        z=cm,
        x=x_labels,
        y=y_labels,
        colorscale=colorscale,
        zmin=0,
        zmax=np.max(cm),
        annotation_text=cm,
        showscale=True
    )
    
    # Adjust the arrangement of values ​​in the matrix
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Actual Classes',
        yaxis_title='Predicted Classes',
        xaxis=dict(tickvals=[0, 1], ticktext=['No', 'Yes']),
        yaxis=dict(tickvals=[0, 1], ticktext=['No', 'Yes']),
        height=500,
        width=600   
    )
    
    # Reverse the order of the axes so that 'No' appears at the top left and 'Yes' at the bottom right
    #fig.update_xaxes(autorange='reversed') # Reverse the order of the x labels
    fig.update_yaxes(autorange='reversed') # Reverse the order of the y labels
    
    # Display the figure
    st.plotly_chart(fig)
    
    # Display the raw confusion matrix for verification
    #st.write("Raw confusion matrix:")
    #cm_df = cm, index=['No', 'Yes'], columns=['No', 'Yes'])
    #st.write(cm_df)



# Display the classification report
@st.cache_data
def display_classification_report(report):
    report_df = pd.DataFrame(report).transpose()
    st.write(report_df)

# Visualization of feature importances
def plot_feature_importances(importances, feature_names):
    """
    Displays feature importances as a bar chart with a different color for each bar.

    Parameters:
    importances (np.ndarray): Feature importances.
    feature_names (Index): Feature names.
    """
    if importances is None:
        st.write("No feature importance available.")
        return
    
    # Create a DataFrame for feature importances and names
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Sort importances in descending order
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    
    # Create the bar chart with Plotly Express
    fig = px.bar(
        importance_df,
        x='Importance',
        y='Feature',
        orientation='h',  # Horizontal bars
        color='Feature',
        color_discrete_sequence=px.colors.qualitative.Pastel,
        title='Importance of Features',
        labels={'Importance': 'Importance', 'Feature': 'Features'}
    )
    
    # Customize the display of the y-axis labels
    fig.update_layout(
        yaxis_title='Features',
        xaxis_title='Importance',
        yaxis=dict(tickmode='linear'),  # Adjust the angle of the labels if necessary
        height=800,  # Adjust the height of the chart to improve bar visibility
        bargap=0.1  # Reduce the gap between bars to make them wider
    )
    
    # Display the figure
    st.plotly_chart(fig)





# FUNCTIONS FOR PAGE CONTENT --------------------------------------------------------------
#PAGE 0 INTRO PROJECT
def show_projet_page():
    st.title("Predicting the success of a bank's marketing campaign")
    st.header("Context")
    #Creation of two columns

    st.markdown("""
The data in the `bank.csv` dataset are related to a direct marketing campaign of a Portuguese banking institution conducted between May 2008 and November 2010. \n
The marketing campaigns were based on telephone calls. \n
Several contacts with the same customer were made to find out whether or not the latter had **subscribed to the product: term bank deposit.**
        """)


    st.markdown("""
Term deposit is a type of investment offered by banks and financial institutions. \n
In a term deposit, an individual deposits a certain amount of money with the bank for a specified period, called the term or maturity.
The money is held by the bank for the specified period, during which it earns a fixed interest rate.
                    """)

    st.write("")
    st.divider()
    st.header("Objective")

    st.markdown("""
Our goal is to analyze the entire dataset in order to **identify trends and factors that influence the subscription of a term deposit product** by customers. \n
On the technical side, the analysis of this dataset represents a unique opportunity to put into practice our skills in data analysis, visualization with Plotly, pre-processing, modeling of classification algorithms up to the publication on Streamlit and Github.
        """)

    st.markdown("""
From a strategic perspective, this project aims to provide **valuable insights to increase subscription rates for the "term deposit" product**. \n
By identifying the key success factors of previous campaigns, we can help the financial institution optimize its marketing resources, target its customers more effectively and improve its conversion rates, thus generating additional revenue.
                    """)

    st.write("")
    st.divider()
    st.header("Source")
   

    st.markdown("""
[Moro et al., 2014] S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, Elsevier, 62:22-31, June 2014
        """)

        # Add a button
    st.markdown(
        """
        <a href="https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset/data" target="_blank" style="
        display: inline-block; 
        background-color: #56CEB2; 
        color: white; 
        font-size: 16px; 
        font-weight: bold; 
        text-align: center; 
        padding: 10px 20px; 
        border-radius: 5px; 
        text-decoration: none; 
        width: 400px;
        ">Dataset Source</a>
        """, unsafe_allow_html=True
    )
    st.write("")

# PAGE1 DATASET
def show_data_set_page():
    st.title("The dataset")
    st.markdown('The bank.csv dataset is based on the UCI Bank Marketing dataset, which can be read here: http://archive.ics.uci.edu/ml/datasets/Bank+Marketing. Creators: S. Moro, P. Rita, P. Cortez.')
    st.header('The variables')
    df2=pd.read_excel(sample_folder+"variables.xlsx")
    st.dataframe(df2)
    st.divider()
    st.header('Data Overview')
    # Load data
    df = load_data()
    # Get data summary
    data_summary = get_data_summary(df)
    #TABS CONTAINERS:
    tab1, tab2, tab3, tab4, tab5,tab6= st.tabs(["Overview", "Dimensions", "Statistics", "Types", "Nulls", "Duplicates"])
    with tab1:
        st.code('df.head(10)')
        st.markdown("Overview of the first 10 lines")
        st.dataframe(data_summary["head"])
    with tab2:
        st.markdown("The dimensions of the dataset: 17 variables and 11,162 rows.")
        st.code('df.shape')
        st.write(data_summary["shape"])
    with tab3:
        st.code('df.describe()')
        st.write(data_summary["description"])
        st.subheader("Observation")
        st.markdown("""
        - age: 50% of the values ​​are between 32 and 49 years old. Many extreme values: max 95.
        - balance: 50% of the values ​​are between 122 and 1708. Presence of negative values ​​and extreme values: min -6,847, max 81,204.
        - duration: 50% of the values ​​are between 138 sec (2min) and 496 (8min). Presence of extreme values: max 3,881.
        - campaign: 50% of the values ​​are between 1 and 3 contacts. Presence of extreme values: max 63.
        - pdays: 50% of the values ​​are between - 1 and 20. The median is -1 which means that half of the customers have never been contacted before this campaign. Presence of extreme values: max 854.
        - previous: 50% of the values ​​are between 0 and 1. Presence of extreme values: max 58.
        """)
    with tab4:
        st.markdown("Data types: ")
        st.code('df.dtypes')
        st.write(data_summary["dtypes"])
    with tab5:
        st.markdown("No missing values: ")
        st.code('df.isna().sum()')
        st.write(data_summary["missing_values"])
    with tab6:
        st.markdown("No duplicates: ")
        st.code('df[df.duplicated()]')
        st.write(data_summary["duplicates"])
        st.divider()

    st.header('Additional Statistics')
       
    # TABS CONTAINERS :
    tab1, tab2, tab3, tab4= st.tabs(["Balance", "Duration","Previous","pdays-previous"])
       
    #TAB1 AVERAGE BALANCE
    with tab1:
        st.markdown("**Average Account Balance for depositor and non-depositor customers and for total customers.**")
        code = '''
        # Balance averaging
        mean_balance_all = df['balance'].mean()
        mean_balance_yes = df[df['deposit'] == 'yes']['balance'].mean()
        mean_balance_no = df[df[ 'deposit'] == 'no']['balance'].mean()
    '''
        st.code(code, language='python')
        mean_balance_all, mean_balance_yes, mean_balance_no = calculate_balance_statistics(df)
        st.write("The average bank balance for the dataset is:", round(mean_balance_all, 2))
        st.write("The average balance for customers who made a deposit is:", round(mean_balance_yes, 2))
        st.write("The average balance for customers who did not make a deposit is:", round(mean_balance_no, 2))

    #TAB2 AVERAGE DURATION
    with tab2:
        st.markdown("**Median contact duration in minutes for depositor and non-depositor customers and for total customers.**")
        # Display code
        code = '''
        # Convert the duration column from seconds to minutes
        df['duration_minutes'] = round((df['duration'] / 60.0), 2)
        # Calculate the median duration for all data
        median_duration_all = df['duration_minutes'].median()
        # Calculate the median duration for those who made a deposit (deposit = yes)
        median_duration_deposit_yes = df[df['deposit'] == 'yes']['duration_minutes'].median()
        # Calculate the median duration for those who did not make a deposit (deposit = no)
        median_duration_deposit_no = df[df['deposit'] == 'no']['duration_minutes'].median()
        '''
        st.code(code, language='python')
        # Execute code
        median_duration_all, median_duration_deposit_yes, median_duration_deposit_no = calculate_duration_statistics(df)
        # Display results
        st.write("The median customer contact duration of the campaign is:", round(median_duration_all, 2), "minutes.")
        st.write("For customers who made a deposit:", round(median_duration_deposit_yes, 2), "minutes.")
        st.write("For customers who did not make a deposit:", round(median_duration_deposit_no, 2), "minutes.")
       
    #TAB3 PREVIOUS
    with tab3:
        st.markdown("**Average number of customer contacts before this campaign for depositor or non-depositor customers and for total customers.**")
        # Display code
        code = ''' 
        # Calculate the means of previous for each group of deposit
        mean_previous_all = np.mean(df['previous'])
        mean_previous_yes = np.mean(df[df['deposit'] == 'yes'][ 'previous'])
        mean_previous_no = np.mean(df[df['deposit'] == 'no']['previous'])
        '''
        st.code(code, language='python')
        # Execute code
        mean_previous_all, mean_previous_yes, mean_previous_no = calculate_previous_statistics(df)
        # Display results
        st.write("The average number of customer contacts before this campaign is:", round(mean_previous_all, 2), "contacts.")
        st.write("For customers who made a deposit:", round(mean_previous_yes, 2), " contacts on average.")
        st.write("For customers who did not make a deposit:", round(mean_previous_no, 2), " contacts on average.")
       
    #TAB4 PDAYS=PREVIOUS
    with tab4:
        st.markdown("""
        **Is the value -1 of pdays equivalent to the value 0 of previous?**
        - pdays : Number of days since the last contact of a previous campaign (value -1 equals no contact)
        - previous : Number of contacts before this campaign (value 0 equals no contact)
        """)
        # Display code
        code = '''
        # Filter lines where pdays is -1
        filtered_df = df[df['pdays'] == -1]
        # Check if, for these lines, previous is 0
        equivaut = (filtered_df['previous'] == 0).all()
        '''

        st.code(code, language='python')
        # Execute code
        equivalence = check_pdays_previous_equivalence(df)
        st.write(f"Do all lines with pdays is equal to -1 and previous value is eqault to 0 => {equivalence}")

# PAGE2 DATAVIZ
def show_data_viz_page():
    st.title("Data Visualization")
    # Load data
    df = load_data()
    col1, col2, col3, col4 = st.columns([1, 1, 1,1])
    button1 = col1.button("Deposit target variable")
    button2 = col2.button("Numerical variables")
    button3 = col3.button("Categorical variables")
    button4 = col4.button("Variables versus Target")    
    
    # Check if a button is clicked
    button_clicked = button1 or button2 or button3 or button4
    
    #Set button1 as default when opening the page
    if not button_clicked or button1:
        # Code to display the chart with Plotly
        count_deposit = df['deposit'].value_counts()
        color_sequence = ['#FACA5E', '#5242EA']
        # pie chart
        pie_chart = go.Pie(
            labels=count_deposit.index,
            values=count_deposit.values,
            marker=dict(colors=color_sequence),
            pull=[0.05, 0]
        )
        # bar chart
        bar_chart = go.Bar(
            x=count_deposit.index,
            y=count_deposit.values,
            text=count_deposit.values,
            textposition='auto',
            marker=dict(color=color_sequence),
            showlegend=False
        )
        # figure with two subplots
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "domain"}, {"type": "xy"}]],
            subplot_titles=("Distribution", "Number of deposits")
        )
        # Add pie chart and bar chart to the figure
        fig.add_trace(pie_chart, row=1, col=1)
        fig.add_trace(bar_chart, row=1, col=2)
        # Update
        fig.update_layout(
            title_text="<b>Analysis of the target variable: term deposit or not</b>",
            legend_title="<b>Deposit</b>"
        )
        # Display with Streamlit
        st.plotly_chart(fig)
        st.subheader("Observation")
        st.markdown("""
        The distribution between customers who have subscribed to a term deposit and those who have not is relatively balanced,
        with a difference of 5.2 points.
        However, there are slightly more people who have not taken out a deposit (52.6%)
        compared to those who have (47.4%).
        """)

    if button2:
        # Code to display histograms of numerical variables
        num_columns = ['balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
        # Create subplots
        fig = make_subplots(rows=2, cols=3, subplot_titles=num_columns)
        # Subplot position
        row = 1
        col = 1
        # Create histograms for each numerical variable
        for num_column in num_columns:
            fig.add_trace(
                go.Histogram(
                    x=df[num_column],
                    marker_color='#56CEB2',
                    opacity=0.6,
                    marker_line_width=0.5,
                    showlegend=False,
                    name=num_column
                ),
                row=row,
                col=col
            )
            fig.update_xaxes(title_text=num_column, row=row, col=col)
            fig.update_yaxes(title_text='Count', row=row, col=col)

            col += 1
            if col > 3:
                row += 1
                col = 1
        # Update chart layout
        fig.update_layout(
            height=800,
            width=1000,
            title_text="<b>Histograms of numerical variables</b>"
        )
        # Display the chart with Streamlit
        st.plotly_chart(fig)
        st.subheader("Observation")
        st.markdown("""
        - **Average bank account balance (balance)**: High concentration of data around 0. Presence of negative values ​​and extreme values.
        - **Contact day (days)**: the telemarketing campaign seems to take place every day of the month, with a notable drop on average on the 10th of the month and between the 22nd and 27th of the month. It should be noted that this variable is smoothed over all months of several years, with the absence of year information, nor that of the day of the week, not allowing us to deduce major trends from this variable.
        - **Contact duration (duration)**: expressed in seconds, presence of extreme values.
        - **Number of contacts in the campaign (campaign)**: presence of extreme values.
        - **Number of days since the previous contact (pdays)**: high presence of negative values, asymmetric distribution, and many extreme values.
        - **Number of previous contacts (previous)**: Very high concentration around 0 which means no previous contacts and presence of extreme values.
        """)
        st.divider()
        # Convert the target variable 'deposit' to numeric
        df['deposit_num'] = df['deposit'].apply(lambda x: 1 if x == 'yes' else 0)
        # Select numerical variables
        var_num_cible = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous', 'deposit_num']
        # Calculate the correlation matrix
        corr_matrix_cible = df[var_num_cible].corr()
        # Create the heatmap with Plotly
        heatmap_fig = px.imshow(corr_matrix_cible, text_auto=True, aspect="auto", color_continuous_scale='Turbo')
        # Update layout
        heatmap_fig.update_layout(
            title="<b>Heatmap of Numerical Variables with the target variable deposit</b>",
            xaxis_title="Variables",
            yaxis_title="Variables"
        )
        # Display the heatmap with Streamlit
        st.plotly_chart(heatmap_fig)
        st.subheader("Observation")
        st.markdown("""
        - In this correlation graph, we note a link between the variables pdays and previous;
        which seems consistent since pdays represents the number of days since the last customer contact and previous represents the number of contacts preceding this campaign.
        - The variable duration - duration of customer contact during the campaign - seems to influence the target variable deposit. We will study more specifically this variable expressed in seconds, and presenting extreme values.
        - To a much lesser extent, the variables pdays, previous and balance seem to slightly influence the target variable deposit.
        """)

    if button3:
        # Categories to display
        cat_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
        # Color palette
        color_pal4 = ['#56CEB2', '#28DCE0', '#57CF8A', '#579DCF']
        # Create subplots
        fig = make_subplots(rows=3, cols=3, subplot_titles=cat_columns)
        # Function to apply colors
        counter = 0
        for cat_column in cat_columns:
            value_counts = df[cat_column].value_counts()
            x_pos = np.arange(0, len(value_counts))
            # Shuffle the colors of the palette randomly
            random_colors = color_pal4.copy()
            random.shuffle(random_colors)
            # Apply shuffled colors to category bars
            colors = [random_colors[i % len(random_colors)] for i in range(len(value_counts))]
            trace_x = counter // 3 + 1
            trace_y = counter % 3 + 1
            # Add the bar
            fig.add_trace(
                go.Bar(
                    x=x_pos,
                    y=value_counts.values,
                    text=value_counts.values,
                    textposition='auto',
                    hoverinfo='text+x',
                    name=cat_column,
                    marker_color=colors,
                    opacity=0.8,
                    showlegend=False,
                ),
                row=trace_x,
                col=trace_y
            )
            # Format the x-axis
            fig.update_xaxes(
                tickvals=x_pos,
                ticktext=value_counts.index,
                row=trace_x,
                col=trace_y
            )
            # Rotate x-axis labels
            fig.update_xaxes(tickangle=45, row=trace_x, col=trace_y)
            counter += 1
        # Update chart layout
        fig.update_layout(
            height=800,
            width=1000,
            title_text="<b>Distribution of modalities of categorical variables</b>",
        )
        # Display the chart with Streamlit
        st.plotly_chart(fig)
        st.markdown("**Observation**")
        st.markdown("""
        - Job: The most common professions are 'management', 'blue-collar' (workers) and 'technician'”.
        - Marital status: The majority of customers are 'married'.
        - Education level: The 'secondary' category is the most common among those who have subscribed to the term deposit product.
        - Default: Very low proportion of customers in default.
        - Housing loan: fairly balanced between customers with a mortgage and not.
        - Personal loan: Very low proportion of customers with a personal loan.
        - Contact type: Contact by mobile is the most frequent.
        - Contact month: The months of May, June, July, and August are the months with the most contacts for this campaign.
        - Previous campaign result (poutcome): A good part of the results of the previous campaign is unknown.
        """)

    if button4:
        # Submenu to navigate different sections of the page
        st.divider()
        st.markdown("""
        ### Analysis in 4 axes:
        - [The customer profile](#the-customer-profile)
        - [The banking profile](#the-banking-profile)
        - [Analysis of customer contacts during the telemarketing campaign](#analysis-of-customer-contacts-during-the-telemarketing-campaign)
        - [Analysis of the previous campaign and its influence on the current campaign](#analysis-of-the-previous-campaign-and-its-influence-on-the-current-campaign)
        """)

        # Section 1: The customer profile
        st.markdown("""
        <a id="the-customer-profile"></a>
        ### The customer profile
        """, unsafe_allow_html=True)

        # Charts for customer profile

        # Define specific colors for each category
        color_sequence = ['#5242EA', '#FACA5E']

        # 1st chart: Age distribution versus deposit
        fig1 = px.box(df, x='age', y='deposit', points='all',
                    color='deposit',
                    title="Age distribution versus deposit",
                    color_discrete_sequence=color_sequence,
                    labels={'deposit': 'Deposit Status'},
                    category_orders={"deposit": ["yes", "no"]} #"yes" is before "no"
                 )

        # 2nd chart: Distribution of deposits by age
        count_deposit = df.groupby(['age', 'deposit']).size().reset_index(name='count')
        fig2 = px.bar(count_deposit, x='age', y='count', color='deposit',
                    barmode='group',
                    title="Distribution of deposits by age",
                    labels={'age': 'Age', 'count': 'Number of deposits', 'deposit': 'Deposit Status'},
                    category_orders={"deposit": ["yes", "no"]}, # "yes" is before "no"
                    color_discrete_sequence=color_sequence
                )

        # Assemble charts
        fig = make_subplots(rows=1, cols=2, subplot_titles=[
        "Age distribution versus deposit",
        "Distribution of deposits according to age"
        ])

        # Add fig1 without legend to avoid duplicates
        for trace in fig1['data']:
            trace.showlegend = False
            fig.add_trace(trace, row=1, col=1)

        # Add fig2 with legend
        for trace in fig2['data']:
            fig.add_trace(trace, row=1, col=2)

        # Update layout
        fig.update_layout(
        height=500,
        width=1500,
        title_text="<b>Age analysis based on deposit",
        showlegend=True,
        bargap=0.1,
        legend=dict(
            title="Deposit")
        )

        fig.update_xaxes(title_text='Customer Age', row=1, col=1)
        fig.update_yaxes(title_text='Deposit', row=1, col=1)

        fig.update_xaxes(title_text='Customer Age', row=1, col=2)
        fig.update_yaxes(title_text='Number of repositories', row=1, col=2)

        # Display chart
        st.plotly_chart(fig)

        # Explanatory text
        st.markdown("**Observation**")
        st.markdown("""
        - We can see that customers who have subscribed to term deposits are on average older than those who have not subscribed (78 years old compared to 70 years old).
        - The following scatterplot highlights that those who have not subscribed are more dispersed after 60 years old.
        - We also note the presence of many extreme values ​​(outliers).
        - Finally, there is clearly a higher proportion of term deposits among those under 30 and those over 60.

        For the rest of the analysis, we have chosen to discretize the 'age' variable by age group to attenuate
        the role of extreme values ​​and to then display several graphs by category.

        """)
        st.divider()
        # 2nd chart: Age discretization
        df['age_cat'] = pd.cut(df.age, bins=[18,29,40,50,60,96], labels=['18-29','30-40','40-50','50-60','60+'])
        df['age_cat'].value_counts()

        # 1ST CHART AGE
        counts_age = df.groupby(['age_cat', 'deposit']).size().unstack()
        total_counts_age = counts_age.sum(axis=1)
        percent_yes_age = (counts_age['yes'] / total_counts_age * 100).round(2)
        percent_no_age = (counts_age['no'] / total_counts_age * 100).round(2)
        df_plot_age = pd.melt(counts_age.reset_index(), id_vars=['age_cat'], value_vars=['yes', 'no'],
                      var_name='deposit', value_name='count')
        df_plot_age['percent'] = percent_yes_age.tolist() + percent_no_age.tolist()

        fig_age = px.bar(df_plot_age, x='age_cat', y='count', color='deposit', barmode='group',
            title="Distribution of deposits by age group",
            labels={'age_cat': 'Age', 'count': 'Number of deposits', 'deposit': 'Deposit'},
            category_orders={"age_cat": ['18-29','30-40','40-50','50-60','Over 60']},
            text=df_plot_age['percent'].apply(lambda x: f"{x:.2f}%"), # add % sign to percentages
            color_discrete_sequence=['#5242EA', '#FACA5E'], # set correct colors
            hover_data={'count': True, 'percent': ':.2f%'} # show details on hover
        )

        # Update layout
        fig_age.update_layout(yaxis_title="Number of repositories",
        legend_title_text='Repository status',
                      xaxis_tickangle=30)

        # 2ND CHART JOB
        counts_job = df.groupby(['job', 'deposit']).size().unstack()
        job_order = df.groupby('job')['deposit'].count().reset_index(name='total_deposits')
        job_order = job_order.sort_values(by='total_deposits', ascending=False)['job']
        job_order = job_order.tolist()
        total_counts_job = counts_job.sum(axis=1)
        percent_yes_job = (counts_job['yes'] / total_counts_job * 100).round(2)
        percent_no_job = (counts_job['no'] / total_counts_job * 100).round(2)
        df_plot_job = pd.melt(counts_job.reset_index(), id_vars=['job'], value_vars=['yes', 'no'],
                      var_name='deposit', value_name='count')
        df_plot_job['percent'] = percent_yes_job.tolist() + percent_no_job.tolist()

        fig_job = px.bar(df_plot_job, x='job', y='count', color='deposit', barmode='group',
            title="Distribution of deposits according to the type of employment",
            labels={'job': 'Job', 'count': 'Number of deposits', 'deposit': 'Deposit'},
            category_orders={'job': job_order},
            color_discrete_sequence=['#5242EA', '#FACA5E'],  # configure correct colors
            hover_data={'count': True}  # show details on hover
        )

        # Update layout
        fig_job.update_layout(yaxis_title="Number of repositories",
        legend_title_text='Repository status',
                      xaxis_tickangle=30,
                      bargap=0.1)

        # 3RD CHART MARITAL
        marital_order = ['married', 'single', 'divorced']
        counts_marital = df.groupby(['marital', 'deposit']).size().unstack()
        total_counts_marital = counts_marital.sum(axis=1)
        percent_yes_marital = (counts_marital['yes'] / total_counts_marital * 100).round(2)
        percent_no_marital = (counts_marital['no'] / total_counts_marital * 100).round(2)
        df_plot_marital = pd.melt(counts_marital.reset_index(), id_vars=['marital'], value_vars=['yes', 'no'],
                      var_name='deposit', value_name='count')
        df_plot_marital['percent'] = percent_yes_marital.tolist() + percent_no_marital.tolist()

        fig_marital = px.bar(df_plot_marital, x='marital', y='count', color='deposit', barmode='stack',
            title="Distribution of deposits by marital status",
            labels={'marital': 'Marital status', 'count': 'Number of deposits', 'deposit': 'Deposit'},
            category_orders={'marital': marital_order},
            text=df_plot_marital['percent'].apply(lambda x: f"{x:.2f}%"),  # add % sign to percentages
            color_discrete_sequence=['#5242EA', '#FACA5E'],  # configure correct colors
            hover_data={'count': True, 'percent': ':.2f%'}  # show details on hover
        )

        # Update layout
        fig_marital.update_layout(yaxis_title="Number of deposits",
            legend_title_text='Deposit status',
            xaxis_tickangle=30)

        # 4TH CHART EDUCATION
        education_order = df.groupby('education')['deposit'].count().reset_index(name='total_deposits')
        education_order = education_order.sort_values(by='total_deposits', ascending=False)['education']
        education_order = education_order.tolist()
        counts_education = df.groupby(['education', 'deposit']).size().unstack()
        total_counts_education = counts_education.sum(axis=1)
        percent_yes_education = (counts_education['yes'] / total_counts_education * 100).round(2)
        percent_no_education = (counts_education['no'] / total_counts_education * 100).round(2)
        df_plot_education = pd.melt(counts_education.reset_index(), id_vars=['education'], value_vars=['yes', 'no'],
                      var_name='deposit', value_name='count')
        df_plot_education['percent'] = percent_yes_education.tolist() + percent_no_education.tolist()

        fig_education = px.bar(df_plot_education, x='education', y='count', color='deposit', barmode='stack',
            title="Distribution of deposits according to the level of education",
            labels={'education': "Level of education", 'count': 'Number of deposits', 'deposit': 'Deposit'},
            category_orders={'education': education_order},
            text=df_plot_education['percent'].apply(lambda x: f"{x:.2f}%"),  # add % sign to percentages
            color_discrete_sequence=['#5242EA', '#FACA5E'],  # configure correct colors
            hover_data={'count': True, 'percent': ':.2f%'}  # show details on hover
        )

        # Update layout
        fig_education.update_layout(yaxis_title="Number of repositories",
            legend_title_text='Repository status',
            xaxis_tickangle=30)

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Age distribution by deposit",
                "Job distribution by deposit",
                "Marital status distribution by deposit",
                "Education level distribution by deposit"
            )
        )

        # Add fig_age
        for trace in fig_age['data']:
            trace.showlegend = False
            fig.add_trace(trace, row=1, col=1)

        # Add fig_job
        for trace in fig_job['data']:
            trace.showlegend = False
            fig.add_trace(trace, row=1, col=2)

        # Add fig_marital
        for trace in fig_marital['data']:
            trace.showlegend = False
            fig.add_trace(trace, row=2, col=1)

        # Add fig_education
        for trace in fig_education['data']:
            fig.add_trace(trace, row=2, col=2)

        # Update axes with specified orders
        fig.update_xaxes(categoryorder='array', categoryarray=job_order, row=1, col=2)
        fig.update_xaxes(categoryorder='array', categoryarray=marital_order, row=2, col=1)

        # Update layout
        fig.update_layout(
        height=900,
        width=1200,
        title_text="<b>Customer profile analysis based on deposit results",
        legend_title="Deposit"
        )

        fig.update_yaxes(title_text='Number of filings', row=1, col=1)
        fig.update_yaxes(title_text='Number of filings', row=1, col=2)
        fig.update_xaxes(title_text='Marital Status' , row=2, col=1)
        fig.update_yaxes(title_text='Number of deposits', row=2, col=1)
        fig.update_xaxes(title_text="Level of studies", row=2, col=2)
        fig.update_yaxes(title_text='Number of deposits', row=2, col=2)

        # Display charts
        st.plotly_chart(fig)
        st.markdown("**Observation**")
        st.markdown("""
        - **Age**: A significant trend is emerging among young people and the elderly to subscribe to term deposits, with nearly 60% of those under 30 and about 82% of those over 60 having opted for this option.
        - **Employment**: While managers, workers, technicians and administrative staff represent a substantial portion of the bank's customers, retirees, students, the unemployed and managers are more inclined to subscribe to term deposits.
        - **Marital status**: Although married customers constitute a significant proportion of the customer base, single people show a greater propensity to subscribe to the deposit, with more than 54% of them having opted for this option.
        - **Educational level**: Although the majority of customers have a secondary education level, a higher proportion of deposit subscribers is observed among those with a higher education level (tertiary), reaching 54%. In contrast, lower levels of education are associated with lower subscription rates.
        """)
        st.divider()
        # Section 2: The banking profile
        st.markdown("""
        <a id="le-profil-bancaire"></a>
        ### The banking profile
        """, unsafe_allow_html=True)

        ## 1ST CHART DEFAULT
        # Calculate counts for each default and deposit category
        counts_default = df.groupby(['default', 'deposit']).size().unstack()
        # Calculate percentages
        total_counts_default = counts_default.sum(axis=1)
        percent_yes_default = (counts_default['yes'] / total_counts_default * 100).round(2)
        percent_no_default = (counts_default['no'] / total_counts_default * 100).round(2)
        # Transform data for Plotly Express
        df_plot_default = pd.melt(counts_default.reset_index(), id_vars=['default'], value_vars=['yes', 'no'],var_name='deposit', value_name='count')

        # Add calculated percentages
        df_plot_default['percent'] = percent_yes_default.tolist() + percent_no_default.tolist()

        # Create chart with Plotly Express
        fig_default = px.bar(df_plot_default, x='default', y='count', color='deposit', barmode='stack',
            title="Distribution of deposits according to payment default",
            labels={'default': 'Payment default', 'count': 'Number of deposits', 'deposit': 'Deposit'},
             text=df_plot_default['percent'].apply(lambda x: f"{x:.2f}%"),  # add % sign to percentages
             color_discrete_sequence=['#5242EA', '#FACA5E'],  # configure correct colors
             hover_data={'count': True, 'percent': ':.2f%'}  # show details on hover
             )

        # Update layout
        fig_default.update_layout(yaxis_title="Nombre de dépôts",
                  legend_title_text='Statut du dépôt')


        # 2ND CHART LOAN
        # Calculate counts for each loan and deposit category
        counts_loan = df.groupby(['loan', 'deposit']).size().unstack()
        # Calculate percentages
        total_counts_loan = counts_loan.sum(axis=1)
        percent_yes_loan = (counts_loan['yes'] / total_counts_loan * 100).round(2)
        percent_no_loan = (counts_loan['no'] / total_counts_loan * 100).round(2)
        # Transform data for Plotly Express
        df_plot_loan = pd.melt(counts_loan.reset_index(), id_vars=['loan'], value_vars=['yes', 'no'],
                  var_name='deposit', value_name='count')

        # Add calculated percentages
        df_plot_loan['percent'] = percent_yes_loan.tolist() + percent_no_loan.tolist()

        # Create chart with Plotly Express
        fig_loan = px.bar(df_plot_loan, x='loan', y='count', color='deposit', barmode='stack',
            title="Distribution of deposits according to personal loan",
            labels={'loan': 'Personal loan', 'count': 'Number of deposits', 'deposit': 'Deposit'},
            text=df_plot_loan['percent'].apply(lambda x: f"{x:.2f}%"),  # add % sign to percentages
            color_discrete_sequence=['#5242EA', '#FACA5E'],  # configure correct colors
            hover_data={'count': True, 'percent': ':.2f%'}  # show details on hover
            )

        # Update layout
        fig_loan.update_layout(yaxis_title="Number of deposits",
        legend_title_text='Deposit status')


        # 3RD CHART HOUSING
        # Calculate counts for each housing and deposit category
        counts_housing = df.groupby(['housing', 'deposit']).size().unstack()
        # Calculate percentages
        total_counts_housing = counts_housing.sum(axis=1)
        percent_yes_housing = (counts_housing['yes'] / total_counts_housing * 100).round(2)
        percent_no_housing = (counts_housing['no'] / total_counts_housing * 100).round(2)
        # Transform data for Plotly Express
        df_plot_housing = pd.melt(counts_housing.reset_index(), id_vars=['housing'], value_vars=['yes', 'no'],
                  var_name='deposit', value_name='count')

        # Add calculated percentages
        df_plot_housing['percent'] = percent_yes_housing.tolist() + percent_no_housing.tolist()

        # Create chart with Plotly Express
        fig_housing = px.bar(df_plot_housing, x='housing', y='count', color='deposit', barmode='stack',
            title="Distribution of deposits according to the Mortgage",
            labels={'housing': 'Mortgage', 'count': 'Number of deposits', 'deposit': 'Deposit'},
            text=df_plot_housing['percent'].apply(lambda x: f"{x:.2f}%"),  # add % sign to percentages
            color_discrete_sequence=['#5242EA', '#FACA5E'],  # configure correct colors
            hover_data={'count': True, 'percent': ':.2f%'}  # show details on hover
            )

        # Update layout
        fig_housing.update_layout(yaxis_title="Number of repositories",
            legend_title_text='Repository status')


        # 4TH CHART BALANCE
        # Distribution of balance versus deposit
        fig_balance = px.box(df, x='deposit', y='balance',
            color='deposit',
            title="Distribution of average account balance",
            color_discrete_sequence=['#5242EA', '#FACA5E'], # set the correct colors,
            labels={'deposit': 'Deposit Status'},
            category_orders={"deposit": ["yes", "no"]}  #"yes" is before "no"
            )

        ## CREATE SUBPLOTS
        fig = make_subplots(
            rows=1, cols=4,
            subplot_titles=(
                "Default",
                "Personal loan",
                "Home loan",
                "Average account balance"
            )
        )

        # Add fig_default
        for trace in fig_default['data']:
            trace.showlegend = False
            fig.add_trace(trace, row=1, col=1)
        # Add fig_loan
        for trace in fig_loan['data']:
            trace.showlegend = False
            fig.add_trace(trace, row=1, col=2)
        # Add fig_housing
        for trace in fig_housing['data']:
            trace.showlegend = False
            fig.add_trace(trace, row=1, col=3)
        # Add fig_balance
        for trace in fig_balance['data']:
            fig.add_trace(trace, row=1, col=4)

        # Update layout
        fig.update_layout(
            height=500,
            width=1400,
            title_text="<b>Banking profile analysis according to deposit results",
            legend_title= "Deposit"
            )
        
        fig.update_xaxes(title_text='default (yes, no)', row=1, col=1)
        fig.update_yaxes(title_text='Number of deposits', row=1, col=1)
        
        fig.update_xaxes(title_text= 'loan (yes, no)', row=1, col=2)
        fig.update_xaxes(title_text='housing (yes, no)', row=1, col=3)
        fig.update_xaxes(title_text='deposit (yes , no)', row=1, col=4)
        
        fig.update_xaxes(title_text='deposit (yes, no)', row=1, col=4)
        fig.update_yaxes(title_text='balance', row=1, col=4)

        ## DISPLAY FIGURE
        st.plotly_chart(fig)
        st.markdown("**Observation**")
        st.markdown("""
        - **Default payment (default)**: Very low proportion of customers in default of payment in the dataset.
        On the other hand, we can clearly see that those in default of payment are much less inclined to subscribe to the deposit (69% have not subscribed, while 52% of other customers have subscribed, which corresponds to the general average of subscribers)
        - **Personal loan (loan)**: low proportion of the bank's customers with a consumer credit or personal loan. However, we can see that those with an ongoing personal loan have subscribed less to the deposit (69% have not subscribed, while 50.4% of other customers have subscribed, which is close to the general average of subscribers)
        - **Housing loan**: We can clearly see here that customers without a mortgage are a majority to subscribe to the deposit (57% of them) and conversely the customers with a current mortgage are less likely to subscribe (63% of them have not subscribed)
        - **Average account balance (balance)**: the data is extensive with many extreme values. We note a median balance of €733 for customers who have subscribed and a lower median balance of €414 for customers who have not subscribed.
        """)
        st.divider()



        # Section 3: Analysis of customer contacts during the telemarketing campaign
        st.markdown("""
        <a id="analysis-of-customer-contacts-during-the-telemarketing-campaign"></a>
        ### Analysis of customer contacts during the telemarketing campaign
        """, unsafe_allow_html=True)
        # 1. Chart Contact
        # Calculate counts for each contact and deposit category
        counts_contact = df.groupby(['contact', 'deposit']).size().unstack(fill_value=0)
        # Calculate percentages
        total_counts_contact = counts_contact.sum(axis=1)
        percent_yes_contact = (counts_contact['yes'] / total_counts_contact * 100).round(2)
        percent_no_contact = (counts_contact['no'] / total_counts_contact * 100).round(2)
        # Transform data for Plotly Express
        df_plot_contact = pd.melt(counts_contact.reset_index(), id_vars=['contact'], value_vars=['yes', 'no'],
                  var_name='deposit', value_name='count')
        # Add calculated percentages
        df_plot_contact['percent'] = percent_yes_contact.tolist() + percent_no_contact.tolist()

        # Create chart
        fig_contact = px.bar(df_plot_contact, x='contact', y='count', color='deposit', barmode='group',
            title="Customer Contact Mode and Deposit Results",
            labels={'contact': 'Customer Contact Mode', 'count': 'Number of Deposits', 'deposit': 'Deposit'},
             color_discrete_sequence=['#5242EA', '#FACA5E'],
             hover_data={'count': True, 'percent': ':.2f%'}
             )

        # Update layout
        fig_contact.update_layout(yaxis_title="Number of repositories",
            legend_title_text='Repository status')


        # 2. Chart Duration
        fig_duration = px.box(df,
           x='duration',  # Change 'duration_minutes' to 'duration'
           y='deposit',
           color='deposit',
           color_discrete_sequence=['#5242EA', '#FACA5E'],
            title='<b>Influence of contact duration on campaign results')

        # 3. Chart Month
        # Calculate the total number of deposits for each month
        month_order = df.groupby('month')['deposit'].count().reset_index(name='total_deposits')
        month_order = month_order.sort_values(by='total_deposits', ascending=False)['month']

        # Convert to list for use in category_orders
        month_order = month_order.tolist()

        # Create histogram
        fig_month = px.histogram(df, x='month', color='deposit', barmode='group',
                    title="Deposit distribution by month",
                    labels={'month': 'Contact month', 'count': 'Number of deposits', 'deposit': 'Deposit'},
                    category_orders={"month": month_order},
                    color_discrete_sequence=['#5242EA', '#FACA5E'])

        # Update layout
        fig_month.update_layout(yaxis_title="Number of repositories",
            legend_title_text='Repository status',
                  xaxis_tickangle=30,
                  bargap=0.1)

        # 4. Chart M Contact
        # Group by month and aggregate counts
        data_month = df.groupby('month').agg(
            campaign_count=('campaign', 'sum'),
            deposit_yes_count=('deposit', lambda x: (x == 'yes').sum()),
            deposit_no_count=('deposit', lambda x: (x == 'no').sum())
        ).reset_index()
        # Add a new column with manual values
        manual_values = [4, 8, 12, 2, 1, 7, 6, 3, 5, 11, 10, 9]
        # Assign manual values to the 'manual_order' column
        data_month['manual_order'] = manual_values
        # Sort the DataFrame by the 'manual_order' column
        data_month_sorted = data_month.sort_values(by='manual_order').reset_index(drop=True)
        # Create chart
        fig_m_contact = px.line()
        # Add curves to the chart
        fig_m_contact.add_scatter(x=data_month_sorted['month'], y=data_month_sorted['campaign_count'], mode='lines', name='Nombre de contact', line=dict(color='#034F84', dash='dash'))
        fig_m_contact.add_scatter(x=data_month_sorted['month'], y=data_month_sorted['deposit_yes_count'], mode='lines', name='Dépôts Yes', line=dict(color='#5242EA'))
        fig_m_contact.add_scatter(x=data_month_sorted['month'], y=data_month_sorted['deposit_no_count'], mode='lines', name='Dépôts No', line=dict(color='#FACA5E'))
        fig_m_contact.update_layout(title='Number of contacts and deposits per month')
        # Adding axes
        fig_m_contact.update_xaxes(title_text='Month')
        fig_m_contact.update_yaxes(title_text='Number of contacts')

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Contact method",
                "Contact duration",
                "Distribution by month",
                "Number of contacts and deposits per month"
            )
        )

        # Add fig_contact
        for trace in fig_contact['data']:
            trace.showlegend = False
            fig.add_trace(trace, row=1, col=1)

        # Add fig_duration
        for trace in fig_duration['data']:
            trace.showlegend = False
            fig.add_trace(trace, row=1, col=2)

        # Add fig_month
        for trace in fig_month['data']:
            fig.add_trace(trace, row=2, col=1)

        # Add fig_m_contact
        for trace in fig_m_contact['data']:
            fig.add_trace(trace, row=2, col=2)

        # Update layout
        fig.update_layout(
            height=600,
            width=1400,
            title_text="<b>Campaign analysis: contact type, number of contacts, period and duration",
            legend_title="Deposit"
            )

        fig.update_xaxes(title_text='Contact method', row=1, col=1)
        fig.update_yaxes(title_text='Number of deposits', row=1, col=1)

        fig.update_xaxes(title_text='Contact duration in minutes', row=1, col=2)
        fig.update_yaxes(title_text='Number of deposits', row=1, col=2)
        
        fig.update_xaxes(title_text='Month of contact', row=2, col= 1)
        fig.update_yaxes(title_text='Number of deposits', row=2, col=1)

        fig.update_xaxes(title_text='Month of contact', row=2, col=2)
        fig.update_yaxes(title_text='Number of deposits', row=2, col=2)

        # Display the figure in Streamlit
        st.plotly_chart(fig)
        st.markdown("**Observation**")
        st.markdown("""
        - **Type of contact**: A majority of contacts for this telemarketing campaign were made on mobile (cellular) and we note a higher proportion of subscriptions for customers who were contacted by this means (more than 54%).
        Nevertheless, a significant part of the type of contact is unknown.
        - **Duration of contact**: It appears that the duration of customer contact influences the result of the term deposit: the longer the contact, the more customers tend to subscribe.
        - **Month of contact**: It is interesting to note on the 2 graphs at the bottom, that if the months of May, June, July, August are the months of highest activity of the campaign,
        these are not the months where the share of subscriptions is the most important. We note in fact a **higher proportion of customers making a deposit during the months of February, March, April, September, October and December**.
        The obvious example is the month of May which seems to be the month with the highest activity of the campaign and for which the deposit share is lower.
        """)
        st.divider()
        # Section 4: Analysis of the previous campaign and its influence on the current campaign
        st.markdown("""
        <a id="analysis-of-the-previous-campaign-and-its-influence-on-the-current-campaign"></a>
        ### Analysis of the previous campaign and its influence on the current campaign
        """, unsafe_allow_html=True)

        # 1. Chart Previous Contacts or Not Contacted
        # Divide into two groups
        df['group'] = df['previous'].apply(lambda x: 'not contacted' if x == 0 else 'contacted')

        # Count deposit values for each group
        count_df = df.groupby(['group', 'deposit']).size().reset_index(name='count')

        # Calculate percentages
        total_counts = count_df.groupby('group')['count'].transform('sum')
        count_df['percentage'] = (count_df['count'] / total_counts * 100).round(2)

        # Create bar plot with Plotly Express
        fig_previous = px.bar(
            count_df,
            x='group',
            y='count',
            color='deposit',
            text=count_df['percentage'].astype(str) + '%',
            color_discrete_sequence=['#5242EA', '#FACA5E'],
        )

        # 2. Chart Number of Days Since Last Contact (pdays)
        # Filter data to exclude 'pdays' values equal to -1
        df_filtered = df[df['pdays'] != -1]

        # Create box plot
        fig_pdays = px.box(df_filtered,
             x='deposit',
             y='pdays',
             color='deposit',
             color_discrete_sequence=['#5242EA', '#FACA5E'],
             )

        # 3. Chart Results of the Previous Campaign (poutcome)
        # Calculate counts for each poutcome and deposit category
        counts_poutcome = df.groupby(['poutcome', 'deposit']).size().unstack()
        # Calculate percentages
        total_counts_poutcome = counts_poutcome.sum(axis=1)
        percent_yes_poutcome = (counts_poutcome['yes'] / total_counts_poutcome * 100).round(2)
        percent_no_poutcome = (counts_poutcome['no'] / total_counts_poutcome * 100).round(2)
        # Transform data for Plotly Express
        df_plot_poutcome = pd.melt(counts_poutcome.reset_index(), id_vars=['poutcome'], value_vars=['yes', 'no'],
                  var_name='deposit', value_name='count')

        # Add calculated percentages
        df_plot_poutcome['percent'] = percent_yes_poutcome.tolist() + percent_no_poutcome.tolist()

        # Create chart with Plotly Express
        fig_poutcome = px.bar(df_plot_poutcome, x='poutcome', y='count', color='deposit', barmode='group',
             text=df_plot_poutcome['percent'].apply(lambda x: f"{x:.2f}%"),  # add % sign to percentages
             color_discrete_sequence=['#5242EA', '#FACA5E'],  # configure correct colors
             hover_data={'count': True, 'percent': ':.2f%'}  # show details on hover
             )

        # Create subplots
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=(
            "Previous contacts or not",
            "Number of days since last contact",
            "Success of the previous campaign"
            )
        )

        # Add fig_previous
        for trace in fig_previous['data']:
            trace.showlegend = False
            fig.add_trace(trace, row=1, col=1)

        # Add fig_pdays
        for trace in fig_pdays['data']:
            trace.showlegend = False
            fig.add_trace(trace, row=1, col=2)

        # Add fig_poutcome
        for trace in fig_poutcome['data']:
            fig.add_trace(trace, row=1, col=3)

        # Update layout
        fig.update_layout(
            height=600,
            width=1400,
            title_text="<b>Analysis of the previous campaign (pdays, previous, poutcome) and influence on the current campaign",
            legend_title="Repository"
            )

        fig.update_xaxes(title_text='Customer group if contacted previously or not (previous)', row=1, col=1)
        fig.update_yaxes(title_text='Number of deposits', row=1, col=1)

        fig.update_xaxes(title_text='Deposit', row=1, col=2)
        fig.update_yaxes(title_text='Number of days since last contact (pdays)', row=1, col=2)

        fig.update_xaxes(title_text='Results of previous campaign (poutcome)', row=1, col=3)
        fig.update_yaxes(title_text='Number of deposits', row=1, col=3)

        # Display the figure in Streamlit
        st.plotly_chart(fig)

        st.markdown("**Observation**")
        st.markdown("""
        - **Previous contacts**: A high proportion of customers have not been contacted previously.
        However, it is interesting to note that customers who have already been contacted before this campaign (during a previous campaign)
        are more likely to subscribe to the deposit: 67% of customers previously contacted subscribed to the deposit during this campaign, and conversely, those who have not been contacted previously were nearly 60% not to subscribe to the deposit during this campaign.
        This indicates that the multiplication of contacts on different campaigns can encourage customers and influence the success of a following campaign.
        - **Number of days since the last contact**: We can see that less time has passed since the last contact for customers subscribing to the deposit on this campaign.
        With a narrower range (between 94 and 246 days) than those who have not subscribed to the deposit (range 148 to 332 days).
        In addition, we can see many extreme values, especially among those who have subscribed to the deposit.
        - **Success of the previous campaign**: A large part of the data is unknown. It is still interesting to note that a customer who has subscribed to a product from a previous campaign (success), are very inclined to subscribe to the deposit of the current campaign: 91% of them have subscribed to the deposit.
        """)
        st.divider()
        # Section 5: Conclusion Analysis
        #st.markdown("""
        #<a id="conclusion_analyse"></a>
        ### Conclusion Analysis explanatory variables vs target variable
        #""", unsafe_allow_html=True)

# PAGE3 PREPROCESSING
def show_preprocessing_page():
    st.title("Pre-processing")
    # Load data
    df = load_data()
    # Use preprocessing functions
    #preprocessed_data = preprocess_data(df)
    #st.write(preprocessed_data)
    st.markdown("#### Data preparation before applying classification algorithms")
    st.header("Pre-processing approach")
    
    # TABS CONTAINERS PREPROCESSING
    tab1, tab2, tab3, tab4 = st.tabs(["Preprocessing", "Game Split", "Standardization and Encodings", "Pipeline"])
    with tab1:
        st.markdown("#### Pre-processing")
        #AGE
        st.markdown("""
        - **Discretize the variable 'age'** by age ranges to mitigate the role of extreme values.
        - **Replace the 'unknown' category of the variable 'education'** with the most frequent category.
        - **Divide the variable 'pdays'** into 2 distinct variables: pdays_contact and the variable pdays_days.
        - **pdays_contact**: value no for values ​​-1 of pdays, and value yes for the other values.
        - **pdays_days**: value 0 for values ​​-1 of pdays and values ​​of pdays >= 0.
        """)
        st.divider()
        st.markdown("**Discretization of the Variable Age**")
        st.code("""
        df['    age_cat'] = pd.cut(df.age, bins=[18,29,40,50,60,np.inf], labels = ['18-29','30-40','40-50','50-60','Over 60 years'],
                right=False) df = df.drop('age', axis = 1) pd.DataFrame(df['age_cat'].unique(), columns=["C age categories"]) 
                """, language='python')
        df = discretize_age(df)
        st.write("Age categories after discretization:")
        st.table(pd.DataFrame(df['age_cat'].unique(), columns=['Age categories']))

        st.write("") 
        
        #EDUCATION
        st.markdown("**Variable Education**")
        st.code("""
        most_frequent = df[df['education'] != 'unknown']['education'].mode()[0]
        df['education'] = df['education'].replace('unknown', most_frequent)
        df['education'].unique()
                """, language='python')
        st.write("Modalities of Education after replacing the value 'unknown' with the most frequent mode:")
        df = replace_unknown_education(df)
        st.write(df['education'].unique())
        st.write("")

        #PDAYS
        st.markdown("**Feature Engineering p-days**")
        st.code("""
        df['pdays_contact'] = df['pdays'].apply(lambda x: 'no' if x == -1 else 'yes')
        df['pdays_days'] = df['pdays'].apply(lambda x: 0 if x == -1 else x)
        df = df.drop('pdays', axis = 1)
        df.head()
                """, language='python')
        st.write("DataFrame df after Pdays Feature Engineering:")
        df = transform_pdays(df)
        st.write(df.head())

    with tab2:
        st.markdown("#### Features / Target Separation")
        st.markdown("Separate the explanatory variables of the target into two datasets.")
        st.code("""
            features = df.drop('deposit', axis = 1)
            target = df['deposit']
                """, language='python')
        features, target = separate_features_target(df, 'deposit')
        
        st.divider()
        
        st.markdown("#### Split Train / Test")
        # Path to the image
        image_path_traintest = "/tmp/files/images/traintest.jpg"
        #Display the image
        st.image(image_path_traintest)

        st.markdown("""
        - Separate the Features and Target sets:
        - a training set (X_train, y_train)
        - a test set (X_test, y_test)
        - the test part contains 25% of the initial dataset
                    """)
        st.code("""
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.25, random_state = 42)
        X_train.shape, X_test.shape
                """, language='python')
        X_train, X_test, y_train, y_test = split_train_test(features, target, test_size=0.25, random_state=42)
        # Display the shapes of X_train and X_test
        st.write(f"Dimensions de X_train: {X_train.shape}")
        st.write(f"Dimensions de X_test: {X_test.shape}")
    
    with tab3:
        st.markdown("#### Standardization and encodings")
        st.markdown("""
        - **LabelEncoder** of the target variable ‘deposit’.
        - **Cyclic encoding** of temporal variables: 'month', 'day'.
        - **RobustScaler** for numeric variables: 'balance', 'duration', 'campaign', 'previous’, ‘pdays_days’.
        - **LabelEncoder** of binary categorical variables (yes/no): 'default', 'housing', loan', ‘pdays_contact’.
        - **OneHotEncoder** of categorical variables: 'job', 'marital', 'contact', 'poutcome'.
        - **OrdinalEncoder** of ordinal variables: 'age', 'education'.
        """)

    with tab4:
        st.markdown("#### Pipeline Approach")
        # Path to the image
        image_path_pipeline = "/tmp/files/images/pipeline.jpg"
        #Display the image
        st.image(image_path_pipeline)
        st.markdown("Thanks to a pipeline, we were able to quickly generate 4 different pre-processings, then tested on different Machine Learning algorithms:")
        st.markdown("""
        - **Pre-processing 1**: without feature engineering of p_days, with discretized age, and a Robust Scaler encoding on the numeric variables.
        - **Pre-processing 2**: with the division of pdays into 2 variables, discretized age, Robust Scaler on the numeric variables.
        - **Pre-processing 3**: equivalent to the previous one but with a Standard Scaler on the numeric variables.
        - **Pre-processing 4**: with age without discretization and Standard Scaler on the numeric variables.
        """)
        st.markdown("""After running different classification algorithms,
        pre-processing 2 had better scores - even if very slightly higher.
        So this is the pipeline used below: """)
        st.divider()
        st.subheader("Pipeline")
        # Load data
        df = load_data()
        # Initial preprocessing
        X_train, X_test, y_train, y_test = preprocess_data(df)
        
        # Check and convert to DataFrame if they are Series
        if isinstance(y_train, pd.Series):
            y_train = y_train.to_frame(name='Deposit')
        if isinstance(y_test, pd.Series):
            y_test = y_test.to_frame(name='Deposit')

        # Convert X_train and X_test to DataFrames if necessary and sort by index
        X_train = pd.DataFrame(X_train).sort_index()
        X_test = pd.DataFrame(X_test).sort_index()
        y_train = y_train.sort_index()
        y_test = y_test.sort_index()
        
        # Dropdown menu to choose the display of results
        option = st.selectbox(
            "**Show datasets before or after pre-processing**",
            ["Before Pipeline", "After Pipeline"]
            )
            
        st.write("")
        st.write("")
            
        if option == "Before Pipeline":
            st.markdown("##### Before Pipeline")
            st.write("**Shapes of datasets:**")
            st.write(f"Shape of X_train before preprocessing: {X_train.shape}")
            st.write(f"Shape of X_test before preprocessing: {X_test.shape}")
            st.write(f"Shape of y_train before preprocessing: {y_train.shape}")
            st.write(f"Shape of y_test before preprocessing: {y_test.shape}")
            st.write("**X_train before preprocessing:**")
            st.dataframe(X_train.head())
            st.write("**X_test before preprocessing:**")
            st.dataframe(X_test.head())
            st.write("**y_train before preprocessing:**")
            st.dataframe(y_train.head())
            st.write("**y_test before preprocessing:**") 
            st.dataframe(y_test.head())

        elif option == "After Pipeline":
            # Apply preprocessing and transformation
            X_train_processed_df, X_test_processed_df, y_train_processed_df, y_test_processed_df = preprocess_and_transform(X_train, X_test, y_train, y_test)
            st.markdown("##### After Pipeline")
            st.write("**Shapes of processed datasets:**")
            st.write(f"Shape of X_train_processed after pipeline: {X_train_processed_df.shape}" )
            st.write(f"Shape of : {y_test_processed_df.shape}")
            st.write("**X_train_processed variables after Pipeline:**")
            st.write(X_train_processed_df.columns)
            st.write("**X_train_processed after pipeline:**")
            st.dataframe(X_train_processed_df.head())
            st.write("**X_test_processed after pipeline:**")
            st.dataframe(X_test_processed_df.head())
            st.write("**y_train_processed after pipeline:**")
            st.dataframe(y_train_processed_df.head())
            st.write("**y_test_processed after pipeline:**")
            st.dataframe(y_test_processed_df.head())


# PAGE4 MODELING
def show_modelling_page():
    initialize_results()
    st.title('Modeling')
    st.markdown("#### Training classification models")
    st.header("Modeling Approach") 
    # Insert containers separated into tabs:
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Steps", "11 models tested", "Training without Duration", "GridSearch","Final modeling"])    
    with tab1:
        st.markdown("##### Steps")
        st.markdown("""
        - First training of 11 classification models
        - Performance analysis
        - Choice to remove the duration variable
        - Training of the 11 models without duration
        - Selection of the 3 best performing models
        - Grid search for optimal hyperparameters (GridSearchCV)
        - Training of the 3 models with their best parameters
        - Performance analysis
        """)

    with tab2:
        st.markdown("##### 11 classification models tested")
        st.markdown("""
                    - Logistic Regression
                    - Random Forest
                    - Gradient Boosting
                    - Support Vector Machine
                    - K-Nearest Neighbors'
                    - AdaBoost:
                    - Naive Bayes
                    - MLP Classifier
                    - XGBoost
                    - LightGBM
                    - Decision Tree """)
        st.write("")     
        st.markdown("##### Performances")
        # Path to the image
        image_path_perf11 = "/tmp/files/images/perf11.png"
        #Display the image
        st.image(image_path_perf11)
        st.write("")
        st.markdown("##### Feature importances")
        st.markdown("For the 3 best performing models")
        # Create a dropdown menu to select the image to display
        option = st.selectbox(
        'Choose the model to display:',
        ['Random Forest', 'XGBoost', 'LightGBM']
       )
        # Define image paths
        image_paths = {
       'Random Forest': "/tmp/files/images/fi_rf.png",
       'XGBoost': "/tmp/files/images/fi_xg.png",
       'LightGBM': "/tmp/files/images/fi_gbm.png"
       }
        # Display the image based on the selection
        st.image(image_paths[option])
        st.markdown("##### Observation")
        st.markdown("""
        - The **'duration'** variable (duration of the call during the campaign) is the explanatory variable that emerges as the most important in predicting the results.
        This was demonstrated in the correlation matrix and now confirmed when scoring the trained models.
        Nevertheless, this variable is known a posteriori of the campaign since it is after the telemarketing call that the duration of the call is known.
        It therefore seems interesting to **test our models by removing this variable from our training and test sets**. We also tested by keeping the duration variable with only its median value and the result was not conclusive.
        - Note that the **'balance'** variable (average account balance) is the one that has the most importance after 'duration'.
        - Also note that the XGBoost model gives more importance to the variable **'poutcome_success'** (results of the previous campaign) in its predictions.
        """)

    with tab3:
        st.markdown("#### Training without Duration")
        st.write("")
        with st.expander('**Pre-processing Without Duration**'):
            # Load data
            df = load_data()
            
            # Preprocess data
            X_train, X_test, y_train, y_test = preprocess_data(df)
            X_train_processed_df, X_test_processed_df, y_train_processed_df, y_test_processed_df = preprocess_and_transform(X_train, X_test, y_train, y_test)
        
            # Call the function to remove the 'duration' column
            X_train_processed_df, X_test_processed_df = remove_duration(X_train_processed_df, X_test_processed_df)
    
            # Displaying the shapes of the datasets after removing 'duration'
            st.write("**Shape of X_train_processed without duration:**", X_train_processed_df.shape)
            st.write("**Shape of X_test_processed without duration:**", X_test_processed_df.shape)

            # Displaying the first rows of the datasets after removing 'duration'
            st.write("**X_train_processed without duration:**")
            st.dataframe(X_train_processed_df.head())
            st.write("**X_test_processed without duration:**")
            st.dataframe(X_test_processed_df.head())
       

        st.divider()     
        st.markdown("##### Performance")
        # Path to the image
        image_path_perfsd = "/tmp/files/images/perfsd.jpg"
        #Display the image
        st.image(image_path_perfsd)
        st.write("")     
        st.markdown("###### Observation")
        st.markdown("""
        Without the Duration variable, the scores and performances are significantly worse,
        which demonstrates the weight of this variable in predictive modeling.
        \nWe will try to optimize the models with a search grid for the best hyperparameters on the 3 best performing models using a GridSearchCV.
        \nLet's look just before the **Feature importances** without the preponderance of duration.
        """)
        st.divider()
        st.write("")
        st.markdown("##### Feature importances")
        st.markdown("For the 3 best performing models")
        # Create a dropdown menu to select the image to display
        option2 = st.selectbox(
        'Choose the model to display:',
       ['Random Forest', 'XGBoost', 'LightGBM'],
       key='feature_importances_selectbox'
       )
        # Define image paths
        image_paths2 = {
       'Random Forest': "/tmp/files/images/fisd_rf.png",
       'XGBoost': "/tmp/files/images/fisd_xg.png",
       'LightGBM': "/tmp/files/images/fisd_gbm.jpg"
       }
        # Display the image based on the selection
        st.image(image_paths2[option2])
        st.markdown("###### Observation")
        st.markdown("""
        - The **'Balance'** variable has taken the ascendancy of the most important feature for the LightGBM and Random Forest models.
        - The **"poutcome_success'** variable remains the most important for the XGBoost model.
        """)

    with tab4:
        st.markdown("#### GridSearchCV")
        # Create a dropdown menu to select the model
        model_option = st.selectbox(
        '**Show best model settings**:',
        ['Random Forest', 'XGBoost', 'LightGBM'],
        key='model_selectbox'
        )
        # Define texts for each model
        model_texts = {
        'Random Forest': """
        Random Forest is a decision tree model that uses multiple trees to improve accuracy and avoid overfitting.
        - Runtime: 4min 5s
        - Best parameters found for Random Forest: {'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}
        - Best score: 0.7336
        """,
        'XGBoost': """
        XGBoost is an implementation of gradient boosting that is efficient and performant for classification and regression tasks.
        - Runtime: 1min 29s
        - Best parameters found for XGBoost: {'colsample_bytree': 0.8, 'learning_rate': 0.01, 'max_depth': 7, 'n_estimators': 100, 'subsample': 0.8}
        - Best score: 0.7360
        """,
        'LightGBM': """
        LightGBM is a tree-based gradient boosting framework that is designed to be distributed and efficient with large data capacity.
        - Runtime: 35s
        - Best parameters found for LightGBM: {'colsample_bytree': 0.8, 'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 31, 'subsample': 0.8}
        - Best score: 0.7373
        """
        }
        # Display the text corresponding to the selected model
        st.markdown(model_texts[model_option])

    with tab5:
        st.subheader("Final Modeling")
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        button5 = col1.button("Random Forest Model")
        button6 = col2.button(" LightGBM Model")
        button7 = col3.button("XGBoost Model")
        button8 = col4.button("Performance Comparison")
    
        # Check if a button is clicked
        button_clicked = button5 or button6 or button7 or button8

        #Set button5 as default when opening the page
        if button5:

            with st.spinner("Analyzing...", show_time=True):
                time.sleep(2)
            st.markdown("#### Random Forest Model")
    
            # Define hyperparameters for the RandomForest model
            rf_params = {
                'max_depth': 10,
                'max_features': 'sqrt',
                'min_samples_leaf': 1,
                'min_samples_split': 2,
                'n_estimators': 100,
                'random_state': 42
            }

            # Use get_model to get the model
            model_rf = get_model('rf')
            if model_rf is not None:
                train_and_evaluate_and_save(RandomForestClassifier, rf_params, 'Random_Forest', model_folder+'results_rf')
            
        if button6:
            with st.spinner("Analyzing...", show_time=True):
                time.sleep(2)
            st.markdown("#### LightGBM Model")
            logger.debug('This is a debug message')

    
            # Define hyperparameters for the LightGBM model
            lgb_params = {
                'colsample_bytree': 0.8,
                'learning_rate': 0.01,
                'n_estimators': 100,
                'num_leaves': 31,
                'subsample': 0.8,
                'random_state': 42
            }
            # Use get_model to get the model
            model_lgb = get_model('lgb')
            if model_lgb is not None:
                train_and_evaluate_and_save(LGBMClassifier, lgb_params, 'LightGBM', model_folder+'results_lgb')

        if  button7:
            with st.spinner("Analyzing...", show_time=True):
                time.sleep(2)
            st.markdown("#### XGBoost Template")
    
            # Define hyperparameters for the XGBoost model
            xgb_params = {
                'colsample_bytree': 0.8,
                'learning_rate': 0.01,
                'max_depth': 7,
                'n_estimators': 100,
                'subsample': 0.8,
                'random_state': 42
            }

            # Use get_model to get the model
            model_xgb = get_model('xgb')
            if model_xgb is not None:
                train_and_evaluate_and_save(XGBClassifier, xgb_params, 'XGBoost', model_folder+'results_xgb')

        

        if  button8:
            st.markdown("#### Model Comparison")

            # Retrieve model results from st.session_state
            results_rf = st.session_state.get('results_rf')
            results_xgb = st.session_state.get('results_xgb')
            results_lgb = st.session_state.get('results_lgb')

            if results_rf is not None and results_xgb is not None and results_lgb is not None:
                # Create a DataFrame with model results
                results = {
                    'Model': ['Random Forest', 'XGBoost', 'LightGBM'],
                    'Accuracy': [results_rf['accuracy'], results_xgb['accuracy'], results_lgb['accuracy']],
                    'Precision': [results_rf['precision'], results_xgb['precision'], results_lgb['precision']],
                    'Recall': [results_rf['recall'], results_xgb['recall'], results_lgb['recall']],
                    'F1 Score': [results_rf['f1'], results_xgb['f1'], results_lgb['f1']]
                }
                results_df = pd.DataFrame(results)

                # Round scores to two decimals
                results_df_rounded = results_df.round(3)

                # Display summary table
                st.write("**Model Performance Summary:**")
                st.write(results_df_rounded)

                # Create a chart of scores with Plotly
                results_df_melted = results_df.melt(id_vars='Model', var_name='Metric', value_name='Score')
                fig = px.bar(results_df_melted, y='Model', x='Score', color='Metric', barmode='group',
                             color_discrete_sequence=px.colors.qualitative.Pastel,
                            labels={'Score': 'Score', 'Metric': 'Metric'},
                             text='Score')
                # Adjust the graduation of the X-axis for better readability
                fig.update_xaxes(
                    title_text='Score',
                    tickformat=".2f",  # Tick format to display two decimals
                    dtick=0.1,  # Tick interval (0.1 for more precision)
                    range=[0, 1]  # Custom range 
                )

                # Adjust the height of the chart
                fig.update_layout(
                    height=500,  # Custom height in pixels (adjust as needed)
                    title_text='Comparison of Classification Models',
                    xaxis_title='Score',
                    yaxis_title='Model'
                )
                # Adjust the text of the bars to display scores rounded to three decimals
                fig.update_traces(texttemplate='%{text:.3f}')
                st.plotly_chart(fig)

            else:
                st.write("Model results are not available. Make sure you have run the corresponding buttons to train and evaluate the models.")
                st.divider()
                            
            st.markdown("#### Interpretation of the results")
            st.markdown("""
            - **Accuracy (overall precision)**: rate of correct predictions
            - Between 74 and 75%
            - This indicates that 75.1% of the predictions made by the model are correct on average.
            - **Precision (precision)**: Proportion of true positives among all cases identified as positive by the model
            - Between 81.2 and 82.7%
            - Among the customers that the model predicted as being interested in the deposit, nearly 83% actually are.
            - This indicates that when a customer is targeted as likely to accept the offer, there is a high probability that this is the case.
            - **Recall (sensitivity)**: Proportion of true positives among all cases that are actually positive
            - Between 60 and 61%
            - Among all the customers that are actually interested in the deposit, the model correctly identified 60%
            - This means that the model captures about 60% of the potential customers for the campaign, which is a good performance, but there is still a portion of the potential customers that the model fails to identify.
            - **F1 Score**: Harmonic mean between Precision and Recall
            - Between 69 and 70%
            - An F1 Score of 70% indicates a good balance between the ability to correctly identify interested customers (recall) and the accuracy of the predictions (precision).
            """)
            st.markdown("#### Classification:")
            st.markdown("""
            - **True Positives**: The Random Forest model has the highest number of true positives (821 vs. 806 XGBoost and 813 LightGBM), which means that it has identified the highest number of truly interested customers for the deposit.
            - **False Negatives**: The Random Forest model has the fewest false negatives (520 vs. 535 XGBoost and 528 LightGBM), meaning that it missed the fewest truly interested customers.
            - **False Positives**: The XGBoost and LightGBM models have the fewest false positives (173 XGBoost and 170 LightGBM vs. 190 for Random Forest), meaning that these algorithms made the fewest errors in identifying uninterested customers as interested.
            - Also, **Random Forest appears to be better at identifying True Positives, which remains the main objective: the correct prediction of interested customers**.
            """)



      
# PAGE5 CONCLUSION
def show_conclusion_page():
    st.title('Conclusion and Perspectives')
    st.markdown("#### Machine Learning to predict customer targets")
    # Insert containers separated into tabs:
    tab1, tab2 = st.tabs(["Customer Targeting", "Predictive Perspectives"])
    
    with tab1:
        st.subheader('Customer Targeting')
        # col1 = st.columns(1)
        # with col1:
        st.markdown("""Data analysis allows targeting the bank's customers most likely to make a term deposit as well as the most effective campaign strategy.
        - A young adult or retired customer, of a relatively higher CSP, without current loans, with a higher average account balance.
        - A campaign from February to April and from September to December, with at least 1 customer contact beforehand, and call durations of at least 4 minutes.
        """)
        st.subheader('Marketing Insights')
        st.markdown(""" Other avenues should not be overlooked:
        - Raise awareness among canvassing and marketing teams about the customer profile and the most effective strategies
        - Diversify customer contact channels (mailing, social networks, etc.)
        - Collect more data using these diversified channels
        - Also review the customer approach and the speech to reach other targets
        """)


    with tab2:
        st.subheader('Machine Learning for targeting')
        st.markdown(""" By taking into account the call duration, the models are very efficient. However, it is preferable to use a model that is applicable on the real and a priori basis of customer data.
        - By deploying one of these algorithms on the bank's customer dataset, it is possible to predict customers likely to subscribe to this type of banking product.
        - This will allow a correct prediction of at least 60% of customers willing to make a deposit.
        - The choice of model among LightGBM, XGBoost and Random Forest also depends on the infrastructure available for deployment and the data set on which the predictions will be made for the next campaign (dimensions of the game, parameterization capacity, etc.).
        - More information below on the constraints of the models.
        """)
        st.write("")
        st.subheader('Continuous optimization')
        st.markdown(""" With each campaign, the collection of additional data will allow the models to refine their knowledge of customers and the predictive system will feed itself.
        - With this prediction approach, coupled with the awareness of marketing agents, the next campaign can hope for better results - higher than the 47% success of the analyzed campaign.
        - A contribution of additional data (Day and year to analyze the most favorable days for the campaign, customer transactions, feedback collection, etc.) would enrich the data to refine the models.
        - The analysis of the next results and of each campaign will allow a process of continuous improvement to always better target, increase the subscription rate and ensure increased efficiency of future campaigns.
        """)
        st.write("")

        st.divider()
        st.write("")
       
        with st.expander('**Model Information**'):
            st.divider()
            st.markdown(""" **Random Forest** is a decision tree model that uses multiple trees to improve accuracy and avoid overfitting.
            - **Pros:**
            - Robust to overfitting
            - Good performance for large data
            - **Cons:**
            - Less interpretable than simple decision trees
            - Can be slow to train for large data sets
            """)
            st.divider()

            st.markdown(""" **XGBoost** is an implementation of gradient boosting that is efficient and performant for classification and regression tasks.
            - **Pros:**
            - Very performant for classification and regression tasks
            - Computationally efficient
            - **Cons:**
            - Complexity of hyperparameters to tune
            - Can be prone to overfitting if poorly parameterized
            """)
        
            st.divider()
            st.markdown(""" **LightGBM** is a tree-based gradient boosting framework that is designed to be distributed and efficient with large data capacity.
            - **Pros:**
            - Very fast and efficient for large datasets
            - Less memory consumed compared to other boosting models
            - **Cons:**
            - Less performant for small datasets
            - Can be harder to interpret
            """)


# Main function to display the selected page
def main():
    if page == "Project":
        show_projet_page()
    elif page == "Dataset":
        show_data_set_page()
    elif page == "Data Visualization":
        show_data_viz_page()
    elif page == "Preprocessing":
        show_preprocessing_page()
    elif page == "Modeling":
        show_modelling_page()
    elif page == "Conclusion":
        show_conclusion_page()
if __name__ == "__main__":
    main()
