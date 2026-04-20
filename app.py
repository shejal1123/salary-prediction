
import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
import numpy as np

# --- Load models ---
@st.cache_resource # Cache the model loading
def load_models():
    linear_reg_model = pickle.load(open('linear_regression_model.pkl', 'rb'))
    decision_tree_model = pickle.load(open('decision_tree_model.pkl', 'rb'))
    random_forest_model = pickle.load(open('random_forest_model.pkl', 'rb'))
    return linear_reg_model, decision_tree_model, random_forest_model

linear_reg, decision_tree_reg, random_forest_reg = load_models()

# --- Load and preprocess data for encoders ---
@st.cache_data # Cache data loading and preprocessing
def load_and_preprocess_data_for_encoders():
    # Load the original dataset to fit LabelEncoders
    original_df = pd.read_csv('/content/Salary_Dataset_DataScienceLovers.csv')

    # Fill null values in 'Company Name' with its mode, as done in training
    company_name_mode = original_df['Company Name'].mode()[0]
    original_df['Company Name'].fillna(company_name_mode, inplace=True)

    # Identify categorical columns
    categorical_cols = original_df.select_dtypes(include=['object']).columns

    # Create and fit LabelEncoders for each categorical column
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        le.fit(original_df[col])
        encoders[col] = le
    return encoders, categorical_cols, original_df

encoders, categorical_cols, original_df_for_enc = load_and_preprocess_data_for_encoders()

# --- Streamlit UI ---
st.title('Salary Prediction App')
st.markdown("""
This app predicts salary based on various job-related features using three different regression models.
""")

# Input features from user
st.sidebar.header('Input Features')

rating = st.sidebar.slider('Rating', 0.0, 5.0, 3.5, 0.1)
salaries_reported = st.sidebar.number_input('Salaries Reported', min_value=1, value=5)

# For categorical features, we need dropdowns with original values
company_name = st.sidebar.selectbox('Company Name', options=sorted(original_df_for_enc['Company Name'].unique()))
job_title = st.sidebar.selectbox('Job Title', options=sorted(original_df_for_enc['Job Title'].unique()))
location = st.sidebar.selectbox('Location', options=sorted(original_df_for_enc['Location'].unique()))
employment_status = st.sidebar.selectbox('Employment Status', options=sorted(original_df_for_enc['Employment Status'].unique()))
job_roles = st.sidebar.selectbox('Job Roles', options=sorted(original_df_for_enc['Job Roles'].unique()))


# --- Prediction ---
if st.sidebar.button('Predict Salary'):
    # Create a DataFrame for the input
    input_data = pd.DataFrame([[rating, company_name, job_title, salaries_reported, location, employment_status, job_roles]],
                              columns=['Rating', 'Company Name', 'Job Title', 'Salaries Reported', 'Location', 'Employment Status', 'Job Roles'])

    # Apply label encoding to the input data
    encoded_input_data = input_data.copy()
    for col in categorical_cols:
        le = encoders[col]
        try:
            encoded_input_data[col] = le.transform(input_data[col])
        except ValueError:
            st.warning(f"Warning: Unknown category '{input_data[col].iloc[0]}' for column '{col}'. Using a default encoded value (0).")
            encoded_input_data[col] = 0 # Fallback for unknown categories


    st.subheader('Prediction Results')

    # Predict with Linear Regression
    pred_lr = linear_reg.predict(encoded_input_data)
    st.write(f'Linear Regression Prediction: ₹{pred_lr[0]:,.2f}')

    # Predict with Decision Tree Regressor
    pred_dt = decision_tree_reg.predict(encoded_input_data)
    st.write(f'Decision Tree Prediction: ₹{pred_dt[0]:,.2f}')

    # Predict with Random Forest Regressor
    pred_rf = random_forest_reg.predict(encoded_input_data)
    st.write(f'Random Forest Prediction: ₹{pred_rf[0]:,.2f}')

    st.success('Prediction complete!')
