import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import plotly.express as px

data = pd.read_csv("data/Financial_inclusion_dataset_final.csv")

st.set_page_config(page_title="Financial Inclusion Prediction", layout="centered")
st.title("🏦 Financial Inclusion Prediction App")
st.markdown("This app predicts whether an individual is likely to **have a bank account** based on demographic and socio-economic information.")

st.write("### Dataset Information")
st.write(data.head(), width=800, height=400)


st.write(f"Number of rows: {data.shape[0]}")
st.write(f"Number of columns: {data.shape[1]}")
st.write("### Columns")
# st.write(data.columns.tolist())


country = st.sidebar.selectbox("Country", data['country'].unique())
year = st.sidebar.selectbox("Year", data['year'].unique())
location_type = st.sidebar.selectbox("Location Type", data['location_type'].unique())
cellphone_access = st.sidebar.selectbox("Cellphone Access", data['cellphone_access'])
household_size = st.sidebar.number_input(
    'Household Size',
    min_value=int(data['household_size'].min()),
    max_value=int(data['household_size'].max()),
    value=int(data['household_size'].median())
)
age_of_respondent = st.sidebar.number_input(
    'Age of Respondent',
    min_value=int(data['age_of_respondent'].min()),
    max_value=int(data['age_of_respondent'].max()),
    value=int(data['age_of_respondent'].median())
)

gender_of_respondent = st.sidebar.selectbox(
    "Gender of Respondent",
    options=sorted(data['gender_of_respondent'].unique()),  # e.g., ["Female", "Male"]
    index=0  # default to first option ("Female" after sorting)
)

   

household_size = st.sidebar.selectbox(
    "Household Size",
    options=sorted(data['household_size'].unique()),  # all unique household sizes
    index=list(sorted(data['household_size'].unique())).index(int(data['household_size'].median()))  # default = median
)


relationship_with_head = st.sidebar.selectbox("Relationship with Head", 
                                              data['relationship_with_head'].unique())

marital_status = st.sidebar.selectbox("Marital Status", data['marital_status'].unique())

education_level = st.sidebar.selectbox("Education Level", data['education_level'].unique())

job_type = st.sidebar.selectbox("Job Type", data['job_type'].unique())


input_data = {
    'country': country,
    'year': year,
    'location_type': location_type,
    'cellphone_access': cellphone_access,
    'household_size': household_size,
    'age_of_respondent': age_of_respondent,
    'gender_of_respondent': gender_of_respondent,
    'relationship_with_head': relationship_with_head,
    'marital_status': marital_status,
    'education_level': education_level,
    'job_type': job_type
}


input_df = pd.DataFrame([input_data])
st.divider()
st.header("User Input")
st.dataframe(input_df)

st.write("### Feature Importance")
# Load the model and feature importance
country_encoder = joblib.load("encoders/country_encoder.pkl")
year_scaler = joblib.load("scalers/year_scaler.pkl")
location_type_encoder = joblib.load("encoders/location_type_encoder.pkl")
cellphone_access_encoder = joblib.load("encoders/cellphone_access_encoder.pkl")
household_size_scaler = joblib.load("scalers/household_size_scaler.pkl")
age_of_respondent_scaler = joblib.load("scalers/age_of_respondent_scaler.pkl")
gender_of_respondent_encoder = joblib.load("encoders/gender_of_respondent_encoder.pkl")
relationship_with_head_encoder = joblib.load("encoders/relationship_with_head_encoder.pkl")
marital_status_encoder = joblib.load("encoders/marital_status_encoder.pkl")
education_level_encoder = joblib.load("encoders/education_level_encoder.pkl")
job_type_encoder = joblib.load("encoders/job_type_encoder.pkl")


# Transform the input data
input_df["country"] = country_encoder.transform(input_df[["country"]])
input_df["year"] = year_scaler.transform(input_df[["year"]])
input_df["location_type"] = location_type_encoder.transform(input_df[["location_type"]])
input_df["cellphone_access"] = cellphone_access_encoder.transform(input_df[["cellphone_access"]])
input_df["household_size"] = household_size_scaler.transform(input_df[["household_size"]])
input_df["age_of_respondent"] = age_of_respondent_scaler.transform(input_df[["age_of_respondent"]])
input_df["gender_of_respondent"] = gender_of_respondent_encoder.transform(input_df[["gender_of_respondent"]])
input_df["relationship_with_head"] = relationship_with_head_encoder.transform(input_df[["relationship_with_head"]])
input_df["marital_status"] = marital_status_encoder.transform(input_df[["marital_status"]])
input_df["education_level"] = education_level_encoder.transform(input_df[["education_level"]])
input_df["job_type"] = job_type_encoder.transform(input_df[["job_type"]])


st.divider()

model = joblib.load("models/Financial_inclusion_dataset.pkl")

predictionButton = st.button("Predict")

if predictionButton:
    pred_class = model.predict(input_df)[0] 
    pred_prob = model.predict_proba(input_df)[0][1]  

    st.write(f"Prediction: {'YES' if pred_class == 1 else 'NO'}")
    st.write(f"Probability of having a bank account: {pred_prob:.2%}")


