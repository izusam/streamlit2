import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import plotly.express as px

data = pd.read_csv("data/expresso_processed_final.csv")

st.markdown("# Expresso Processed Data Overview")
st.write("This application provides an overview of the processed Expresso dataset.")
# Display the dataset information
st.write("### Dataset Information")
st.write(data.head(), width=800, height=400)

st.header("Project Data")
st.write("This section provides an overview of the processed Expresso dataset, including its shape and columns.")
st.write("### Dataset Shape")

st.write(f"Number of rows: {data.shape[0]}")
st.write(f"Number of columns: {data.shape[1]}")
st.write("### Columns")
# st.write(data.columns.tolist())



# tenure = st.sidebar.selectbox(
#     'Tenure',
#     sorted(data['TENURE'].unique())  # list of available 
#     # data['TENURE'].unique().tolist()  # list of available tenures
# )

montant = st.sidebar.number_input(
    'Montant',
    min_value=int(data['MONTANT'].min()),
    max_value=int(data['MONTANT'].max()),
    value=int(data['MONTANT'].median())
)

frequence_rech = st.sidebar.number_input(
    'Frequency_Rech',
    min_value=int(data['FREQUENCE_RECH'].min()),
    max_value=int(data['FREQUENCE_RECH'].max()),
    value=int(data['FREQUENCE_RECH'].median())
)

revenue = st.sidebar.number_input(
    'Revenue',
    min_value=int(data['REVENUE'].min()),
    max_value=int(data['REVENUE'].max()),
    value=int(data['REVENUE'].median())
)

# appu_argument = st.sidebar.number_input(
#     'Arpu_segment',
#     min_value=int(data['ARPU_SEGMENT'].min()),
#     max_value=int(data['ARPU_SEGMENT'].max()),
#     value=int(data['ARPU_SEGMENT'].median())
# )

frequence = st.sidebar.number_input(
    'Frequence',
    min_value=int(data['FREQUENCE'].min()),
    max_value=int(data['FREQUENCE'].max()),
    value=int(data['FREQUENCE'].median())
)

data_volume = st.sidebar.number_input(
    'data_volume',
    min_value=int(data["DATA_VOLUME"].min()),
    max_value=int(data["DATA_VOLUME"].max()),
    value=int(data["DATA_VOLUME"].median())
)

on_net = st.sidebar.number_input(
    'on_net',
    min_value=int(data["ON_NET"].min()),
    max_value=int(data["ON_NET"].max()),
    value=int(data["ON_NET"].median())
)

# mrg = st.sidebar.selectbox(
#     'mrg',
#     sorted(data['MRG'].unique())  
# )

regularity = st.sidebar.number_input(
    'regularity',
    min_value=int(data["REGULARITY"].min()),
    max_value=int(data["REGULARITY"].max()),
    value=int(data["REGULARITY"].median())
)


input_data = {
    # 'TENURE': tenure,
    'MONTANT': montant,
    'FREQUENCE_RECH': frequence_rech,
    'REVENUE': revenue,
    # 'ARPU_SEGMENT': appu_argument,
    'FREQUENCE': frequence,
    'DATA_VOLUME': data_volume,
    'ON_NET': on_net,
    # 'MRG': mrg,
    'REGULARITY': regularity
}


input_df = pd.DataFrame([input_data])
st.divider()
st.header("User Input")
st.dataframe(input_df)

# tenure_encoder = joblib.load("encoders/TENURE_encoder.pkl")
montant_scaler = joblib.load("scalers/MONTANT_scaler.pkl")
frequence_rech_scaler = joblib.load("scalers/FREQUENCE_RECH_scaler.pkl")
revenue_scaler = joblib.load("scalers/REVENUE_scaler.pkl")
# appu_argument_scaler = joblib.load("scalers/ARPU_SEGMENT_scaler.pkl")
frequence_scaler = joblib.load("scalers/FREQUENCE_scaler.pkl")
data_volume_scaler = joblib.load("scalers/DATA_VOLUME_scaler.pkl")
on_net_scaler = joblib.load("scalers/ON_NET_scaler.pkl")
# mrg_encoder = joblib.load("encoders/MRG_encoder.pkl")
regularity_scaler = joblib.load("scalers/REGULARITY_scaler.pkl")

# input_df["TENURE"]= tenure_encoder.transform(input_df[["TENURE"]])
input_df["MONTANT"]= montant_scaler.transform(input_df[["MONTANT"]])
input_df["FREQUENCE_RECH"]= frequence_rech_scaler.transform(input_df[["FREQUENCE_RECH"]])
input_df["REVENUE"]= revenue_scaler.transform(input_df[["REVENUE"]])
# input_df["ARPU_SEGMENT"]= appu_argument_scaler.transform(input_df[["ARPU_SEGMENT"]])
input_df["FREQUENCE"]= frequence_scaler.transform(input_df[["FREQUENCE"]])
input_df["DATA_VOLUME"]= data_volume_scaler.transform(input_df[["DATA_VOLUME"]])
input_df["ON_NET"]= on_net_scaler.transform(input_df[["ON_NET"]])
# input_df["MRG"]= mrg_encoder.transform(input_df[["MRG"]])
input_df["REGULARITY"]= regularity_scaler.transform(input_df[["REGULARITY"]])


st.divider()

model = joblib.load("models/expresso_processed_final.pkl")

predictionButton = st.button("Predict")

# if predictionButton:
#     prediction = model.predict(input_df)
#     st.success(f"Predicted Sales: {prediction[0]:.2f} units")

if predictionButton:
    pred_class = model.predict(input_df)[0] 
    pred_prob = model.predict_proba(input_df)[0][1]  

    st.write(f"Prediction: {'YES' if pred_class == 1 else 'NO'}")
    st.write(f"Probability of Churn: {pred_prob:.2%}")


