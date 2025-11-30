import pandas as pd
import requests
import streamlit as st

API_URL = "http://localhost:8000"


# Util Functions
def predict_single(data):
    response = requests.post(f"{API_URL}/predict", json=data)
    if response.status_code != 200:
        st.error("Prediction error: " + response.text)
        return None
    return response.json()


def predict_batch(df):
    payload = {"records": df.to_dict(orient="records")}
    response = requests.post(f"{API_URL}/predict_batch", json=payload)
    if response.status_code != 200:
        st.error("Batch prediction error: " + response.text)
        return None
    return response.json()


def health_check():
    response = requests.get(f"{API_URL}/health")
    if response.status_code == 200:
        return response.json()
    return None


# Streamlit UI
st.title("Churn Prediction App")
st.write("Enter customer information to predict churn.")


# Show health status
status = health_check()
if status:
    st.success(
        f"Model Loaded: {status['model_loaded']} | Version: {status['model_version']}"
    )
else:
    st.error("Backend API is offline.")


# Input Form
st.subheader("Single Prediction")

with st.form(key="single_prediction_form"):
    Total_Relationship_Count = st.number_input("Total Relationship Count")
    Credit_Limit = st.number_input("Credit Limit")
    Total_Revolving_Bal = st.number_input("Total Revolving Balance")
    Total_Amt_Chng_Q4_Q1 = st.number_input("Total Amount Change Q4 to Q1")
    Total_Trans_Amt = st.number_input("Total Transaction Amount")
    Total_Trans_Ct = st.number_input("Total Transaction Count")
    Total_Ct_Chng_Q4_Q1 = st.number_input("Total Count Change Q4 to Q1")
    Avg_Utilization_Ratio = st.number_input("Average Utilization Ratio")

    submit_single = st.form_submit_button("Predict")

if submit_single:
    payload = {
        "Total_Relationship_Count": Total_Relationship_Count,
        "Credit_Limit": Credit_Limit,
        "Total_Revolving_Bal": Total_Revolving_Bal,
        "Total_Amt_Chng_Q4_Q1": Total_Amt_Chng_Q4_Q1,
        "Total_Trans_Amt": Total_Trans_Amt,
        "Total_Trans_Ct": Total_Trans_Ct,
        "Total_Ct_Chng_Q4_Q1": Total_Ct_Chng_Q4_Q1,
        "Avg_Utilization_Ratio": Avg_Utilization_Ratio,
    }

    result = predict_single(payload)

    if result:
        st.success(
            f"Prediction: {'Churn' if result['predictions'][0] == 1 else 'No Churn'}"
        )
        st.json(result)


# Batch Prediction
st.subheader("Batch Prediction")

uploaded_file = st.file_uploader("Upload CSV for batch prediction")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

    if st.button("Predict Batch"):
        result = predict_batch(df)
        if result:
            st.success("Batch Prediction Completed")
            st.write(
                f"Churn Rate: \n {sum(result['predictions']) / len(result['predictions']):.2%}"
            )
            st.json(result)
