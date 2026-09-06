import streamlit
import streamlit as st
import pandas as pd
import pickle
import sklearn

# Load model
with open("credit_card_default (1).pkl", "rb") as file:
    model = pickle.load(file)

st.title("Credit Card Default Prediction")
LIMIT_BAL = st.number_input("Limit Balance", min_value=0, value=1000000)



PAY_0 = st.selectbox("PAY_0", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8])

PAY_2 = st.selectbox("PAY_2", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8])

PAY_3 = st.selectbox("PAY_3", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8])

PAY_4 = st.selectbox("PAY_4", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8])

PAY_5 = st.selectbox("PAY_5", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8])

PAY_6 = st.selectbox("PAY_6", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8])

PAY_AMT1 = st.number_input("PAY_AMT1", min_value=0, value=1000000)

PAY_AMT2 = st.number_input("PAY_AMT2", min_value=0, value=1000000)

PAY_AMT3 = st.number_input("PAY_AMT3", min_value=0, value=1000000)

PAY_AMT4 = st.number_input("PAY_AMT4", min_value=0, value=1000000)

PAY_AMT5 = st.number_input("PAY_AMT5", min_value=0, value=1000000)

PAY_AMT6 = st.number_input("PAY_AMT6", min_value=0, value=1000000)

EDUCATION=st.selectbox("EDUCATION", ["University", "Graduate School","High School","Unknown","Others"])

DELAY_COUNT = st.number_input("Delay Count", min_value=0, value=6)


input_data = pd.DataFrame({
    "LIMIT_BAL": [LIMIT_BAL],
    "PAY_0": [PAY_0],
    "PAY_2": [PAY_2],
    "PAY_3": [PAY_3],
    "PAY_4": [PAY_4],
    "PAY_5": [PAY_5],
    "PAY_6": [PAY_6],
    "DELAY_COUNT": [DELAY_COUNT],
    "EDUCATION": [EDUCATION],
    "PAY_AMT1": [PAY_AMT1],
    "PAY_AMT2": [PAY_AMT2],     
    "PAY_AMT3": [PAY_AMT3],
    "PAY_AMT4": [PAY_AMT4],
    "PAY_AMT5": [PAY_AMT5],
    "PAY_AMT6": [PAY_AMT6]
})

if st.button("Predict Default Status"):
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ Customer is likely to Default")
    else:
        st.success("✅ Customer is Not likely to Default")
