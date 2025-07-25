import pandas as pd
import streamlit as st

@st.cache_data
def load_csv_from_drive(file_id):
    # Convert file ID to direct download URL
    url = f"https://drive.google.com/uc?id={file_id}"
    return pd.read_csv(url)

# 📂 File IDs (replace with your actual IDs)
summary_id = "1ko0CO920amqyw6Trc3Xxp1Z72l4xC5Fc"
train_id   = "1MxDUIqQ6S9cmi068I62xkbCwZHkpAeJI"
test_id    = "1d3bAfqatHaUWlRhc70YHW_Ay_co_ZmVu"

# 📊 Load data
summary_data = load_csv_from_drive(summary_id)
train_data   = load_csv_from_drive(train_id)
test_data    = load_csv_from_drive(test_id)

st.success("✅ All datasets loaded from Google Drive!")

st.write("### Preview of Summary Data", summary_data.head())
st.write("### Preview of Training Data", train_data.head())
st.write("### Preview of Test Data", test_data.head())
