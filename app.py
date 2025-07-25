import pandas as pd
import streamlit as st

@st.cache_data
def load_csv_from_drive(file_id):
    # Convert file ID to direct download URL
    url = f"https://drive.google.com/uc?export=download&id={file_id}" # Added &export=download for direct download
    try:
        df = pd.read_csv(url)
        return df
    except Exception as e:
        st.error(f"Error loading file ID {file_id}: {e}")
        st.warning(f"Attempting to read URL content directly to debug: {url}")
        # For debugging: try to read as text to see what's actually being downloaded
        # import requests
        # response = requests.get(url)
        # st.text(response.text[:500]) # Show first 500 characters of the response
        return pd.DataFrame() # Return an empty DataFrame on error

# 📂 File IDs (replace with your actual IDs)
summary_id = "1ko0CO920amqyw6Trc3Xxp1Z72l4xC5Fc"
train_id   = "1MxDUIqQ6S9cmi068I62xkbCwZHkpAeJI"
test_id    = "1d3bAfqatHaUW1Rhc70YHW_Ay_co_ZmVu" # Corrected test_id from the image

# 📊 Load data
summary_data = load_csv_from_drive(summary_id)
train_data   = load_csv_from_drive(train_id)
test_data    = load_csv_from_drive(test_id)

st.success("✅ All datasets loaded from Google Drive!")

# Display data using st.dataframe() for better rendering
st.write("### Preview of Summary Data")
st.dataframe(summary_data.head())

st.write("### Preview of Training Data")
st.dataframe(train_data.head()) # This will show an empty DataFrame or error if the load failed

st.write("### Preview of Test Data")
st.dataframe(test_data.head())
