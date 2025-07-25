import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import your machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier # Example model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# If you are using Keras/TensorFlow for the loss plot:
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense


@st.cache_data
def load_csv_from_drive(file_id, file_name):
    # Ensure this URL is correct for direct download, especially for large files
    url = f"https://drive.google.com/uc?id={file_id}"
    try:
        df = pd.read_csv(url)
        st.success(f"✅ Loaded {file_name} from Google Drive!")
        return df
    except Exception as e:
        st.error(f"❌ Error loading {file_name} (ID: {file_id}): {e}")
        st.warning(f"This might be due to Google Drive's virus scan warning for large files or incorrect sharing settings. Ensure '{file_name}' is publicly accessible for direct download.")
        st.warning(f"Attempted URL: {url}")
        # Optional: For deep debugging, uncomment to show raw response if it's HTML
        # import requests
        # response = requests.get(url)
        # st.code(response.text[:1000], language='html')
        return pd.DataFrame() # Return an empty DataFrame on failure

# 📂 File IDs (replace with your actual IDs)
summary_id = "1ko0CO920amqyw6Trc3Xxp1Z72l4xC5Fc"
train_id   = "1MxDUIqQ6S9cmi068I62xkbCwZHkpAeJI" # Your 250MB file
test_id    = "1d3bAfqatHaUW1Rhc70YHW_Ay_co_ZmVu"

st.set_page_config(layout="wide") # Use wide layout for better display of plots/tables
st.title("Exoplanet Detection Model Dashboard")

# 📊 Load data
summary_data = load_csv_from_drive(summary_id, "Summary Data")
train_data   = load_csv_from_drive(train_id, "Training Data")
test_data    = load_csv_from_drive(test_id, "Test Data")

# --- Display Data Previews ---
with st.expander("Show Data Previews"):
    if not summary_data.empty:
        st.subheader("Summary Data")
        st.dataframe(summary_data.head())

    if not train_data.empty:
        st.subheader("Training Data")
        st.dataframe(train_data.head())
        st.write(f"Shape of Training Data: {train_data.shape}")

    if not test_data.empty:
        st.subheader("Test Data")
        st.dataframe(test_data.head())
        st.write(f"Shape of Test Data: {test_data.shape}")

# --- Machine Learning Model Training and Evaluation ---
if not train_data.empty and not test_data.empty:
    st.header("2. Model Training and Evaluation")

    try:
        # --- Data Preparation (Adapt this to your notebook's logic) ---
        # Assuming 'LABEL' is the target column
        # Features are all other columns (FLUX.1, FLUX.2, etc.)
        if 'LABEL' in train_data.columns and 'LABEL' in test_data.columns:
            X_train = train_data.drop('LABEL', axis=1)
            y_train = train_data['LABEL']

            X_test = test_data.drop('LABEL', axis=1)
            y_test = test_data['LABEL']

            # Ensure columns match after dropping 'LABEL'
            # This is important if your test_data has different columns than train_data
            common_features = list(set(X_train.columns) & set(X_test.columns))
            X_train = X_train[common_features]
            X_test = X_test[common_features]

            st.subheader("Data Preparation Complete")
            st.write(f"Training features shape: {X_train.shape}")
            st.write(f"Training target shape: {y_train.shape}")
            st.write(f"Test features shape: {X_test.shape}")
            st.write(f"Test target shape: {y_test.shape}")

            # --- Model Training (Copy from your notebook) ---
            st.subheader("Training Model...")
            # Example using RandomForestClassifier, replace with your actual model
            # e.g., Keras model, another scikit-learn model
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1) # Use all available cores
            model.fit(X_train, y_train)
            st.success("Model trained successfully!")

            # --- Model Prediction ---
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            # --- Display Accuracies (as in your notebook) ---
            st.subheader("Model Performance Metrics")
            train_accuracy = accuracy_score(y_train, y_pred_train)
            test_accuracy = accuracy_score(y_test, y_pred_test)

            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Training Accuracy", value=f"{train_accuracy:.4f}")
            with col2:
                st.metric(label="Test Accuracy", value=f"{test_accuracy:.4f}")

            # --- Classification Report ---
            st.subheader("Classification Report (Test Set)")
            report = classification_report(y_test, y_pred_test, output_dict=True)
            df_report = pd.DataFrame(report).transpose()
            st.dataframe(df_report) # Display as a DataFrame

            # --- Confusion Matrix Plot ---
            st.subheader("Confusion Matrix (Test Set)")
            cm = confusion_matrix(y_test, y_pred_test)
            fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax_cm,
                        xticklabels=['Predicted 0', 'Predicted 1'],
                        yticklabels=['Actual 0', 'Actual 1'])
            ax_cm.set_xlabel('Predicted Label')
            ax_cm.set_ylabel('True Label')
            ax_cm.set_title('Confusion Matrix')
            st.pyplot(fig_cm) # Display the plot

            # --- Loss Plot (if applicable, e.g., for Keras/TensorFlow models) ---
            # If your model training returned a 'history' object (e.g., Keras model.fit)
            # history = model.fit(X_train, y_train, validation_split=0.2, epochs=10, batch_size=32)
            # st.subheader("Training & Validation Loss")
            # fig_loss, ax_loss = plt.subplots(figsize=(8, 5))
            # ax_loss.plot(history.history['loss'], label='Training Loss')
            # ax_loss.plot(history.history['val_loss'], label='Validation Loss')
            # ax_loss.set_title('Model Loss Over Epochs')
            # ax_loss.set_xlabel('Epoch')
            # ax_loss.set_ylabel('Loss')
            # ax_loss.legend()
            # st.pyplot(fig_loss)

        else:
            st.error("The 'LABEL' column was not found in one or both of the training/test datasets. Cannot proceed with model training.")


    except KeyError as ke:
        st.error(f"Missing expected column for model training/evaluation: {ke}. Please check your DataFrame column names (e.g., 'LABEL', 'FLUX.1').")
    except Exception as e:
        st.error(f"An unexpected error occurred during model training or evaluation: {e}")
        st.warning("Please ensure your data preprocessing and model code are robust and compatible with the loaded DataFrames.")
        st.exception(e) # Display full traceback for debugging
else:
    st.info("Training and/or Test data could not be loaded. Please check the file IDs and Google Drive sharing settings.")
