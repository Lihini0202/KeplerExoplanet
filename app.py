import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ✅ Function to load CSV from a full URL (e.g., Google Drive direct link)
@st.cache_data
def load_csv_from_url(file_url, file_name):
    try:
        df = pd.read_csv(file_url)
        st.success(f"✅ Loaded {file_name} from URL!")
        return df
    except Exception as e:
        st.error(f"❌ Error loading {file_name}: {e}")
        st.warning(f"Make sure '{file_name}' is publicly accessible and in raw CSV format.")
        st.warning(f"Attempted URL: {file_url}")
        return pd.DataFrame()

# ✅ Replace these with direct download links from Google Drive
summary_url = "https://drive.google.com/uc?export=download&id=1ko0CO920amqyw6Trc3Xxp1Z72l4xC5Fc"
train_url   = "https://drive.google.com/uc?export=download&id=1MxDUIqQ6S9cmi068I62xkbCwZHkpAeJI"
test_url    = "https://drive.google.com/uc?export=download&id=1d3bAfqatHaUWlRhc70YHW_Ay_co_ZmVu"  # Make sure this ID is complete

st.set_page_config(layout="wide")
st.title("🔭 Exoplanet Detection Model Dashboard")

# 📊 Load data
summary_data = load_csv_from_url(summary_url, "Summary Data")
train_data   = load_csv_from_url(train_url, "Training Data")
test_data    = load_csv_from_url(test_url, "Test Data")

# --- Display Data Previews ---
with st.expander("📄 Show Data Previews"):
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

# --- Model Training ---
if not train_data.empty and not test_data.empty:
    st.header("⚙️ Model Training and Evaluation")

    try:
        if 'LABEL' in train_data.columns and 'LABEL' in test_data.columns:
            X_train = train_data.drop('LABEL', axis=1)
            y_train = train_data['LABEL']

            X_test = test_data.drop('LABEL', axis=1)
            y_test = test_data['LABEL']

            # Ensure columns match
            common_features = list(set(X_train.columns) & set(X_test.columns))
            X_train = X_train[common_features]
            X_test = X_test[common_features]

            st.success("✅ Data Preparation Complete")
            st.write(f"Train features: {X_train.shape}, Test features: {X_test.shape}")

            # 🔍 Train model
            st.subheader("🎯 Training Random Forest Classifier")
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)

            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            train_acc = accuracy_score(y_train, y_pred_train)
            test_acc = accuracy_score(y_test, y_pred_test)

            col1, col2 = st.columns(2)
            col1.metric("Training Accuracy", f"{train_acc:.4f}")
            col2.metric("Test Accuracy", f"{test_acc:.4f}")

            # 📊 Classification report
            st.subheader("📋 Classification Report (Test Set)")
            report = classification_report(y_test, y_pred_test, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())

            # 📉 Confusion matrix
            st.subheader("🔀 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred_test)
            fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                        xticklabels=['Predicted 0', 'Predicted 1'],
                        yticklabels=['Actual 0', 'Actual 1'],
                        ax=ax_cm)
            ax_cm.set_xlabel('Predicted')
            ax_cm.set_ylabel('Actual')
            ax_cm.set_title('Confusion Matrix')
            st.pyplot(fig_cm)

        else:
            st.error("❗ 'LABEL' column not found in both datasets. Please check your files.")

    except Exception as e:
        st.error(f"❌ An error occurred during training/evaluation: {e}")
        st.exception(e)
else:
    st.info("ℹ️ Please verify that training and test data are properly loaded.")
