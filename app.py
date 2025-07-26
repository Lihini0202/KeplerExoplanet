import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D,
    Dense, Dropout, BatchNormalization
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)
import xgboost as xgb
import requests
import os

st.title('Kepler Exoplanet Search Results 🔭')

# Helper function to download file from Google Drive
def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)
    
    # Get the total file size
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 # 1 Kibibyte
    
    progress_bar = st.progress(0)
    with open(destination, "wb") as f:
        bytes_written = 0
        for chunk in response.iter_content(block_size):
            if chunk:
                f.write(chunk)
                bytes_written += len(chunk)
                progress_bar.progress(min(bytes_written / total_size, 1.0))
    progress_bar.empty()


# Load Data with caching and download logic
@st.cache_data
def load_data():
    # Create a data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')

    # File IDs from your Google Drive links
    file_ids = {
        "data/cumulative.csv": "1ko0CO920amqyw6Trc3Xxp1Z72l4xC5Fc",
        "data/exoTrain.csv": "1MxDUIqQ6S9cmi068I62xkbCwZHkpAeJI",
        "data/exoTest.csv": "1d3bAfqatHaUWlRhc70YHW_Ay_co_ZmVu"
    }

    # Download files if they don't exist
    for filepath, file_id in file_ids.items():
        if not os.path.exists(filepath):
            st.info(f'Downloading {os.path.basename(filepath)}...')
            download_file_from_google_drive(file_id, filepath)

    summary_data = pd.read_csv("data/cumulative.csv")
    raw_train_data = pd.read_csv("data/exoTrain.csv")
    raw_test_data = pd.read_csv("data/exoTest.csv")
    return summary_data, raw_train_data, raw_test_data

summary_data, raw_train_data, raw_test_data = load_data()

st.header('Data Exploration')

# Display first 5 rows of each dataset
st.subheader('Summary Data')
st.write(summary_data.head())

st.subheader('Raw Training Data')
st.write(raw_train_data.head())

# Data Visualization
st.header('Data Visualization')

# Bar Chart of Dispositions
st.subheader('Distribution of KOI Dispositions')
fig, ax = plt.subplots(figsize=(8, 5))
sns.countplot(data=summary_data, x='koi_disposition', order=summary_data['koi_disposition'].value_counts().index, palette='Set2', ax=ax)
ax.set_title("Bar Chart: Distribution of KOI Dispositions")
ax.set_xlabel("Disposition")
ax.set_ylabel("Count")
ax.grid(axis='y', linestyle='--', alpha=0.5)
st.pyplot(fig)

# Pie Chart of Dispositions
st.subheader('Pie Chart: KOI Dispositions')
disposition_counts = summary_data['koi_disposition'].value_counts()
fig, ax = plt.subplots(figsize=(6, 6))
ax.pie(disposition_counts, labels=disposition_counts.index, autopct='%1.1f%%', colors=sns.color_palette('Set2'))
ax.set_title("Pie Chart: KOI Dispositions")
st.pyplot(fig)

# Histogram of Planet Radius
st.subheader('Histogram: Planet Radius Distribution (koi_prad)')
fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(summary_data['koi_prad'].dropna(), bins=50, kde=True, color='skyblue', ax=ax)
ax.set_title("Histogram: Planet Radius Distribution (koi_prad)")
ax.set_xlabel("Planet Radius (Earth radii)")
ax.set_ylabel("Frequency")
ax.grid(True, linestyle='--', alpha=0.3)
st.pyplot(fig)

# Model Training and Prediction
st.header('Model Training and Prediction')

if st.button('Train and Evaluate Models'):
    with st.spinner('Training and evaluating models... This may take a few minutes.'):
        # Preprocessing
        y_train = raw_train_data['LABEL'].values - 1
        X_train = raw_train_data.drop('LABEL', axis=1).values
        y_test = raw_test_data['LABEL'].values - 1
        X_test = raw_test_data.drop('LABEL', axis=1).values
        
        X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # CNN Model
        def create_cnn_model(input_shape):
            inp = Input(shape=input_shape)
            x = Conv1D(64, 5, activation='relu', padding='same')(inp)
            x = BatchNormalization()(x)
            x = MaxPooling1D(2)(x)
            x = Conv1D(128, 5, activation='relu', padding='same')(x)
            x = BatchNormalization()(x)
            x = MaxPooling1D(2)(x)
            x = Conv1D(256, 5, activation='relu', padding='same')(x)
            x = BatchNormalization()(x)
            x = GlobalAveragePooling1D()(x)
            x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
            x = Dropout(0.5)(x)
            out = Dense(1, activation='sigmoid')(x)
            model = Model(inp, out)
            return model

        model_cnn = create_cnn_model((X_train_cnn.shape[1], 1))
        model_cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

        st.text("Training CNN...")
        model_cnn.fit(X_train_cnn, y_train, epochs=50, batch_size=64, validation_split=0.2, callbacks=[early_stopping, reduce_lr], verbose=0)
        
        y_pred_cnn_proba = model_cnn.predict(X_test_cnn).ravel()
        y_pred_cnn = (y_pred_cnn_proba > 0.5).astype(int)

        st.subheader('CNN Model Results')
        st.text(classification_report(y_test, y_pred_cnn, target_names=['Not Exoplanet', 'Exoplanet']))
        st.write(f"Accuracy: {accuracy_score(y_test, y_pred_cnn):.4f}")

        # XGBoost Model
        st.text("Training XGBoost...")
        model_xgb = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        model_xgb.fit(X_train, y_train)
        
        y_pred_xgb = model_xgb.predict(X_test)
        y_pred_xgb_proba = model_xgb.predict_proba(X_test)[:, 1]

        st.subheader('XGBoost Model Results')
        st.text(classification_report(y_test, y_pred_xgb, target_names=['Not Exoplanet', 'Exoplanet']))
        st.write(f"Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}")

        # ROC Curve
        st.subheader('ROC Curve Comparison')
        fpr_cnn, tpr_cnn, _ = roc_curve(y_test, y_pred_cnn_proba)
        fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_pred_xgb_proba)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr_cnn, tpr_cnn, label=f'CNN (AUC = {roc_auc_score(y_test, y_pred_cnn_proba):.2f})')
        ax.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {roc_auc_score(y_test, y_pred_xgb_proba):.2f})')
        ax.plot([0, 1], [0, 1], 'k--', label='Chance')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
