import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import xgboost as xgb
import requests
import os
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Kepler Exoplanet Explorer",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Helper function to download file from Google Drive ---
def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    
    progress_bar = st.progress(0, text=f"Downloading {os.path.basename(destination)}...")
    bytes_written = 0
    with open(destination, "wb") as f:
        for chunk in response.iter_content(block_size):
            if chunk:
                f.write(chunk)
                bytes_written += len(chunk)
                progress_value = min(bytes_written / total_size, 1.0) if total_size > 0 else 0
                progress_bar.progress(progress_value, text=f"Downloading {os.path.basename(destination)}...")
    progress_bar.empty()

# --- Caching and Data Loading ---
@st.cache_data(show_spinner="Loading datasets...")
def load_data():
    if not os.path.exists('data'):
        os.makedirs('data')

    file_ids = {
        "data/cumulative.csv": "1ko0CO920amqyw6Trc3Xxp1Z72l4xC5Fc",
        "data/exoTrain.csv": "1MxDUIqQ6S9cmi068I62xkbCwZHkpAeJI",
        "data/exoTest.csv": "1d3bAfqatHaUWlRhc70YHW_Ay_co_ZmVu"
    }

    for filepath, file_id in file_ids.items():
        if not os.path.exists(filepath):
            download_file_from_google_drive(file_id, filepath)

    summary = pd.read_csv("data/cumulative.csv")
    train = pd.read_csv("data/exoTrain.csv")
    test = pd.read_csv("data/exoTest.csv")
    return summary, train, test

# --- Model Training and Caching ---
@st.cache_resource(show_spinner="Training models... this can take a few minutes.")
def train_and_evaluate_models(X_train, y_train, X_test, y_test):
    # CNN Model
    X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    inp = Input(shape=(X_train_cnn.shape[1], 1))
    x = Conv1D(64, 5, activation='relu', padding='same')(inp)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Conv1D(128, 5, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    out = Dense(1, activation='sigmoid')(x)
    model_cnn = Model(inp, out)
    
    model_cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model_cnn.fit(X_train_cnn, y_train, epochs=20, batch_size=64, validation_split=0.2, 
                  callbacks=[EarlyStopping(patience=5, restore_best_weights=True)], verbose=0)
    
    pred_cnn_proba = model_cnn.predict(X_test_cnn).ravel()
    
    # XGBoost Model
    model_xgb = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=150, learning_rate=0.1, max_depth=5)
    model_xgb.fit(X_train, y_train)
    pred_xgb_proba = model_xgb.predict_proba(X_test)[:, 1]
    
    return model_cnn, model_xgb, pred_cnn_proba, pred_xgb_proba

# --- Pages ---
def home():
    st.title("Welcome to the Kepler Exoplanet Explorer 🔭")
    st.markdown("---")
    st.markdown(
        """
        This interactive application allows you to explore the fascinating dataset from NASA's Kepler mission,
        which has been instrumental in discovering thousands of exoplanets.
        
        ### What can you do here?
        - **Explore Datasets**: View the raw data from the Kepler mission.
        - **Analyze Visually**: Dive deep into the data with interactive charts and plots.
        - **Train Models**: Build and evaluate powerful machine learning models (CNN and XGBoost) to classify exoplanet candidates.
        - **Make Predictions**: Use the trained models to predict whether a star is likely to host an exoplanet.
        
        Use the sidebar on the left to navigate between the different sections of the app.
        """
    )
    st.image("https://www.nasa.gov/wp-content/uploads/2023/04/kepler-artwork-1-final.jpg", caption="Artistic impression of the Kepler Space Telescope. Image Credit: NASA")

def data_explorer(summary, train, test):
    st.header("📊 Data Explorer")
    st.markdown("---")
    
    st.subheader("Kepler Objects of Interest (KOI) Summary Data")
    st.dataframe(summary)
    with st.expander("Show Summary Statistics"):
        st.dataframe(summary.describe())

    tab1, tab2 = st.tabs(["Flux Training Data (exoTrain)", "Flux Test Data (exoTest)"])
    with tab1:
        st.subheader("Flux Training Data (`exoTrain.csv`)")
        st.dataframe(train)
    with tab2:
        st.subheader("Flux Test Data (`exoTest.csv`)")
        st.dataframe(test)

def visual_analytics(summary):
    st.header("📈 Visual Analytics")
    st.markdown("---")
    
    st.subheader("1. KOI Disposition Analysis")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        sns.countplot(data=summary, x='koi_disposition', ax=ax, palette='viridis')
        ax.set_title("Distribution of KOI Dispositions")
        st.pyplot(fig)
    with col2:
        disposition_counts = summary['koi_disposition'].value_counts()
        fig, ax = plt.subplots()
        ax.pie(disposition_counts, labels=disposition_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('viridis'))
        ax.set_title("Proportion of KOI Dispositions")
        st.pyplot(fig)

    st.subheader("2. Planet Characteristics")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        sns.histplot(summary['koi_prad'].dropna(), ax=ax, kde=True, bins=30, color='dodgerblue')
        ax.set_title("Planet Radius Distribution (Earth Radii)")
        ax.set_xlabel("Planet Radius")
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots()
        sns.histplot(summary['koi_period'].dropna(), ax=ax, kde=True, bins=30, color='orchid', log_scale=True)
        ax.set_title("Orbital Period Distribution (Days, Log Scale)")
        ax.set_xlabel("Orbital Period")
        st.pyplot(fig)

    st.subheader("3. Correlation Heatmap")
    with st.expander("Show Correlation Heatmap"):
        numeric_cols = summary.select_dtypes(include=np.number).columns.tolist()
        corr_matrix = summary[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, ax=ax, cmap='coolwarm', annot=False)
        ax.set_title("Correlation Matrix of Numeric Features")
        st.pyplot(fig)

def model_performance(train_df, test_df):
    st.header("🤖 Model Performance")
    st.markdown("---")

    y_train = train_df['LABEL'].values - 1
    X_train = train_df.drop('LABEL', axis=1).values
    y_test = test_df['LABEL'].values - 1
    X_test = test_df.drop('LABEL', axis=1).values

    model_cnn, model_xgb, pred_cnn_proba, pred_xgb_proba = train_and_evaluate_models(X_train, y_train, X_test, y_test)
    
    pred_cnn = (pred_cnn_proba > 0.5).astype(int)
    pred_xgb = (pred_xgb_proba > 0.5).astype(int)
    
    st.subheader("Model Accuracy Scores")
    acc_cnn = accuracy_score(y_test, pred_cnn)
    acc_xgb = accuracy_score(y_test, pred_xgb)
    st.metric("CNN Accuracy", f"{acc_cnn:.2%}")
    st.metric("XGBoost Accuracy", f"{acc_xgb:.2%}")

    tab1, tab2, tab3 = st.tabs(["Classification Reports", "Confusion Matrices", "ROC Curve"])
    
    with tab1:
        st.subheader("Classification Reports")
        st.text("CNN Model:")
        st.code(classification_report(y_test, pred_cnn, target_names=['Not Exoplanet', 'Exoplanet']))
        st.text("XGBoost Model:")
        st.code(classification_report(y_test, pred_xgb, target_names=['Not Exoplanet', 'Exoplanet']))

    with tab2:
        st.subheader("Confusion Matrices")
        col1, col2 = st.columns(2)
        with col1:
            cm_cnn = confusion_matrix(y_test, pred_cnn)
            fig, ax = plt.subplots()
            sns.heatmap(cm_cnn, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
            ax.set_title('CNN Confusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)
        with col2:
            cm_xgb = confusion_matrix(y_test, pred_xgb)
            fig, ax = plt.subplots()
            sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Greens', ax=ax, xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
            ax.set_title('XGBoost Confusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)
            
    with tab3:
        st.subheader("ROC Curve Comparison")
        fpr_cnn, tpr_cnn, _ = roc_curve(y_test, pred_cnn_proba)
        fpr_xgb, tpr_xgb, _ = roc_curve(y_test, pred_xgb_proba)
        fig, ax = plt.subplots()
        ax.plot(fpr_cnn, tpr_cnn, label=f'CNN (AUC = {roc_auc_score(y_test, pred_cnn_proba):.3f})')
        ax.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {roc_auc_score(y_test, pred_xgb_proba):.3f})')
        ax.plot([0, 1], [0, 1], 'k--', label='Chance')
        ax.set_title('ROC Curve')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend()
        st.pyplot(fig)


def prediction_playground(test_df):
    st.header("🚀 Prediction Playground")
    st.markdown("---")
    st.info("Select a star from the test set to see the models' predictions.")
    
    # We need to re-train the models here or load them if saved
    # For simplicity in this example, we re-train, but in production you'd load saved models
    y_train = train_df['LABEL'].values - 1
    X_train = train_df.drop('LABEL', axis=1).values
    y_test = test_df['LABEL'].values - 1
    X_test = test_df.drop('LABEL', axis=1).values
    
    model_cnn, model_xgb, _, _ = train_and_evaluate_models(X_train, y_train, X_test, y_test)
    
    star_index = st.slider("Select a Star Index from the Test Set", 0, len(test_df) - 1, 0)
    
    star_data = X_test[star_index]
    actual_label = "Exoplanet" if y_test[star_index] == 1 else "Not Exoplanet"

    st.subheader(f"Flux data for Star #{star_index}")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(star_data, color='royalblue')
    ax.set_title(f"Light Curve (Actual: {actual_label})")
    ax.set_xlabel("Time")
    ax.set_ylabel("Flux")
    st.pyplot(fig)
    
    if st.button("Predict for this Star"):
        # CNN Prediction
        cnn_input = star_data.reshape(1, -1, 1)
        cnn_prob = model_cnn.predict(cnn_input)[0][0]
        cnn_pred = "Exoplanet" if cnn_prob > 0.5 else "Not Exoplanet"

        # XGBoost Prediction
        xgb_input = star_data.reshape(1, -1)
        xgb_prob = model_xgb.predict_proba(xgb_input)[0][1]
        xgb_pred = "Exoplanet" if xgb_prob > 0.5 else "Not Exoplanet"
        
        st.subheader("Prediction Results")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("CNN Prediction", cnn_pred)
            st.progress(cnn_prob)
            st.write(f"Confidence: {cnn_prob:.2%}")
        with col2:
            st.metric("XGBoost Prediction", xgb_pred)
            st.progress(xgb_prob)
            st.write(f"Confidence: {xgb_prob:.2%}")

# --- Main App Logic ---
summary_df, train_df, test_df = load_data()

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Data Explorer", "📈 Visual Analytics", "🤖 Model Performance", "🚀 Prediction Playground"])

# Page Routing
if page == "🏠 Home":
    home()
elif page == "📊 Data Explorer":
    data_explorer(summary_df, train_df, test_df)
elif page == "📈 Visual Analytics":
    visual_analytics(summary_df)
elif page == "🤖 Model Performance":
    model_performance(train_df, test_df)
elif page == "🚀 Prediction Playground":
    prediction_playground(test_df)
