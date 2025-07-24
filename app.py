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
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)
import xgboost as xgb
from sklearn.model_selection import train_test_split

# --- App Configuration ---
st.set_page_config(page_title="Exoplanet Discovery Model Comparison", layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)

# --- Title and Description ---
st.title("🌌 Exoplanet Discovery: CNN vs. XGBoost")
st.markdown("""
This application analyzes Kepler telescope data to classify exoplanet candidates.
It trains and compares two different models: a **1D Convolutional Neural Network (CNN)** and an **XGBoost Classifier**.
- **CNN**: Ideal for learning features directly from time-series data like light flux curves.
- **XGBoost**: A powerful gradient boosting algorithm effective on tabular data.

**Instructions:** Click the "Train and Evaluate Models" button in the sidebar to start the process.
""")

# --- Caching Functions for Performance ---

@st.cache_data
def load_data():
    """Loads the training and test datasets."""
    try:
        raw_train_data = pd.read_csv("exoTrain.csv")
        raw_test_data = pd.read_csv("exoTest.csv")
        return raw_train_data, raw_test_data
    except FileNotFoundError:
        st.error("Error: `exoTrain.csv` or `exoTest.csv` not found. Please place them in the same directory as this script.")
        return None, None

@st.cache_data
def preprocess_data(_train_df, _test_df):
    """Preprocesses the data: separates features and labels, scales features."""
    # Prepare data
    X_train = _train_df.drop('LABEL', axis=1)
    y_train = _train_df['LABEL'] - 1  # Convert labels to 0 and 1
    X_test = _test_df.drop('LABEL', axis=1)
    y_test = _test_df['LABEL'] - 1

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Reshape for CNN
    X_train_cnn = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
    X_test_cnn = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

    return X_train_scaled, y_train, X_test_scaled, y_test, X_train_cnn, X_test_cnn

def build_cnn_model(input_shape):
    """Builds the 1D CNN model architecture."""
    input_layer = Input(shape=input_shape)
    x = Conv1D(filters=32, kernel_size=5, activation='relu', padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Conv1D(filters=64, kernel_size=5, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Conv1D(filters=128, kernel_size=5, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    output_layer = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

@st.cache_resource
def train_models(X_train_scaled, y_train, X_test_scaled, y_test, X_train_cnn, X_test_cnn):
    """Trains both CNN and XGBoost models and returns them."""
    # --- Train CNN ---
    with st.spinner('Training CNN Model... This might take a few minutes.'):
        cnn_model = build_cnn_model((X_train_cnn.shape[1], 1))
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
        cnn_model.fit(X_train_cnn, y_train,
                      validation_data=(X_test_cnn, y_test),
                      epochs=20,  # Reduced epochs for faster app demo
                      batch_size=64,
                      callbacks=[early_stopping, reduce_lr],
                      verbose=0) # Set verbose to 0 for cleaner Streamlit output
    st.success('CNN Model Trained!')

    # --- Train XGBoost ---
    with st.spinner('Training XGBoost Model...'):
        xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, n_estimators=100)
        xgb_model.fit(X_train_scaled, y_train,
                      eval_set=[(X_test_scaled, y_test)],
                      early_stopping_rounds=10,
                      verbose=False)
    st.success('XGBoost Model Trained!')

    return cnn_model, xgb_model

def plot_roc_curve(y_true, y_pred_proba, model_name, ax):
    """Plots a single ROC curve on a given matplotlib axes."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    ax.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})')

# --- Main App Logic ---
raw_train_data, raw_test_data = load_data()

if raw_train_data is not None:
    # Sidebar
    st.sidebar.header("Model Controls")
    if st.sidebar.button("Train and Evaluate Models"):
        st.header("📊 Model Performance Evaluation")

        # Preprocess data
        X_train_scaled, y_train, X_test_scaled, y_test, X_train_cnn, X_test_cnn = preprocess_data(raw_train_data, raw_test_data)

        # Train models
        cnn_model, xgb_model = train_models(X_train_scaled, y_train, X_test_scaled, y_test, X_train_cnn, X_test_cnn)

        # --- Predictions ---
        y_pred_proba_cnn = cnn_model.predict(X_test_cnn, verbose=0).ravel()
        y_pred_cnn = (y_pred_proba_cnn > 0.5).astype(int)

        y_pred_proba_xgb = xgb_model.predict_proba(X_test_scaled)[:, 1]
        y_pred_xgb = (y_pred_proba_xgb > 0.5).astype(int)

        # --- Display Results in Tabs ---
        tab1, tab2, tab3 = st.tabs(["🧠 CNN Results", "🌳 XGBoost Results", "🚀 Model Comparison"])

        with tab1:
            st.subheader("1D Convolutional Neural Network (CNN)")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Accuracy", f"{accuracy_score(y_test, y_pred_cnn):.2%}")
                st.text("Classification Report:")
                st.text(classification_report(y_test, y_pred_cnn, target_names=['Not Exoplanet', 'Exoplanet']))
            with col2:
                st.text("Confusion Matrix:")
                cm = confusion_matrix(y_test, y_pred_cnn)
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Exoplanet', 'Exoplanet'], yticklabels=['Not Exoplanet', 'Exoplanet'], ax=ax)
                plt.ylabel('Actual')
                plt.xlabel('Predicted')
                st.pyplot(fig)

        with tab2:
            st.subheader("XGBoost Classifier")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Accuracy", f"{accuracy_score(y_test, y_pred_xgb):.2%}")
                st.text("Classification Report:")
                st.text(classification_report(y_test, y_pred_xgb, target_names=['Not Exoplanet', 'Exoplanet']))
            with col2:
                st.text("Confusion Matrix:")
                cm = confusion_matrix(y_test, y_pred_xgb)
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=['Not Exoplanet', 'Exoplanet'], yticklabels=['Not Exoplanet', 'Exoplanet'], ax=ax)
                plt.ylabel('Actual')
                plt.xlabel('Predicted')
                st.pyplot(fig)

        with tab3:
            st.subheader("Performance Comparison")

            # --- Comparison Table ---
            report_cnn = classification_report(y_test, y_pred_cnn, output_dict=True)
            report_xgb = classification_report(y_test, y_pred_xgb, output_dict=True)

            comparison_data = {
                'Metric': ['Accuracy', 'Precision (Exoplanet)', 'Recall (Exoplanet)', 'F1-Score (Exoplanet)'],
                'CNN': [
                    report_cnn['accuracy'],
                    report_cnn['Exoplanet']['precision'],
                    report_cnn['Exoplanet']['recall'],
                    report_cnn['Exoplanet']['f1-score']
                ],
                'XGBoost': [
                    report_xgb['accuracy'],
                    report_xgb['Exoplanet']['precision'],
                    report_xgb['Exoplanet']['recall'],
                    report_xgb['Exoplanet']['f1-score']
                ]
            }
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df.style.format({'CNN': '{:.3f}', 'XGBoost': '{:.3f}'}), use_container_width=True)

            # --- Combined ROC Curve ---
            st.subheader("Receiver Operating Characteristic (ROC) Curve")
            fig, ax = plt.subplots(figsize=(8, 6))
            plot_roc_curve(y_test, y_pred_proba_cnn, 'CNN', ax)
            plot_roc_curve(y_test, y_pred_proba_xgb, 'XGBoost', ax)
            ax.plot([0, 1], [0, 1], 'k--', label='Chance')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Model ROC Curve Comparison')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

    # --- Data Exploration Section ---
    st.sidebar.header("Data Exploration")
    if st.sidebar.checkbox("Show Raw Data"):
        st.header("🔭 Raw Data Preview")
        st.dataframe(raw_train_data.head())

    if st.sidebar.checkbox("Show Light Curve Examples"):
        st.header("📈 Light Curve Examples")
        st.markdown("Randomly selected light curves from the training data. 'Confirmed' indicates a star with a confirmed exoplanet.")
        num_samples = 3
        sample_indices = np.random.choice(raw_train_data.shape[0], num_samples, replace=False)

        for idx in sample_indices:
            label = raw_train_data.iloc[idx]['LABEL']
            flux_values = raw_train_data.iloc[idx, 1:]
            fig, ax = plt.subplots(figsize=(14, 4))
            ax.plot(flux_values, label=f"Sample #{idx} | Label: {'Confirmed' if label == 2 else 'False Positive'}")
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Flux (Brightness)")
            ax.set_title(f"Light Curve for Sample #{idx}")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

