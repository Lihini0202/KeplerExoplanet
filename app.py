import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import streamlit.components.v1 as components
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import xgboost as xgb
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Kepler Exoplanet Explorer",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="collapsed", # Sidebar is not used for navigation anymore
)

# --- Data Loading ---
@st.cache_data(show_spinner="Loading datasets...")
def load_data():

    data_dir = 'datasets'
    try:
        if not os.path.exists(data_dir):
            st.error(f"Error: The directory '{data_dir}' was not found. Please create it and add the required CSV files.")
            st.stop()
        summary = pd.read_csv(os.path.join(data_dir, "cumulative.csv"))
        train = pd.read_csv(os.path.join(data_dir, "exoTrain.csv"))
        test = pd.read_csv(os.path.join(data_dir, "exoTest.csv"))
    except FileNotFoundError as e:
        st.error(f"Error loading data: {e}. Make sure 'cumulative.csv', 'exoTrain.csv', and 'exoTest.csv' are in a 'datasets' folder.")
        st.stop()
    return summary, train, test

# --- Model Training and Caching ---
@st.cache_resource(show_spinner="Training models... this can take a few minutes.")
def train_and_evaluate_models(_train_df, _test_df):
    y_train = _train_df['LABEL'].values - 1
    X_train = _train_df.drop('LABEL', axis=1).values

    # CNN Model
    X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

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

    # XGBoost Model
    model_xgb = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=150, learning_rate=0.1, max_depth=5)
    model_xgb.fit(X_train, y_train)

    return model_cnn, model_xgb

# --- Page Content Functions ---
def home_page():
    st.header("Welcome to the Kepler Exoplanet Explorer! 🪐")
    st.markdown(
        """
        This interactive application allows you to explore the fascinating dataset from NASA's Kepler mission,
        which has been instrumental in discovering thousands of exoplanets.

        #### What can you do here?
        - **Explore Datasets**: View the raw data from the Kepler mission.
        - **Analyze Visually**: Dive deep into the data with interactive charts.
        - **Evaluate Models**: See the performance of ML models trained to classify exoplanet candidates.
        - **Make Predictions**: Predict whether a star is likely to host an exoplanet using its light curve data.

        Navigate through the tabs above to explore different sections of the app.
        """
    )
    st.subheader("🎥 The Legacy of NASA's Kepler Space Telescope")

    components.html(
       """
       <iframe width="700" height="400"
       src="https://www.youtube.com/embed/3yij1rJOefM"
       title="The Legacy of NASA's Kepler Space Telescope"
       frameborder="0"
       allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
       allowfullscreen>
       </iframe>
       """,
       height=420,
    )

def data_explorer_page(summary, train, test):
    st.header("📊 Data Explorer")
    st.markdown("---")

    st.subheader("Kepler Objects of Interest (KOI) Summary Data")
    st.dataframe(summary)
    with st.expander("Show Summary Statistics"):
        st.dataframe(summary.describe())

    sub_tab1, sub_tab2 = st.tabs(["Flux Training Data (exoTrain)", "Flux Test Data (exoTest)"])
    with sub_tab1:
        st.subheader("Flux Training Data (`exoTrain.csv`)")
        st.dataframe(train)
    with sub_tab2:
        st.subheader("Flux Test Data (`exoTest.csv`)")
        st.dataframe(test)

def visual_analytics_page(summary):
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
        ax.axis('equal')
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
        s_period = summary['koi_period'].dropna()
        s_period = s_period[s_period < 2000] # filter extreme outliers
        sns.histplot(s_period, ax=ax, kde=True, bins=30, color='orchid', log_scale=True)
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

def model_performance_page(train_df, test_df):
    st.header("🤖 Model Performance")
    st.markdown("---")

    # The cached function ensures models are trained only once.
    model_cnn, model_xgb = train_and_evaluate_models(train_df, test_df)

    y_test = test_df['LABEL'].values - 1
    X_test = test_df.drop('LABEL', axis=1).values

    pred_cnn_proba = model_cnn.predict(X_test.reshape(X_test.shape[0], X_test.shape[1], 1)).ravel()
    pred_cnn = (pred_cnn_proba > 0.5).astype(int)
    pred_xgb_proba = model_xgb.predict_proba(X_test)[:, 1]
    pred_xgb = (pred_xgb_proba > 0.5).astype(int)

    st.subheader("Model Accuracy Scores")
    col1, col2 = st.columns(2)
    acc_cnn = accuracy_score(y_test, pred_cnn)
    acc_xgb = accuracy_score(y_test, pred_xgb)
    with col1:
        st.metric("CNN Accuracy", f"{acc_cnn:.2%}")
    with col2:
        st.metric("XGBoost Accuracy", f"{acc_xgb:.2%}")

    sub_tab1, sub_tab2, sub_tab3 = st.tabs(["Classification Reports", "Confusion Matrices", "ROC Curve"])

    with sub_tab1:
        st.subheader("Classification Reports")
        col1, col2 = st.columns(2)
        with col1:
            st.text("CNN Model:")
            st.code(classification_report(y_test, pred_cnn, target_names=['Not Exoplanet', 'Exoplanet']))
        with col2:
            st.text("XGBoost Model:")
            st.code(classification_report(y_test, pred_xgb, target_names=['Not Exoplanet', 'Exoplanet']))

    with sub_tab2:
        st.subheader("Confusion Matrices")
        col1, col2 = st.columns(2)
        with col1:
            cm_cnn = confusion_matrix(y_test, pred_cnn)
            fig, ax = plt.subplots()
            sns.heatmap(cm_cnn, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=['Not Exoplanet', 'Exoplanet'], yticklabels=['Not Exoplanet', 'Exoplanet'])
            ax.set_title('CNN Confusion Matrix')
            ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
            st.pyplot(fig)
        with col2:
            cm_xgb = confusion_matrix(y_test, pred_xgb)
            fig, ax = plt.subplots()
            sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Greens', ax=ax, xticklabels=['Not Exoplanet', 'Exoplanet'], yticklabels=['Not Exoplanet', 'Exoplanet'])
            ax.set_title('XGBoost Confusion Matrix')
            ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
            st.pyplot(fig)

    with sub_tab3:
        st.subheader("ROC Curve Comparison")
        fpr_cnn, tpr_cnn, _ = roc_curve(y_test, pred_cnn_proba)
        fpr_xgb, tpr_xgb, _ = roc_curve(y_test, pred_xgb_proba)
        fig, ax = plt.subplots()
        ax.plot(fpr_cnn, tpr_cnn, label=f'CNN (AUC = {roc_auc_score(y_test, pred_cnn_proba):.3f})')
        ax.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {roc_auc_score(y_test, pred_xgb_proba):.3f})')
        ax.plot([0, 1], [0, 1], 'k--', label='Chance')
        ax.set_title('ROC Curve'); ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate'); ax.legend()
        st.pyplot(fig)

def prediction_playground_page(train_df, test_df):
    st.header("🚀 Prediction Playground")
    st.markdown("---")
    st.info("Select a star from the test set to see the models' predictions.")

    # Models will be loaded from cache if already trained in the performance tab.
    model_cnn, model_xgb = train_and_evaluate_models(train_df, test_df)

    y_test = test_df['LABEL'].values - 1
    X_test = test_df.drop('LABEL', axis=1).values

    star_index = st.slider("Select a Star Index from the Test Set", 0, len(test_df) - 1, 0)

    star_data = X_test[star_index]
    actual_label = "Exoplanet" if y_test[star_index] == 1 else "Not Exoplanet"

    st.subheader(f"Flux data for Star #{star_index}")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(star_data, color='royalblue')
    ax.set_title(f"Light Curve (Actual: {actual_label})")
    ax.set_xlabel("Time"); ax.set_ylabel("Flux")
    st.pyplot(fig)

    if st.button("Predict for this Star"):
        with st.spinner("Analyzing star..."):
            cnn_input = star_data.reshape(1, -1, 1)
            cnn_prob = model_cnn.predict(cnn_input)[0][0]
            cnn_pred = "Exoplanet" if cnn_prob > 0.5 else "Not Exoplanet"

            xgb_input = star_data.reshape(1, -1)
            xgb_prob = model_xgb.predict_proba(xgb_input)[0][1]
            xgb_pred = "Exoplanet" if xgb_prob > 0.5 else "Not Exoplanet"

        st.subheader("Prediction Results")
        st.markdown(f"**Actual Label:** `{actual_label}`")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("CNN Prediction", cnn_pred)
            st.progress(float(cnn_prob))
            st.write(f"Confidence (is Exoplanet): {cnn_prob:.2%}")
        with col2:
            st.metric("XGBoost Prediction", xgb_pred)
            st.progress(float(xgb_prob))
            st.write(f"Confidence (is Exoplanet): {xgb_prob:.2%}")

# --- Main App Logic ---
st.title("Kepler Exoplanet Explorer 🔭")
st.markdown("---")

# Load data once
summary_df, train_df, test_df = load_data()

# Create tabs for navigation
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Home",
    "📊 Data Explorer",
    "📈 Visual Analytics",
    "🤖 Model Performance",
    "🚀 Prediction Playground"
])

with tab1:
    home_page()

with tab2:
    data_explorer_page(summary_df, train_df, test_df)

with tab3:
    visual_analytics_page(summary_df)

with tab4:
    model_performance_page(train_df, test_df)

with tab5:
    prediction_playground_page(train_df, test_df)
