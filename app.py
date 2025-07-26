import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from PIL import Image
import os
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.models import load_model

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="Kepler Exoplanet Explorer",
    page_icon="🔭",
    layout="wide"
)

# --- Load Data ---
@st.cache_data
def load_data():
    summary = pd.read_csv("datasets/cumulative.csv")
    train = pd.read_csv("datasets/exoTrain.csv")
    test = pd.read_csv("datasets/exoTest.csv")
    return summary, train, test

# --- Load XGBoost Model ---
@st.cache_resource
def load_xgb_model():
    return joblib.load("models/xgb_model.pkl")

# --- Load Label Encoder ---
@st.cache_resource
def load_label_encoder():
    return joblib.load("models/label_encoder.pkl")

# --- Page: Home ---
def home():
    st.title("🔭 Kepler Exoplanet Explorer")
    st.markdown("---")
    st.markdown("""
        Explore NASA's Kepler dataset and see how ML models detect exoplanets.
    """)
    st.subheader("🎥 NASA’s Kepler Mission")
    st.video("https://www.youtube.com/watch?v=3yij1rJOefM")

# --- Page: Data Explorer ---
def data_explorer(summary, train, test):
    st.header("📊 Data Explorer")

    st.subheader("KOI Summary Data")
    st.dataframe(summary)

    with st.expander("Show Summary Statistics"):
        st.dataframe(summary.describe())

    st.subheader("Training Data Sample")
    st.dataframe(train.head())

    st.subheader("Test Data Sample")
    st.dataframe(test.head())

# --- Page: Visual Analytics ---
def visual_analytics(summary):
    st.header("📈 Visual Analytics")

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        sns.countplot(x="koi_disposition", data=summary, ax=ax)
        ax.set_title("KOI Disposition Distribution")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        summary["koi_disposition"].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
        ax.set_ylabel("")
        ax.set_title("Disposition Proportions")
        st.pyplot(fig)

# --- Page: SHAP Explainability ---
def shap_explainer():
    st.header("🧠 SHAP Explainability")

    st.subheader("Bar Plot")
    bar_img = Image.open("shap_summary_bar.png")
    st.image(bar_img, use_column_width=True)

    st.subheader("Dot Plot")
    dot_img = Image.open("shap_summary_dot.png")
    st.image(dot_img, use_column_width=True)

# --- Page: CNN Fold Evaluation ---
def cnn_fold_evaluator(test_df):
    st.header("📊 CNN Fold Evaluation")

    X_test = test_df.drop("LABEL", axis=1).values
    y_test = test_df["LABEL"].values - 1  # assuming labels are 1 & 2
    X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    folds = [1, 2, 3, 4, 5]
    selected_fold = st.selectbox("Select Fold", folds)

    model_path = f"models/exoplanet_cnn_fold{selected_fold}.keras"
    if os.path.exists(model_path):
        try:
            model = load_model(model_path, compile=False)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return

        y_pred = (model.predict(X_test_cnn) > 0.5).astype("int").ravel()
        acc = accuracy_score(y_test, y_pred)

        st.metric(f"Fold {selected_fold} Accuracy", f"{acc:.2%}")

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=["Not Exoplanet", "Exoplanet"],
                    yticklabels=["Not Exoplanet", "Exoplanet"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
    else:
        st.error("Model file not found.")

# --- Page: Prediction Playground (XGBoost only) ---
def prediction_playground(test_df, model):
    st.header("🚀 Prediction Playground")

    y_test = test_df['LABEL'].values - 1
    X_test = test_df.drop("LABEL", axis=1).values

    idx = st.slider("Select a Star Index", 0, len(test_df)-1, 0)
    star = X_test[idx]
    actual = "Exoplanet" if y_test[idx] == 1 else "Not Exoplanet"

    st.subheader(f"Light Curve for Star #{idx}")
    fig, ax = plt.subplots()
    ax.plot(star, color="royalblue")
    ax.set_title(f"Actual: {actual}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Flux")
    st.pyplot(fig)

    if st.button("Run Prediction"):
        prob = model.predict_proba([star])[0][1]
        pred = "Exoplanet" if prob > 0.5 else "Not Exoplanet"

        st.subheader("Prediction Result")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Prediction", pred)
            st.write(f"Confidence: {prob:.2%}")
        with col2:
            st.metric("Actual", actual)
        st.progress(prob)

# --- Main Driver ---
summary_df, train_df, test_df = load_data()
xgb_model = load_xgb_model()
label_encoder = load_label_encoder()  # Load label encoder if needed

st.sidebar.title("🌌 Navigation")
page = st.sidebar.radio("Go to", [
    "🏠 Home",
    "📊 Data Explorer",
    "📈 Visual Analytics",
    "🧠 SHAP Explainability",
    "📊 CNN Fold Evaluation",
    "🚀 Prediction Playground"
])

if page == "🏠 Home":
    home()
elif page == "📊 Data Explorer":
    data_explorer(summary_df, train_df, test_df)
elif page == "📈 Visual Analytics":
    visual_analytics(summary_df)
elif page == "🧠 SHAP Explainability":
    shap_explainer()
elif page == "📊 CNN Fold Evaluation":
    cnn_fold_evaluator(test_df)
elif page == "🚀 Prediction Playground":
    prediction_playground(test_df, xgb_model)
