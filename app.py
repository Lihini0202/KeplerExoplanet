import streamlit as st
import pandas as pd
import numpy as np
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score
from streamlit_shap import st_shap

st.set_page_config(page_title="Ghosting Prediction App", layout="wide")
st.title("💘 Ghosting Prediction in Dating Apps")
st.markdown("An AI-powered Streamlit tool to analyze, model, and predict ghosting behavior.")

# Sidebar navigation
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", [
    "📁 Data Upload & Overview",
    "📊 Data Preprocessing",
    "🧠 Model Training (XGBoost)",
    "📈 Model Evaluation",
    "🤖 Live Prediction",
    "🔍 SHAP Interpretability"
])

# Session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'X_scaled' not in st.session_state:
    st.session_state.X_scaled = None
if 'y' not in st.session_state:
    st.session_state.y = None

# 1. Data Upload
if page == "📁 Data Upload & Overview":
    st.header("📁 Upload Your Dataset")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.success("Data uploaded successfully!")
        st.dataframe(df.head(10))
        st.write("### Basic Info")
        st.write(df.describe())
        st.write("### Missing Values")
        st.write(df.isnull().sum())

# 2. Preprocessing
elif page == "📊 Data Preprocessing":
    st.header("📊 Data Preprocessing")
    if 'df' in st.session_state:
        df = st.session_state.df.dropna()
        st.write("✅ Dropped missing values.")
        st.dataframe(df.head())

        X = df.drop(columns=['Match_Outcome'])  # <-- change target if needed
        y = df['Match_Outcome']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X.select_dtypes(include=[np.number]))

        st.session_state.X_scaled = X_scaled
        st.session_state.y = y
        st.success("✅ Preprocessing complete. Data is scaled and ready.")
    else:
        st.warning("⚠️ Please upload your dataset first.")

# 3. Model Training
elif page == "🧠 Model Training (XGBoost)":
    st.header("🧠 Train XGBoost Model")
    if st.session_state.X_scaled is not None and st.session_state.y is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            st.session_state.X_scaled, st.session_state.y, test_size=0.2, stratify=st.session_state.y, random_state=42)
        
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train, y_train)
        st.success("✅ Model trained successfully!")
        
        st.session_state.model = model
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test

        st.write("### Feature Importance")
        xgb.plot_importance(model)
        st.pyplot()
    else:
        st.warning("⚠️ Please preprocess the data first.")

# 4. Evaluation
elif page == "📈 Model Evaluation":
    st.header("📈 Model Evaluation")
    model = st.session_state.model
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    if model is not None and X_test is not None:
        y_pred = model.predict(X_test)
        st.subheader("📊 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        st.pyplot()

        st.subheader("📄 Classification Report")
        st.text(classification_report(y_test, y_pred))

        st.subheader("📉 ROC Curve")
        y_proba = model.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.figure()
        plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
        plt.plot([0,1],[0,1],'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        st.pyplot()
    else:
        st.warning("⚠️ Train the model first.")

# 5. Live Prediction
elif page == "🤖 Live Prediction":
    st.header("🤖 Live Prediction")
    model = st.session_state.model
    if model is not None:
        st.write("Upload new data (same format, no label):")
        new_file = st.file_uploader("Upload new user CSV", key="predict_upload")
        if new_file:
            new_data = pd.read_csv(new_file)
            X_new = StandardScaler().fit_transform(new_data.select_dtypes(include=[np.number]))
            pred = model.predict(X_new)
            st.write("### Predictions")
            st.write(pred)
    else:
        st.warning("⚠️ Model not trained yet.")

# 6. SHAP Explainability
elif page == "🔍 SHAP Interpretability":
    st.header("🔍 SHAP Explainability")
    model = st.session_state.model
    X_test = st.session_state.X_test
    if model is not None and X_test is not None:
        explainer = shap.Explainer(model)
        shap_values = explainer(X_test)
        st.subheader("📌 SHAP Summary Plot")
        st_shap(shap.plots.beeswarm(shap_values), height=400)
    else:
        st.warning("⚠️ Train the model first to view SHAP explanations.")
