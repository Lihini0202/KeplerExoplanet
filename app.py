import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import streamlit.components.v1 as components

# Load datasets
@st.cache_data
def load_data():
    summary_df = pd.read_csv("datasets/cumulative.csv")
    train_df = pd.read_csv("datasets/exoTrain.csv")
    test_df = pd.read_csv("datasets/exoTest.csv")
    return summary_df, train_df, test_df

summary_df, train_df, test_df = load_data()

# -------------------- PAGE FUNCTIONS --------------------

def home():
    st.title("🔭 Kepler Exoplanet Explorer")
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/6f/Kepler_Space_Telescope_spacecraft_model_2.png/800px-Kepler_Space_Telescope_spacecraft_model_2.png", use_column_width=True)
    
    st.markdown("""
        Welcome to the **Kepler Exoplanet Explorer App**. This tool allows you to explore NASA's Kepler mission data and apply machine learning models to predict whether a signal indicates a confirmed exoplanet or a false positive.
    """)
    
    st.subheader("🎥 The Legacy of NASA's Kepler Space Telescope")
    components.html(
        """
        <iframe width="700" height="400"
        src="https://www.youtube.com/embed/3yij1rJOefM"
        frameborder="0" allowfullscreen></iframe>
        """,
        height=420,
    )

def data_explorer(summary, train, test):
    st.title("📊 Data Explorer")

    st.subheader("🔎 Kepler Summary Dataset")
    st.dataframe(summary.head())
    st.write("Shape:", summary.shape)

    st.subheader("🧪 Flux Training Data")
    st.dataframe(train.head())
    st.write("Shape:", train.shape)

    st.subheader("🧾 Flux Test Data")
    st.dataframe(test.head())
    st.write("Shape:", test.shape)

def visual_analytics(summary):
    st.title("📈 Visual Analytics")

    st.subheader("Histogram of Planet Candidates")
    fig, ax = plt.subplots()
    summary['koi_disposition'].value_counts().plot(kind='bar', ax=ax)
    st.pyplot(fig)

    st.subheader("Correlation Heatmap (Numerical Features Only)")
    numeric_cols = summary.select_dtypes(include=[np.number])
    corr = numeric_cols.corr()
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)

def model_performance(train_df, test_df):
    st.title("🤖 Model Performance")

    X_train = train_df.iloc[:, 1:].values
    y_train = train_df.iloc[:, 0].values
    X_test = test_df.iloc[:, 1:].values
    y_test = test_df.iloc[:, 0].values

    input_shape = (X_train.shape[1], 1)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    inputs = Input(shape=input_shape)
    x = Conv1D(32, kernel_size=5, activation='relu')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = BatchNormalization()(x)
    x = Conv1D(64, kernel_size=5, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    early_stop = EarlyStopping(patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(patience=3)

    history = model.fit(X_train, y_train, validation_split=0.2, epochs=15,
                        callbacks=[early_stop, reduce_lr], verbose=0)

    st.subheader("📉 Training History")
    st.line_chart(pd.DataFrame({
        "Train Loss": history.history["loss"],
        "Val Loss": history.history["val_loss"]
    }))

    y_pred = model.predict(X_test)
    y_pred = np.where(y_pred > 0.5, 1, 0)

    st.subheader("📋 Classification Report")
    st.text(classification_report(y_test, y_pred))

    st.subheader("🌀 Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

def prediction_playground(train_df, test_df):
    st.title("🚀 Prediction Playground")

    st.markdown("Try adjusting parameters to simulate exoplanet classification.")
    selected_flux = st.slider("Simulated Flux Value", float(train_df.iloc[:, 1:].min().min()), float(train_df.iloc[:, 1:].max().max()), 0.0)

    dummy_input = np.array([[selected_flux] * (train_df.shape[1] - 1)])
    dummy_input = dummy_input.reshape(dummy_input.shape[0], dummy_input.shape[1], 1)

    input_shape = (dummy_input.shape[1], 1)
    inputs = Input(shape=input_shape)
    x = Conv1D(32, kernel_size=5, activation='relu')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = BatchNormalization()(x)
    x = Conv1D(64, kernel_size=5, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_df.iloc[:, 1:].values.reshape(-1, train_df.shape[1] - 1, 1),
              train_df.iloc[:, 0].values,
              epochs=1, verbose=0)

    prediction = model.predict(dummy_input)
    result = "🪐 Exoplanet Detected" if prediction[0][0] > 0.5 else "❌ Not an Exoplanet"
    st.subheader(f"Prediction: {result}")

# -------------------- MAIN NAVIGATION --------------------

st.set_page_config(page_title="Kepler Exoplanet Explorer", layout="wide")

# Use modern tabbed navigation
tabs = st.tabs(["🏠 Home", "📊 Data Explorer", "📈 Visual Analytics", "🤖 Model Performance", "🚀 Prediction Playground"])

with tabs[0]:
    home()

with tabs[1]:
    data_explorer(summary_df, train_df, test_df)

with tabs[2]:
    visual_analytics(summary_df)

with tabs[3]:
    model_performance(train_df, test_df)

with tabs[4]:
    prediction_playground(train_df, test_df)
