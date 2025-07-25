# mini_project.py

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st

# Set Streamlit config
st.set_page_config(page_title="💘 Ghosting Prediction App", layout="wide")
st.title("💘 Ghosting Prediction in Dating Apps")
st.markdown("An interactive ML demo predicting whether a match will ghost based on behavioral features.")

# Load data from Google Drive URLs
summary_url = "https://drive.google.com/uc?export=download&id=1ko0CO920amqyw6Trc3Xxp1Z72l4xC5Fc"
train_url   = "https://drive.google.com/uc?export=download&id=1MxDUIqQ6S9cmi068I62xkbCwZHkpAeJI"
test_url    = "https://drive.google.com/uc?export=download&id=1d3bAfqatHaUWlRhc70YHW_Ay_co_ZmVu"

summary_data = pd.read_csv(summary_url)
train_data = pd.read_csv(train_url)
test_data = pd.read_csv(test_url)

# Preview
st.subheader("📊 Data Preview")
st.dataframe(summary_data.head())

# Feature Engineering (example only)
X = train_data.drop(columns=["match_outcome"])
y = train_data["match_outcome"]

X_test = test_data.drop(columns=["match_outcome"])
y_test = test_data["match_outcome"]

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Model Training
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_scaled.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

with st.spinner("Training model..."):
    model.fit(X_scaled, y, epochs=10, batch_size=32, verbose=0)

# Evaluate
predictions = model.predict(X_test_scaled)
predictions_binary = (predictions > 0.5).astype(int)

accuracy = accuracy_score(y_test, predictions_binary)
st.success(f"✅ Model Accuracy: **{accuracy * 100:.2f}%**")

# Classification Report
report = classification_report(y_test, predictions_binary, output_dict=True)
st.subheader("📋 Classification Report")
st.dataframe(pd.DataFrame(report).transpose())

# Confusion Matrix
st.subheader("📉 Confusion Matrix")
cm = confusion_matrix(y_test, predictions_binary)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["Not Ghosted", "Ghosted"], yticklabels=["Not Ghosted", "Ghosted"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(fig)
