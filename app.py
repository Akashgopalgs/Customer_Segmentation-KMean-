import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load the preprocessed data and model
with open('preprocessed_customer_data.pkl', 'rb') as file:
    df = pickle.load(file)

with open('customer_segmentation_model.pkl', 'rb') as file:
    model_data = pickle.load(file)

kmeans = model_data['model']
scaler = model_data['scaler']
label_encoders = model_data['label_encoders']
features = model_data['features']

# Streamlit app
st.title("Customer Segmentation Prediction")

# User input features
st.sidebar.header("Input Features")


def get_user_input():
    input_data = {}
    for feature in features:
        if df[feature].dtype == 'object':
            unique_values = df[feature].unique()
            if feature in label_encoders:
                unique_values = label_encoders[feature].classes_
            input_data[feature] = st.sidebar.selectbox(feature, unique_values)
        else:
            input_data[feature] = st.sidebar.number_input(feature, value=float(df[feature].mean()), step=1.0)
    return pd.DataFrame([input_data])


user_input = get_user_input()

# Preprocess the input data
preprocessed_input = user_input.copy()
for feature in preprocessed_input.columns:
    if df[feature].dtype == 'object' and feature in label_encoders:
        preprocessed_input[feature] = label_encoders[feature].transform(preprocessed_input[feature])

# Ensure the input has the same number of features as the model expects
preprocessed_input = preprocessed_input[features]

# Scale the data
preprocessed_input_scaled = scaler.transform(preprocessed_input)

# Predict the cluster
try:
    prediction = kmeans.predict(preprocessed_input_scaled)
    cluster = prediction[0]
    st.write(f"Predicted Cluster: {cluster}")

    # Cluster Profile
    st.header("Cluster Profile")
    if cluster == 0:
        st.success("Cluster 0: Low spending and average to low income")
    elif cluster == 1:
        st.success("Cluster 1: High spending and High Income")

    # Scatter Plot
    st.subheader("Scatter Plot of Clusters")

    # Add cluster labels to the original data
    df['Cluster'] = kmeans.predict(scaler.transform(df[features]))

    # Plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df['Spent'], y=df['Income'], hue=df['Cluster'], palette='viridis', alpha=0.6, edgecolor='w')
    plt.scatter(preprocessed_input['Spent'], preprocessed_input['Income'], c='red', marker='x', s=100,
                label='User Input')
    plt.title('Customer Segmentation Clusters')
    plt.xlabel('Spent')
    plt.ylabel('Income')
    plt.legend()
    st.pyplot(plt)

except Exception as e:
    st.write(f"Error predicting cluster: {e}")
