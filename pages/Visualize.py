import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def load_data(csv_file):
    return pd.read_csv(csv_file)

def visualize_clusters(data, labels):
    plt.scatter(data['X'], data['Y'], c=labels, cmap='viridis')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('K-Means Clustering')
    st.pyplot()


st.title('K-Means Clustering Visualization')
    
# File uploader
uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
if uploaded_file is not None:
    data = load_data(uploaded_file)
    st.write("Data loaded successfully!")

    # Display the uploaded data
    st.write("Uploaded data:")
    st.write(data.head())

# Number of clusters input
k = st.slider("Select number of clusters (k)", min_value=1, max_value=10, value=3)

# Perform k-means clustering
kmeans = KMeans(n_clusters=k)
kmeans.fit(data[['X', 'Y']])
labels = kmeans.labels_

# Visualize clusters
visualize_clusters(data, labels)