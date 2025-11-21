import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch
import os

# -------------------------------
# Page config & Title
# -------------------------------
st.set_page_config(page_title="Customer Segmentation Clustering", layout="wide")
st.title("Customer Segmentation Clustering")

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.header("About App")
st.sidebar.info("""
This app performs Customer Segmentation using:
- K-Means Clustering
- Agglomerative Clustering

Upload your CSV/TSV file, select cluster numbers, and explore customer segments.
""")

# -------------------------------
# Download sample CSV (if exists)
# -------------------------------
sample_path = "sample.csv"
if os.path.exists(sample_path):
    st.sidebar.warning("You can download the sample CSV below")
    st.sidebar.download_button(
        label="Download sample.csv",
        data=open(sample_path, "rb"),
        file_name="sample.csv",
        mime="text/csv"
    )
else:
    st.sidebar.warning("sample.csv not found on server.")

# -------------------------------
# CSV Upload
# -------------------------------
st.subheader("Upload your CSV file")
uploaded_file = st.file_uploader("Upload CSV or TSV", type=["csv", "txt"])


if uploaded_file:
    try:
        # Try reading as normal CSV
        try:
            df = pd.read_csv(uploaded_file)
        except:
            # If fails, try tab-separated
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep='\t')

        st.subheader("Preview of Uploaded Data")
        st.dataframe(df.head())

        # -----------------------------------
        # Select numeric features
        # -----------------------------------
        numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_features) == 0:
            st.error("No numeric columns found for clustering!")
            st.stop()

        X = df[numeric_features].fillna(0)

        # -----------------------------------
        # Standardization
        # -----------------------------------
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # -----------------------------------
        # K-Means
        # -----------------------------------
        st.subheader("K-Means Clustering")
        k = st.slider("Select number of K-Means clusters", 2, 10, 4)
        kmeans = KMeans(n_clusters=k, random_state=42)
        df['Cluster_KMeans'] = kmeans.fit_predict(X_scaled)

        st.write("### K-Means Cluster Summary")
        st.dataframe(df.groupby('Cluster_KMeans')[numeric_features].mean())

        # Plot
        st.write("### K-Means Cluster Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x='Cluster_KMeans', data=df, ax=ax)
        st.pyplot(fig)

        # -----------------------------------
        # Agglomerative Clustering
        # -----------------------------------
        st.subheader("Agglomerative Clustering")
        n_agg = st.slider("Select number of Agglomerative clusters", 2, 10, 4)
        agg = AgglomerativeClustering(n_clusters=n_agg)
        df['Cluster_Agg'] = agg.fit_predict(X_scaled)

        st.write("### Agglomerative Cluster Summary")
        st.dataframe(df.groupby('Cluster_Agg')[numeric_features].mean())

        # Plot
        st.write("### Agglomerative Cluster Distribution")
        fig2, ax2 = plt.subplots()
        sns.countplot(x='Cluster_Agg', data=df, ax=ax2)
        st.pyplot(fig2)

        # -----------------------------------
        # Download processed data
        # -----------------------------------
        st.subheader("Download Clustered Data")
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Results CSV",
            data=csv,
            file_name="clustered_results.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error processing file: {e}")