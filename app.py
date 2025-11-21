import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(page_title="Customer Segmentation Clustering", layout="wide")
st.title("Customer Segmentation Clustering")

# -------------------- SIDEBAR --------------------
st.sidebar.header("About App")
st.sidebar.info("""
This app performs Customer Segmentation using:
- K-Means Clustering
- Agglomerative Clustering

It automatically detects CSV/TSV separators including:
comma, tab, semicolon, and spaces.
""")

# -------------------- SAMPLE DOWNLOAD --------------------
sample_path = "marketing_campaign.csv"

if os.path.exists(sample_path):
    st.sidebar.success("You can download the sample CSV below")
    st.sidebar.download_button(
        label="Download marketing_campaign.csv",
        data=open(sample_path, "rb"),
        file_name="marketing_campaign.csv",
        mime="text/csv"
    )
else:
    st.sidebar.warning("Sample file 'marketing_campaign.csv' not found on server.")

# -------------------- FILE UPLOAD --------------------
st.subheader("Upload your CSV file")
uploaded_file = st.file_uploader("Upload CSV or TXT", type=["csv", "txt"])


def auto_read(file):
    """Try reading the uploaded file with multiple separators."""
    seps = [",", "\t", ";", r"\s+"]
    for sep in seps:
        try:
            file.seek(0)
            df = pd.read_csv(file, sep=sep, engine="python")
            if df.shape[1] > 1:
                return df
        except:
            pass
    
    file.seek(0)
    return pd.read_csv(file, engine="python")


if uploaded_file:
    try:
        df = auto_read(uploaded_file)

        st.subheader("Preview of Uploaded Data")
        st.dataframe(df.head())

        # -----------------------------------
        # Numeric columns
        # -----------------------------------
        numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_features) == 0:
            st.error("No numeric columns found! Try checking your file separator.")
            st.stop()

        X = df[numeric_features].fillna(0)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # -----------------------------------
        # K-Means
        # -----------------------------------
        st.subheader("K-Means Clustering")
        k = st.slider("Select number of K-Means clusters", 2, 10, 4)
        kmeans = KMeans(n_clusters=k, random_state=42)
        df["Cluster_KMeans"] = kmeans.fit_predict(X_scaled)

        st.write("### K-Means Cluster Summary")
        st.dataframe(df.groupby("Cluster_KMeans")[numeric_features].mean())

        fig, ax = plt.subplots()
        sns.countplot(x="Cluster_KMeans", data=df, ax=ax)
        st.pyplot(fig)

        # -----------------------------------
        # Agglomerative Clustering
        # -----------------------------------
        st.subheader("Agglomerative Clustering")
        n = st.slider("Select number of Agglomerative clusters", 2, 10, 4)
        agg = AgglomerativeClustering(n_clusters=n)
        df["Cluster_Agg"] = agg.fit_predict(X_scaled)

        st.write("### Agglomerative Cluster Summary")
        st.dataframe(df.groupby("Cluster_Agg")[numeric_features].mean())

        fig2, ax2 = plt.subplots()
        sns.countplot(x="Cluster_Agg", data=df, ax=ax2)
        st.pyplot(fig2)

        # -----------------------------------
        # Download result
        # -----------------------------------
        st.subheader("Download Clustered Data")
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Clustered Results",
            data=csv,
            file_name="clustered_results.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error: {e}")