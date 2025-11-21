import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch

# ----------------------------------------------------
# عنوان التطبيق
# ----------------------------------------------------
st.title("Customer Segmentation Clustering")

# ----------------------------------------------------
# رفع الداتا
# ----------------------------------------------------
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file, sep='\t')
    st.dataframe(data.head())
    
    # ------------------------------------------------
    # اختيار المميزات الرقمية
    # ------------------------------------------------
    numeric_features = data.select_dtypes(include=['int64','float64']).columns
    X = data[numeric_features].fillna(0)
    
    # Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # ------------------------------------------------
    # K-Means Clustering
    # ------------------------------------------------
    k = st.slider("Select number of clusters for K-Means", 2, 10, 4)
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters_km = kmeans.fit_predict(X_scaled)
    data['Cluster_KMeans'] = clusters_km
    
    st.subheader("K-Means Cluster Summary")
    st.dataframe(data.groupby('Cluster_KMeans')[numeric_features].mean())
    
    # ------------------------------------------------
    # Agglomerative Clustering
    # ------------------------------------------------
    n_agg = st.slider("Select number of clusters for Agglomerative", 2, 10, 4)
    agg = AgglomerativeClustering(n_clusters=n_agg)
    clusters_agg = agg.fit_predict(X_scaled)
    data['Cluster_Agg'] = clusters_agg
    
    st.subheader("Agglomerative Cluster Summary")
    st.dataframe(data.groupby('Cluster_Agg')[numeric_features].mean())
    
    # ------------------------------------------------
    # رسومات
    # ------------------------------------------------
    st.subheader("Cluster Distribution (K-Means)")
    fig, ax = plt.subplots()
    sns.countplot(x='Cluster_KMeans', data=data, ax=ax)
    st.pyplot(fig)
    
    st.subheader("Cluster Distribution (Agglomerative)")
    fig2, ax2 = plt.subplots()
    sns.countplot(x='Cluster_Agg', data=data, ax=ax2)
    st.pyplot(fig2)