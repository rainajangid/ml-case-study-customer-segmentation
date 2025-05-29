
!pip install matplotlib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.decomposition import PCA

# Load data and model
df = pd.read_csv("segmented_customers.csv")
scaler, kmeans = joblib.load("kmeans_pipeline.pkl")

# Title
st.title("🧙 Customer Segmentation Wizard")
st.markdown("View your customers by clusters using PCA & clustering magic!")

# Sidebar Filters
st.sidebar.header("Filters")
selected_cluster = st.sidebar.selectbox("Select KMeans Cluster", sorted(df["Cluster_KMeans"].unique()))

# Filtered Data
df_filtered = df[df["Cluster_KMeans"] == selected_cluster]

# Metrics
st.subheader(f"Cluster {selected_cluster} Summary")
st.dataframe(df_filtered.describe()[['Income', 'Recency', 'CustomerTenure', 'FamilySize', 'TotalSpend']].T)

# PCA Scatter Plot
st.subheader("PCA Visualization")
fig, ax = plt.subplots()
sns.scatterplot(data=df, x="PCA1", y="PCA2", hue="Cluster_KMeans", palette="Set2", ax=ax)
ax.set_title("PCA - KMeans Clusters")
st.pyplot(fig)

# Heatmap
st.subheader("Feature Correlation Heatmap")
fig2, ax2 = plt.subplots()
sns.heatmap(df[['Income', 'Recency', 'CustomerTenure', 'FamilySize', 'TotalSpend', 'Education', 'Marital_Status']].corr(), annot=True, cmap="coolwarm", ax=ax2)
st.pyplot(fig2)

# Export Option
st.download_button("Download Clustered Data", df.to_csv(index=False), "segmented_customers.csv")