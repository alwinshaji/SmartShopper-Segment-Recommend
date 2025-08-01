import streamlit as st
import pandas as pd
import joblib
from zipfile import ZipFile
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import os

# Load product data from zip
@st.cache_data
def load_product_data():
    with ZipFile("product_data.zip", 'r') as zip_ref:
        zip_ref.extractall("temp_data")
    df = pd.read_csv("temp_data/product_data.csv")
    return df

# Load RFM clustered data
@st.cache_data
def load_rfm_data():
    return pd.read_csv("rfm_clustered.csv")

# Load models
@st.cache_resource
def load_models():
    kmeans = joblib.load("kmeans_model.joblib")
    scaler = joblib.load("scaler.joblib")
    return kmeans, scaler

# Recommendation function
def get_recommendations(product_name, df):
    product_features = df[['StockCode', 'Description']].drop_duplicates()
    pivot = df.pivot_table(index='CustomerID', columns='Description', values='TotalPrice', aggfunc='sum', fill_value=0)
    similarity = cosine_similarity(pivot.T)
    sim_df = pd.DataFrame(similarity, index=pivot.columns, columns=pivot.columns)

    if product_name not in sim_df.columns:
        return ["Product not found in dataset"]

    recommendations = sim_df[product_name].sort_values(ascending=False)[1:6].index.tolist()
    return recommendations

# Segmentation function
def predict_cluster(r, f, m, model, scaler):
    data = pd.DataFrame([[r, f, m]], columns=["Recency", "Frequency", "Monetary"])
    scaled = scaler.transform(data)
    cluster = model.predict(scaled)[0]
    labels = {0: "High-Value", 1: "Regular", 2: "Occasional", 3: "At-Risk"}
    return labels.get(cluster, f"Cluster {cluster}")

# UI starts here
st.set_page_config(page_title="SmartShopper", layout="wide")
st.title("🛍️ SmartShopper - Personalized Recommendations & Customer Insights")

product_df = load_product_data()
rfm_df = load_rfm_data()
kmeans_model, scaler_model = load_models()

st.sidebar.header("🔍 Choose a Module")
option = st.sidebar.radio("Select Module:", ["Product Recommendation", "Customer Segmentation"])

if option == "Product Recommendation":
    st.subheader("🎯 Product Recommendation")
    product_list = sorted(product_df['Description'].dropna().unique().tolist())
    selected_product = st.selectbox("Choose a product to get similar recommendations:", product_list)

    if st.button("🔎 Get Recommendations"):
        with st.spinner("Finding similar products..."):
            recs = get_recommendations(selected_product, product_df)
        st.success("Here are 5 similar products:")
        for i, rec in enumerate(recs, 1):
            st.markdown(f"**{i}. {rec}**")

elif option == "Customer Segmentation":
    st.subheader("📊 Customer Segmentation")
    st.markdown("Input your **Recency**, **Frequency**, and **Monetary** values:")

    recency = st.number_input("Recency (days since last purchase)", min_value=0, value=30)
    frequency = st.number_input("Frequency (number of purchases)", min_value=0, value=5)
    monetary = st.number_input("Monetary (total spend)", min_value=0.0, value=200.0, step=10.0)

    if st.button("📈 Predict Cluster"):
        with st.spinner("Analyzing customer segment..."):
            segment = predict_cluster(recency, frequency, monetary, kmeans_model, scaler_model)
        st.success(f"🧠 Predicted Segment: **{segment}**")

# Clean up extracted temp files
if os.path.exists("temp_data"):
    import shutil
    shutil.rmtree("temp_data")
