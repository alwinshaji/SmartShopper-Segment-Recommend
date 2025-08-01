import streamlit as st
import pandas as pd
import joblib
import zipfile
import os
import requests
from io import BytesIO

# Set app title
st.set_page_config(page_title="SmartShopper", layout="centered")
st.title("🛍️ SmartShopper")

# Load and extract product data.zip from a hosted URL
@st.cache_data
def load_product_data():
    url = "https://github.com/alwnshaji/public-assets/raw/main/product_data.zip"  # replace with your actual public zip file link
    r = requests.get(url)
    z = zipfile.ZipFile(BytesIO(r.content))
    z.extractall("product_data")
    df = pd.read_csv("product_data/product_data.csv")
    return df

# Load models
@st.cache_resource
def load_models():
    kmeans_model = joblib.load("kmeans_model.joblib")
    scaler = joblib.load("scaler.joblib")
    return kmeans_model, scaler

# Load product data
product_df = load_product_data()

# Load joblib models
kmeans_model, scaler = load_models()

# Sidebar navigation
page = st.sidebar.radio("Choose Module", ["📦 Product Recommendation", "👥 Customer Segmentation"])

# 1️⃣ Product Recommendation Module
if page == "📦 Product Recommendation":
    st.subheader("🎯 Product Recommendation")

    product_names = sorted(product_df['ProductName'].unique())
    selected_product = st.selectbox("Choose a product", product_names)

    if st.button("🔍 Get Recommendations"):
        from sklearn.metrics.pairwise import cosine_similarity

        product_pivot = product_df.pivot_table(index='CustomerID', columns='ProductName', values='Rating').fillna(0)

        if selected_product not in product_pivot.columns:
            st.warning("Product not found in the dataset.")
        else:
            similarity_scores = cosine_similarity(product_pivot.T)
            product_index = list(product_pivot.columns).index(selected_product)
            similar_indices = similarity_scores[product_index].argsort()[::-1][1:6]
            recommended_products = [product_pivot.columns[i] for i in similar_indices]

            st.markdown("### 🧠 Recommended Products:")
            for i, prod in enumerate(recommended_products, 1):
                st.success(f"{i}. {prod}")

# 2️⃣ Customer Segmentation Module
elif page == "👥 Customer Segmentation":
    st.subheader("🎯 Customer Segmentation")

    col1, col2, col3 = st.columns(3)
    with col1:
        recency = st.number_input("Recency (days)", min_value=0)
    with col2:
        frequency = st.number_input("Frequency (purchases)", min_value=0)
    with col3:
        monetary = st.number_input("Monetary (₹)", min_value=0)

    if st.button("📊 Predict Cluster"):
        input_data = pd.DataFrame([[recency, frequency, monetary]], columns=['Recency', 'Frequency', 'Monetary'])
        input_scaled = scaler.transform(input_data)
        cluster = kmeans_model.predict(input_scaled)[0]

        labels = {
            0: "💎 High-Value",
            1: "🧍 Regular",
            2: "📉 At-Risk",
            3: "⏳ Occasional"
        }

        st.markdown("### 🧬 Customer Segment:")
        st.info(f"Predicted Segment: **{labels.get(cluster, 'Unknown')}**")
