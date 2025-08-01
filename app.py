import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import joblib
import zipfile
import os

# ---------- PAGE SETUP ----------
st.set_page_config(page_title="SmartShopper 🛍️", layout="centered")
st.title("🛍️ SmartShopper")
st.markdown("#### Your Personal Assistant for Shopping and Customer Insights")


# ---------- LOAD MODELS ----------
@st.cache_resource
def load_kmeans_model():
    return joblib.load("kmeans_model.joblib")

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.joblib")

@st.cache_data
def load_rfm_data():
    return pd.read_csv("rfm_clustered.csv")


# ---------- TAB LAYOUT ----------
tab1, tab2 = st.tabs(["📦 Product Recommender", "👥 Customer Segmentation"])

# ========== TAB 1: PRODUCT RECOMMENDER ==========
with tab1:
    st.subheader("📦 Get Product Recommendations")
    uploaded_file = st.file_uploader("Upload product_data.zip", type="zip")

    if uploaded_file:
        with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
            zip_ref.extractall("product_data")
        try:
            product_df = pd.read_csv("product_data/product_data.csv")
            st.success("Data loaded successfully!")
        except Exception as e:
            st.error(f"Error loading CSV: {e}")
            st.stop()

        # Pivot Table: Product-user matrix
        df_pivot = product_df.pivot_table(index="Description", columns="CustomerID", values="Quantity", aggfunc="sum").fillna(0)

        # Cosine similarity matrix
        similarity = cosine_similarity(df_pivot)

        # Map for index
        product_names = df_pivot.index.tolist()
        product_map = {name: idx for idx, name in enumerate(product_names)}

        product_input = st.text_input("Enter a product name (case-sensitive)", "")
        if st.button("🔍 Get Recommendations"):
            if product_input in product_map:
                idx = product_map[product_input]
                sim_scores = list(enumerate(similarity[idx]))
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
                top_5 = [product_names[i] for i, score in sim_scores[1:6]]

                st.markdown("### 🧠 You might also like:")
                for i, rec in enumerate(top_5, 1):
                    st.markdown(f"**{i}. {rec}**")
            else:
                st.warning("Product not found. Try with a valid product name from the dataset.")

# ========== TAB 2: CUSTOMER SEGMENTATION ==========
with tab2:
    st.subheader("👥 RFM-based Customer Segmentation")

    col1, col2, col3 = st.columns(3)
    with col1:
        recency = st.number_input("Recency (days)", min_value=0, value=30)
    with col2:
        frequency = st.number_input("Frequency", min_value=1, value=5)
    with col3:
        monetary = st.number_input("Monetary Value", min_value=1, value=100)

    if st.button("🎯 Predict Cluster"):
        kmeans = load_kmeans_model()
        scaler = load_scaler()

        input_df = pd.DataFrame([[recency, frequency, monetary]], columns=["Recency", "Frequency", "Monetary"])
        input_scaled = scaler.transform(input_df)
        cluster = kmeans.predict(input_scaled)[0]

        cluster_labels = {
            0: "🎯 High-Value Customer",
            1: "🔁 Regular Buyer",
            2: "🧊 Occasional Buyer",
            3: "⚠️ At-Risk Customer"
        }

        st.success(f"Segment: {cluster_labels.get(cluster, 'Unknown')}")
        st.markdown("Use this insight to personalize marketing and sales strategies!")

# ---------- FOOTER ----------
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | SmartShopper 2025")

