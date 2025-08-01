import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# Load preprocessed data and models
product_pivot = pd.read_pickle("data/product_pivot.pkl")  # Item-user matrix
product_names = list(product_pivot.index)
rfm_model = joblib.load("models/rfm_model.pkl")  # KMeans or similar clustering model

st.set_page_config(page_title="E-Commerce Insights App", layout="centered")
st.title("🛍️ E-Commerce Insights")

# --- TABS ---
tabs = st.tabs(["📦 Product Recommendation", "🧮 Customer Segmentation"])

# -----------------------------------
# 📦 PRODUCT RECOMMENDATION TAB
# -----------------------------------
with tabs[0]:
    st.header("🔎 Find Similar Products")
    product_input = st.text_input("Enter a product name:")

    if st.button("Get Recommendations"):
        if product_input not in product_names:
            st.warning("Product not found. Please enter an exact product name from the dataset.")
        else:
            # Compute cosine similarity
            similarity_scores = cosine_similarity(product_pivot.loc[[product_input]], product_pivot)[0]
            similar_indices = similarity_scores.argsort()[::-1][1:6]
            similar_products = [product_pivot.index[i] for i in similar_indices]

            st.success("Top 5 similar products:")
            for prod in similar_products:
                st.markdown(f"- {prod}")

# -----------------------------------
# 🧮 CUSTOMER SEGMENTATION TAB
# -----------------------------------
with tabs[1]:
    st.header("👤 Predict Customer Cluster")

    recency = st.number_input("Recency (days since last purchase):", min_value=0, max_value=1000, step=1)
    frequency = st.number_input("Frequency (number of purchases):", min_value=0, max_value=1000, step=1)
    monetary = st.number_input("Monetary (total spend):", min_value=0.0, step=10.0)

    if st.button("Predict Cluster"):
        rfm_values = np.array([[recency, frequency, monetary]])
        cluster_label = int(rfm_model.predict(rfm_values)[0])

        segment_map = {
            0: "🟢 High-Value",
            1: "🟡 Regular",
            2: "🔵 Occasional",
            3: "🔴 At-Risk"
        }

        label_display = segment_map.get(cluster_label, f"Cluster {cluster_label}")
        st.success(f"Predicted Customer Segment: {label_display}")
