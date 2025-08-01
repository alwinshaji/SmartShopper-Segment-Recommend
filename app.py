import streamlit as st
import pandas as pd
import joblib
import zipfile
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# -------- App Setup --------
st.set_page_config(page_title="SmartShopper", layout="centered")
st.title("🛒 SmartShopper")
st.markdown("### 💡 Intelligent Product Recommendation & Customer Segmentation")

# -------- Data Load --------
@st.cache_data
def load_product_data():
    if os.path.exists("product_data.csv"):
        return pd.read_csv("product_data.csv")
    elif os.path.exists("product_data.zip"):
        with zipfile.ZipFile("product_data.zip", 'r') as zip_ref:
            zip_ref.extractall()
        return pd.read_csv("product_data.csv")
    else:
        st.error("❌ Product data not found. Please upload `product_data.zip` containing `product_data.csv`.")
        return None

# Load Product Data
product_df = load_product_data()

# Load KMeans Model
@st.cache_resource
def load_model():
    return joblib.load("Kmeans_model.pkl")

model = load_model()

# -------- Tabs --------
tab1, tab2 = st.tabs(["🎯 Product Recommendation", "📊 Customer Segmentation"])

# -------- TAB 1: Product Recommendation --------
with tab1:
    st.subheader("🔍 Find Similar Products")

    if product_df is not None:
        product_list = product_df['Description'].dropna().unique()
        product_input = st.text_input("Enter a product name:", "")

        if st.button("Get Recommendations"):
            if product_input:
                vectorizer = CountVectorizer().fit_transform(product_list)
                input_vec = CountVectorizer().fit(product_list).transform([product_input])
                similarity = cosine_similarity(input_vec, vectorizer).flatten()

                top_indices = similarity.argsort()[-6:][::-1]  # top 5 + input itself
                recommendations = [product_list[i] for i in top_indices if product_list[i] != product_input][:5]

                if recommendations:
                    st.markdown("##### 🧠 Recommended Products:")
                    for rec in recommendations:
                        st.success(rec)
                else:
                    st.warning("No similar products found.")
            else:
                st.warning("Please enter a product name.")

# -------- TAB 2: Customer Segmentation --------
with tab2:
    st.subheader("👥 Segment a Customer Using RFM Values")

    recency = st.number_input("Recency (days since last purchase)", min_value=0)
    frequency = st.number_input("Frequency (number of purchases)", min_value=0)
    monetary = st.number_input("Monetary (total spend)", min_value=0.0, step=0.01)

    if st.button("Predict Cluster"):
        input_data = pd.DataFrame([[recency, frequency, monetary]],
                                  columns=["Recency", "Frequency", "Monetary"])
        cluster_label = model.predict(input_data)[0]

        segment_names = {
            0: "High-Value",
            1: "Regular",
            2: "Occasional",
            3: "At-Risk"
        }

        st.success(f"Predicted Segment: **{segment_names.get(cluster_label, 'Unknown')}**")
