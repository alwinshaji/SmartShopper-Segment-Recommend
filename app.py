import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
import os
import shutil
from joblib import load

# Set page config
st.set_page_config(
    page_title="SmartShopper Segment & Recommend",
    layout="wide",
    page_icon="🛍️"
)

# App title and header
st.markdown("""
    <h1 style='text-align: center; color: #2E86C1;'>🛍️ SmartShopper: Segment & Recommend</h1>
    <h4 style='text-align: center; color: #117A65;'>Know your customer. Serve better. Sell smart.</h4>
    <hr style='border: 1px solid #D5DBDB;'>
""", unsafe_allow_html=True)

# Load the model
model = load("kmeans_model.joblib")

# Load RFM clustered data
rfm_df = pd.read_csv("rfm_clustered.csv")

# Load and extract product data
@st.cache_data

def load_product_data():
    zip_path = "product_data.zip"
    extract_dir = "temp_data"

    if not os.path.exists(extract_dir):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

    for file in os.listdir(extract_dir):
        if file.endswith(".csv"):
            product_df = pd.read_csv(os.path.join(extract_dir, file))
            return product_df
    return pd.DataFrame()

product_df = load_product_data()

# Color palette
cluster_colors = ['#58D68D', '#F4D03F', '#5DADE2', '#EC7063']

# Tabs
tabs = st.tabs(["📊 Segment Summary", "👤 Customer Prediction", "🎯 Product Recommendations"])

# --- Tab 1: Segment Summary --- #
with tabs[0]:
    st.markdown("## 🔍 Customer Segment Summary")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(x='Cluster', data=rfm_df, palette=cluster_colors, ax=ax)
    ax.set_title("Customer Distribution by Cluster", fontsize=14)
    st.pyplot(fig)

    st.markdown("### 📈 Segment Characteristics")
    seg_stats = rfm_df.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean().round(2)
    st.dataframe(seg_stats.style.background_gradient(cmap='YlGnBu'))

# --- Tab 2: Prediction --- #
with tabs[1]:
    st.markdown("## 👤 Predict Customer Segment")
    with st.form("predict_form"):
        recency = st.number_input("Recency (days since last purchase):", min_value=0)
        frequency = st.number_input("Frequency (no. of purchases):", min_value=0)
        monetary = st.number_input("Monetary (total amount spent):", min_value=0.0)
        submitted = st.form_submit_button("Predict Segment")

        if submitted:
            cluster = model.predict([[recency, frequency, monetary]])[0]
            st.success(f"Predicted Customer Segment: {cluster}")
            st.markdown(f"<p style='color:{cluster_colors[cluster]}; font-size:18px;'>Segment {cluster}: Tailored marketing strategy applicable.</p>", unsafe_allow_html=True)

# --- Tab 3: Product Recommendations --- #
with tabs[2]:
    st.markdown("## 🎯 Get Product Recommendations")

    product_names = product_df['Description'].dropna().unique()
    selected_product = st.selectbox("Choose a product you like:", sorted(product_names))

    st.markdown("### 🛍️ Similar Products You Might Like")
    if selected_product:
        matched_products = product_df[product_df['Description'].str.contains(selected_product[:4], case=False, na=False)]
        recommendations = matched_products['Description'].dropna().unique().tolist()[:10]

        if recommendations:
            for rec in recommendations:
                st.markdown(f"- {rec}")
        else:
            st.warning("No similar products found. Try a broader product keyword.")

# Clean temp_data if it exists (safely)
if os.path.exists("temp_data"):
    try:
        shutil.rmtree("temp_data")
    except Exception as e:
        pass
