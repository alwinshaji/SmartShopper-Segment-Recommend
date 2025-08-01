# 🛍️ SmartShopper Segment Recommend

An interactive Streamlit web app that recommends relevant products to e-commerce customers based on **RFM (Recency, Frequency, Monetary)** segmentation using **KMeans clustering**.

🔗 **Live Demo:** [smartshopper-segment-recommend.streamlit.app](https://smartshopper-segment-recommend.streamlit.app)

---

## 📌 Project Overview

Understanding your customers is the foundation of any business. This app segments e-commerce users using RFM analysis and recommends products that fit their purchasing behavior.

### 🧠 What it does:
- Uses **RFM clustering** to classify customers into behavioral segments.
- Lets you input a `CustomerID` to get personalized product recommendations.
- Shows segmented purchase behavior visually.
- Built with `Streamlit`, powered by `KMeans`, and backed by real purchase data.

---

## 📂 Files in this Repo

| File / Folder         | Purpose |
|-----------------------|---------|
| `app.py`              | Main Streamlit app |
| `Ecommerce_Customer_Segmentation_RFM.ipynb` | Notebook for RFM analysis and model creation |
| `rfm_clustered.csv`   | Preprocessed customer RFM segments |
| `product_data.zip`    | Compressed original purchase dataset |
| `kmeans_model.joblib` | Trained clustering model |
| `scaler.joblib`       | Fitted scaler used in preprocessing |
| `requirements.txt`    | Python dependencies for the app |

---

## 🚀 Getting Started Locally

### 1. Clone the repository

bash
git clone https://github.com/alwinshaji/SmartShopper-Segment-Recommend.git
cd SmartShopper-Segment-Recommend

2. Install dependencies
It's recommended to use a virtual environment.
pip install -r requirements.txt

3. Run the app
streamlit run app.py

---

📊 Tech Stack

Frontend: Streamlit

Backend / ML: Python, Pandas, Scikit-learn, Joblib

Data: Online retail purchase data (CSV inside zip)

Model: KMeans clustering with standard scaler

---
🤝 Contributing
Found a bug or have a feature request? Feel free to open an issue or PR.

---

📜 License
This project is open source and available under the MIT License.

---
## 🙋‍♂️ Author
[**Alwin Shaji**](https://www.linkedin.com/in/alwnshaji)



















