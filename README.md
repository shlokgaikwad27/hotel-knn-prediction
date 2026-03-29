# 🏨 Hotel Booking Cancellation Prediction using KNN

## 📌 Overview
This project predicts whether a hotel reservation will be **cancelled or not** using a Machine Learning model (K-Nearest Neighbors).

It also includes a **Streamlit web application** for real-time prediction and data visualization.

---

## 🎯 Problem Statement
To build a predictive system that helps hotels identify bookings that are likely to be canceled, enabling better revenue management and planning.

---

## 📊 Dataset
- Hotel Booking Demand Dataset (Kaggle)
- Contains booking details, pricing, and customer behavior

---

## ⚙️ Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn (KNN)
- Streamlit
- Matplotlib

---

## 🔍 Features Used
- Lead Time
- Average Daily Rate (ADR)
- Number of Adults
- Number of Children
- Previous Cancellations

---

## 🧠 Model
- Algorithm: K-Nearest Neighbors (KNN)
- Feature Scaling: StandardScaler
- Hyperparameter: K = 5

---

## 📈 Results
- Accuracy: ~80%
- Key Insight: Higher lead time increases cancellation probability

---

## 🌐 Streamlit App

### 🔹 Input Interface
Users can enter:
- Lead Time
- Price (ADR)
- Adults
- Children
- Previous Cancellations

### 🔹 Output
- Prediction: Cancelled / Not Cancelled
- Probability Score

---

## 📸 App Screenshots

### 🔹 Home Page
![Home](images/home.png)

### 🔹 Prediction Output
![Prediction](images/prediction.png)

### 🔹 Data Insights Graphs
![Graphs](images/graphs.png)

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/hotel-knn-prediction.git
cd hotel-knn-prediction
