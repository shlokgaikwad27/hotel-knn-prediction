import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/hotel_bookings.csv')
df = df.fillna(0)

# Load model
model = pickle.load(open('model.pkl','rb'))
scaler = pickle.load(open('scaler.pkl','rb'))

st.title("🏨 Hotel Booking Cancellation Prediction")
st.subheader("📊 Data Insights")

lead_time = st.slider("Lead Time", 0, 365)
adr = st.number_input("Price per Night")
adults = st.number_input("Adults", 1, 10)
children = st.number_input("Children", 0, 10)
prev_cancel = st.number_input("Previous Cancellations", 0, 10)

if st.button("Predict"):
    data = np.array([[lead_time, adr, adults, children, prev_cancel]])
    data = scaler.transform(data)

    result = model.predict(data)
    proba = model.predict_proba(data)

    if result[0] == 1:
        st.error("⚠️ Booking Likely to be Cancelled")
    else:
        st.success("✅ Booking Likely to be Confirmed")

    st.write(f"Cancellation Probability: {proba[0][1]*100:.2f}%")
st.set_page_config(page_title="Hotel Prediction", layout="centered")

st.title("🏨 Hotel Booking Cancellation Predictor")
st.markdown("### Enter booking details below")

st.divider()

proba = model.predict_proba(data)

st.write(f"Cancellation Probability: {proba[0][1]*100:.2f}%")

st.info("💡 High lead time increases cancellation chances")

if adr <= 0:
    st.warning("Price must be greater than 0")

fig1, ax1 = plt.subplots()
df['is_canceled'].value_counts().plot(kind='bar', ax=ax1)
ax1.set_title("Cancellation Distribution")
st.pyplot(fig1)

fig2, ax2 = plt.subplots()
ax2.scatter(df['lead_time'], df['is_canceled'])
ax2.set_title("Lead Time vs Cancellation")
ax2.set_xlabel("Lead Time")
ax2.set_ylabel("Cancelled (0/1)")
st.pyplot(fig2)

fig3, ax3 = plt.subplots()
df['adr'].hist(ax=ax3)
ax3.set_title("Price Distribution (ADR)")
st.pyplot(fig3)