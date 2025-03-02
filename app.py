import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest

# Load trained model and scaler
model = joblib.load("demand_forecast_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load dataset for visualization
df = pd.read_csv("Daily Demand Forecasting Orders.csv")
df = df.drop(columns=["Unnamed: 0"], errors='ignore')
df.columns = ["week", "day", "non_urgent", "urgent", "fiscal_orders", "traffic_orders", "total_orders"]

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Prediction", "Visualizations", "Forecasting & Anomalies"])

if page == "Home":
    st.title("Daily Demand Forecasting")
    st.image("forecasting_image.jpg", use_container_width=True)
    st.write("Welcome to the Daily Demand Forecasting App. This tool helps predict total orders based on past trends and key factors such as urgent and non-urgent orders.")

elif page == "Prediction":
    st.title("Predict Total Orders")
    week = st.number_input("Week of the month", min_value=1, max_value=5, step=1)
    day = st.number_input("Day of the week", min_value=1, max_value=7, step=1)
    non_urgent = st.number_input("Non-urgent orders", min_value=0.0, step=1.0)
    urgent = st.number_input("Urgent orders", min_value=0.0, step=1.0)
    fiscal_orders = st.number_input("Fiscal sector orders", min_value=0.0, step=1.0)
    traffic_orders = st.number_input("Orders from traffic controller sector", min_value=0, step=1)
    
    if st.button("Predict Demand"):
        if non_urgent == 0 and urgent == 0 and fiscal_orders == 0 and traffic_orders == 0:
            prediction = 0
        else:
            input_data = np.array([[week, day, non_urgent, urgent, fiscal_orders, traffic_orders]])
            input_data_scaled = scaler.transform(input_data)
            prediction = model.predict(input_data_scaled)[0]
        
        st.success(f"Predicted Total Orders: {prediction:.2f}")

elif page == "Visualizations":
    st.title("Data Visualizations")
    
    st.subheader("Total Orders Over Weeks (Line Plot)")
    fig, ax = plt.subplots()
    sns.lineplot(x=df["week"], y=df["total_orders"], marker="o", ax=ax)
    ax.set_xlabel("Week")
    ax.set_ylabel("Total Orders")
    st.pyplot(fig)
    
    st.subheader("Distribution of Total Orders (Box Plot)")
    fig, ax = plt.subplots()
    sns.boxplot(x=df["total_orders"], ax=ax)
    ax.set_xlabel("Total Orders")
    st.pyplot(fig)

elif page == "Forecasting & Anomalies":
    st.title("Forecasting & Anomaly Detection")
    
    # Forecasting next few weeks' demand using moving average
    df["moving_avg"] = df["total_orders"].rolling(window=3, min_periods=1).mean()
    st.subheader("Forecasted Demand Over Next Weeks")
    fig, ax = plt.subplots()
    sns.lineplot(x=df["week"], y=df["moving_avg"], marker="o", ax=ax, label="Forecast")
    sns.lineplot(x=df["week"], y=df["total_orders"], marker="o", ax=ax, label="Actual")
    ax.set_xlabel("Week")
    ax.set_ylabel("Total Orders")
    st.pyplot(fig)
    
    # Anomaly Detection using Isolation Forest
    st.subheader("Anomaly Detection in Orders")
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    df["anomaly"] = iso_forest.fit_predict(df[["total_orders"]])
    df["anomaly"] = df["anomaly"].map({1: "Normal", -1: "Anomaly"})
    
    fig, ax = plt.subplots()
    sns.scatterplot(x=df["week"], y=df["total_orders"], hue=df["anomaly"], palette={"Normal": "blue", "Anomaly": "red"}, ax=ax)
    ax.set_xlabel("Week")
    ax.set_ylabel("Total Orders")
    st.pyplot(fig)
    
    st.write("Red points indicate anomalies in order demand trends, suggesting unusual spikes or drops.")


    