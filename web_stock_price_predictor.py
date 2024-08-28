import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

st.markdown("""
    <style>
        .main {
            background-color: #f5f5f5;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #4A4A4A;
            text-align: center;
        }
        .stTextInput input {
            font-size: 1.2rem;
            color: #000;
            border-radius: 10px;
            padding: 8px;
            border: 2px solid #4A4A4A;
        }
        .stButton button {
            background-color: #4A90E2;
            color: white;
            border-radius: 10px;
            padding: 10px 20px;
            font-size: 1.1rem;
            border: none;
        }
        .stDataFrame, .stTable {
            background-color: white;
            border-radius: 10px;
            padding: 10px;
            box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
        }
    </style>
""", unsafe_allow_html=True)

# Title of the app
st.title("📈 Stock Price Predictor")

# Input section
st.subheader("Enter the Stock ID")
stock = st.text_input("Stock ID (e.g., GOOG)", "GOOG")

# Fetching data
from datetime import datetime
end = datetime.now()
start = datetime(end.year-20, end.month, end.day)

google_data = yf.download(stock, start, end)

# Loading the pre-trained model
model = load_model("Latest_stock_price_model.keras")

# Displaying the stock data
st.subheader("Stock Data")
st.dataframe(google_data)

# Moving Averages
google_data['MA_for_250_days'] = google_data.Close.rolling(250).mean()
google_data['MA_for_200_days'] = google_data.Close.rolling(200).mean()
google_data['MA_for_100_days'] = google_data.Close.rolling(100).mean()

# Function to plot graphs
def plot_graph(figsize, values, full_data, extra_data=0, extra_dataset=None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values, 'Orange')
    plt.plot(full_data.Close, 'b')
    if extra_data:
        plt.plot(extra_dataset, 'g')
    plt.title("Stock Price and Moving Averages")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.grid(True)
    return fig

st.subheader("Close Price and Moving Averages")
st.pyplot(plot_graph((15, 6), google_data['MA_for_250_days'], google_data))
st.pyplot(plot_graph((15, 6), google_data['MA_for_200_days'], google_data))
st.pyplot(plot_graph((15, 6), google_data['MA_for_100_days'], google_data))
st.pyplot(plot_graph((15, 6), google_data['MA_for_100_days'], google_data, 1, google_data['MA_for_200_days']))

# Scaling and preparing data for prediction
splitting_len = int(len(google_data) * 0.7)
x_test = pd.DataFrame(google_data.Close[splitting_len:])

if not x_test.empty:
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(x_test[['Close']])

    x_data = []
    y_data = []

    for i in range(100, len(scaled_data)):
        x_data.append(scaled_data[i-100:i])
        y_data.append(scaled_data[i])

    x_data, y_data = np.array(x_data), np.array(y_data)

    # Predictions
    predictions = model.predict(x_data)
    inv_predictions = scaler.inverse_transform(predictions)
    inv_y_test = scaler.inverse_transform(y_data)

    # Plotting original vs predicted values
    plotting_data = pd.DataFrame(
        {
            'Original Test Data': inv_y_test.reshape(-1),
            'Predictions': inv_predictions.reshape(-1)
        },
        index = google_data.index[splitting_len+100:]
    )

    st.subheader("Original vs Predicted Values")
    st.line_chart(plotting_data)

    # Plotting full data with predictions
    st.subheader('Close Price: Original vs Predicted')
    fig = plt.figure(figsize=(15, 6))
    plt.plot(pd.concat([google_data.Close[:splitting_len+100], plotting_data]))
    plt.legend(["Data Not Used", "Original Test Data", "Predicted Test Data"])
    plt.grid(True)
    st.pyplot(fig)
else:
    st.warning("Not enough data for predictions. Please check the stock ID or try another stock.")
