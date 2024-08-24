# Stock Price Prediction

This project implements a stock price prediction model using Long Short-Term Memory (LSTM) networks with Keras. The model predicts stock prices for various companies, including Google, Microsoft, HDFC, and Bitcoin. A web application built with Streamlit allows users to input a company stock ID and visualize the predicted stock data, comparing original prices against moving averages and predicted prices.

## Project Description
The Stock Price Prediction Model utilizes LSTM networks to forecast future stock prices based on historical data. The model is trained on past stock price data, allowing it to learn patterns and trends. The web application provides a user-friendly interface to visualize predictions, making it accessible for users interested in stock market trends.

## Steps Involved
- Data Collection: Historical stock price data is fetched from [Yahoo Finance](https://finance.yahoo.com/).
- Data Preprocessing: The data is cleaned and scaled using MinMaxScaler to prepare it for the LSTM model.
- Model Development: An LSTM model is built using Keras to predict future stock prices based on the training data.
- Visualization: Matplotlib is used to create graphs comparing original stock prices with predicted prices and moving averages.
- Web Application: A Streamlit app is developed to allow users to input stock IDs and visualize predictions.

## Technologies Used
- Python: The primary programming language used for the project.
- Keras: A high-level neural networks API, written in Python, used for building the LSTM model.
- TensorFlow: The backend for Keras, required for running the LSTM model.
- Pandas: Used for data manipulation and analysis.
- NumPy: Used for numerical operations on arrays.
- Matplotlib: A plotting library used for visualizing stock price trends.
- Streamlit: A framework for building web applications in Python.

## Installation and Running Locally
1. Clone the Repository:
   git clone https://github.com/codebybishwa/Stock-Price-Prediction

2. Save the Keras Model Locally: Before running the web app, ensure that you have trained and saved the Keras model.
  - Go to Stock_Price_Prediction code (or copy the code) and run it locally and ensure to save the keras model (model.save('Latest_stock_price_model.keras'))

4. Install Required Libraries:
   cd Stock-Price-Prediction
   pip install -r requirements.txt  (in terminal)

6. Run the Streamlit App:
   streamlit run web_stock_price_predictor.py
   Get the stock id from [Yahoo Finance](https://finance.yahoo.com/)

## Microsoft Stock Price Prediction (Sample Usage)
- Stock Data
  ![image](https://github.com/user-attachments/assets/d33756b1-3c31-47d8-b251-cc3df9ef93fc)

- Original Close Price and Moving average for 100 days
  ![image](https://github.com/user-attachments/assets/fbf6cdd2-9379-4a6f-9ed8-60027053b8fa)

- Original vs Predicted Values
  ![image](https://github.com/user-attachments/assets/48ebae61-4f8a-417f-9d6f-d746792272d4)
  ![image](https://github.com/user-attachments/assets/40502e72-1037-4c2c-bf0c-4f76bdcd93d9)






## Conclusion
This project demonstrates the effectiveness of LSTM networks in predicting stock prices based on historical data. The web application built with Streamlit provides an intuitive interface for users to interact with the model and visualize predictions.

