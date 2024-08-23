# Stock Price Prediction Using LSTM

This repository contains the code and resources for predicting stock prices using Long Short-Term Memory (LSTM) neural networks. The goal of this project is to leverage historical stock market data to forecast future stock prices, enabling informed investment decisions.

## Overview
In this project, we utilize an LSTM model, a type of recurrent neural network (RNN) that is particularly effective for time series forecasting. The model learns from past price patterns and trends, allowing it to predict future stock prices with improved accuracy.

## Dataset
The dataset used in this project consists of historical stock prices for Yahoo. It includes features such as opening price, closing price, high, low, and volume. The data is preprocessed to split it into training and testing sets, and necessary transformations are applied to prepare it for model training.

## Model Training
The LSTM model is built using TensorFlow and Keras. Key steps in the model training process include:
- Data Preprocessing: Normalizing the data using MinMaxScaler and splitting it into training and testing datasets.
- Model Architecture: Designing the LSTM model with appropriate layers, including LSTM layers and dropout layers to prevent overfitting.
- Hyperparameter Tuning: Adjusting parameters such as the number of epochs, batch size, and learning rate for optimal performance.
  
## Evaluation and Results
After training the model, its performance is evaluated on the testing dataset. We compute metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE) to assess accuracy. The results are visualized to compare predicted stock prices against actual prices, providing insights into the model's effectiveness.

![image](https://github.com/user-attachments/assets/d717a3ae-d0f7-4c53-b5ed-52bc13ae4ac6)
