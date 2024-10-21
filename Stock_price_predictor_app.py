# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 19:25:32 2024

@author: Sahil Kaladgi
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model


st.title("Stock Price Predictor App")

stock = st.text_input("Enter The Stock ID", "INFY.NS")

from datetime import datetime
end = datetime.now()
start = datetime(end.year-20,end.month,end.day)

infy_data = yf.download(stock, start, end)

lstm_model = load_model("C:/Users/91776/Downloads/stock_price_lstm_model.keras")
st.subheader("Stock Data")
st.write(infy_data)

splitting_len = int(len(infy_data)*0.7)
x_test = pd.DataFrame(infy_data.Close[splitting_len:])

def plot_graph(figsize, values, full_data, extra_data = 0, extra_dataset = None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values,'Orange')
    plt.plot(full_data.Close, 'b')
    if extra_data:
        plt.plot(extra_dataset)
    return fig

st.subheader('Original Close Price and MA for 250 days')
infy_data['250 Day MA'] = infy_data.Close.rolling(250).mean()
st.pyplot(plot_graph((15,6), infy_data['250 Day MA'],infy_data,0))

st.subheader('Original Close Price and MA for 50 days')
infy_data['50 Day MA'] = infy_data.Close.rolling(50).mean()
st.pyplot(plot_graph((15,6), infy_data['50 Day MA'],infy_data,0))


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(x_test[['Close']])

x_infy_data = []
y_infy_data = []

for i in range(100,len(scaled_data)):
    x_infy_data.append(scaled_data[i-100:i])
    y_infy_data.append(scaled_data[i])

x_infy_data, y_infy_data = np.array(x_infy_data), np.array(y_infy_data)

predictions = lstm_model.predict(x_infy_data)

inv_predictions = scaler.inverse_transform(predictions)
inv_y_infy_test = scaler.inverse_transform(y_infy_data)

ploting_data = pd.DataFrame(
 {
  'original_test_data': inv_y_infy_test.reshape(-1),
    'predictions': inv_predictions.reshape(-1)
 } ,
    index = infy_data.index[splitting_len+100:]
)
st.subheader("Original values vs Predicted values")
st.write(ploting_data)

st.subheader('Original Close Price vs Predicted Close price')
fig = plt.figure(figsize=(15,6))
plt.plot(pd.concat([infy_data.Close[:splitting_len+100],ploting_data], axis=0))
plt.legend(["Data not used", "Original Test data", "Predicted Test data"])
st.pyplot(fig)
