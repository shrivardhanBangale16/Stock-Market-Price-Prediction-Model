import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model # type: ignore
import streamlit as st
import matplotlib.pyplot as plt

model = load_model('model = load_model('Stock Predictions Model.keras')

import streamlit as st

# Background image URL (from your link)
background_url = "https://png.pngtree.com/background/20250116/original/pngtree-stock-market-analysis-with-colorful-candlestick-chart-picture-image_16020049.jpg"

# Apply background image using custom CSS
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("{background_url}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    /* Optional: Add a white transparent overlay for better text readability */
    .stApp > .main {{
        background-color: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(3px);
        padding: 1rem;
        border-radius: 10px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Sample content to test background
st.title("📈 Stock Market Prediction App")
st.write("Welcome to the prediction dashboard. Select a stock and get forecast insights!")

# Add more of your Streamlit components below...


st.header('Stock Market Price Predictor')

stocks=("RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS ",
        "HINDUNILVR.NS","LT.NS","BHARTIAIRTEL.NS","BAJFINANCE.NS","AXISBANK.NS",
        "MARUTI.NS","TATAMOTORS.NS","TECHM.N","SBIN.NS","KOTAKBANK.NS","M&M.NS",
        "TITAN.NS","ITC.NS","ADANIENT.NS","ULTRACEMCO.NS"
        "AAPL","MSFT ","GOOGL","AMZN","TSLA","NVDA","META","BRK-B","JPM","KO","INTC","NFLX","WMT","MCD")

stock =st.selectbox('Enter Stock from dataset',stocks)
start = '2012-01-01'
end = '2022-12-31'

data = yf.download(stock, start ,end)

st.subheader('Stock Data') #displays the data in pandas dataframe table
st.write(data)

data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

pas_100_days = data_train.tail(100)#past hundred days Data from Training Set
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

st.subheader('Price vs MA50')#ma=moving average for 50 days
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig1)

st.subheader('Price vs MA50 vs MA100')
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r')
plt.plot(ma_100_days, 'b')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig2)

st.subheader('Price vs MA100 vs MA200')
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8,6))
plt.plot(ma_100_days, 'r')
plt.plot(ma_200_days, 'b')
plt.plot(data.Close, 'g',label='Close Price')
plt.show()
st.pyplot(fig3)

x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])

x,y = np.array(x), np.array(y)

predict = model.predict(x)

scale = 1/scaler.scale_

predict = predict * scale
y = y * scale

st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(8,6))
plt.plot(predict, 'r',label='Original Price')
plt.plot(y, 'g',label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()
st.pyplot(fig4)
