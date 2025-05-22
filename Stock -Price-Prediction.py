#!/usr/bin/env python
# coding: utf-8

# In[2]:


import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


# In[4]:


# Download data from Yahoo Finance
stock = 'AAPL'
data = yf.download(stock, start='2015-01-01', end='2024-12-31')
data = data[['Close']]
data.head()


# In[6]:


# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Create sequences
X, y = [], []
sequence_length = 60

for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i, 0])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))


# In[8]:


split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]


# In[10]:


model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()


# In[12]:


model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))


# In[14]:


predicted_prices = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))


# In[16]:


plt.figure(figsize=(14,6))
plt.plot(actual_prices, color='black', label='Actual Prices')
plt.plot(predicted_prices, color='green', label='Predicted Prices')
plt.title(f'{stock} Stock Price Prediction using LSTM')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()


# In[18]:


from sklearn.metrics import mean_squared_error

rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
print(f'Root Mean Squared Error: {rmse}')


# In[ ]:




