# stock-prediction-using-LSTM-code
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Load the stock price data
data = pd.read_csv('stock_prices.csv')
prices = data['Close'].values.reshape(-1, 1)

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(prices)

# Split the data into training and test sets
train_size = int(len(scaled_prices) * 0.8)
train_data = scaled_prices[:train_size, :]
test_data = scaled_prices[train_size:, :]

# Create the input and output sequences
def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 30
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))

# Compile and train the model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Make predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Transform the predictions back to original scale
train_predictions = scaler.inverse_transform(train_predictions)
y_train = scaler.inverse_transform(y_train)
test_predictions = scaler.inverse_transform(test_predictions)
y_test = scaler.inverse_transform(y_test)

# Plot the predictions
train_plot = np.empty_like(prices)
train_plot[:,:] = np.nan
train_plot[seq_length:len(train_predictions)+seq_length,:] = train_predictions

test_plot = np.empty_like(prices)
test_plot[:,:] = np.nan
test_plot[len(train_predictions)+(seq_length*2)+1:len(prices)-1,:] = test_predictions

plt.plot(prices, color='blue', label='Actual Prices')
plt.plot(train_plot, color='green', label='Training Predictions')
plt.plot(test_plot, color='red', label='Testing Predictions')
plt.legend()
plt.show()
