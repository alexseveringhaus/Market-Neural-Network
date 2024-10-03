import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dense
import yfinance as yf
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.layers import Dropout

def get_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
    data['Returns'] = np.log(data['Adj Close'] / data['Adj Close'].shift(1))
    return data

"""def build_model(input_shape):
    model = Sequential()
    model.add(Dense(32, input_dim=input_shape, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(), loss='binary_crossentropy')
    return model"""

def run_strategy(data, model, X, transaction_cost=0.0):
    data['Prediction'] = model.predict(X).round()
    data['Strategy'] = data['Prediction'].shift(1) * data['Returns']
    data['Cumulative_Strategy'] = (data['Strategy'] - transaction_cost).cumsum().apply(np.exp)
    data['Cumulative_Market'] = data['Returns'].cumsum().apply(np.exp)

    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['Cumulative_Strategy'], label='Strategy')
    plt.plot(data.index, data['Cumulative_Market'], label='Market')
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

def prepare_features(data, lags=20):
    for lag in range(1, lags + 1):
        data[f'Lag_{lag}'] = data['Returns'].shift(lag)
    data['Rolling_Mean'] = data['Returns'].rolling(window=5).mean()
    data['Volatility'] = data['Returns'].rolling(window=5).std()
    data.dropna(inplace=True)
    scaler = StandardScaler()
    X = scaler.fit_transform(data[[f'Lag_{i}' for i in range(1, lags + 1)] + ['Rolling_Mean', 'Volatility']])
    return X, data['Returns']

def fit_model(X_train, y_train):
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
    return model

def execute_strategy(symbol='AAPL', start_date='2022-01-01', end_date='2024-01-01', lags=3, transaction_cost=0.001):
    data = get_data(symbol, start_date, end_date)
    X, y = prepare_features(data, lags)
    y_sign = np.where(y > 0, 1, 0)  # Classify returns as 1 (up) or 0 (down)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y_sign, test_size=0.2, shuffle=False)
    
    model = fit_model(X_train, y_train)
    
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    print(classification_report(y_test, y_pred))
    
    run_strategy(data.iloc[len(X_train):], model, X_test, transaction_cost)

execute_strategy()
