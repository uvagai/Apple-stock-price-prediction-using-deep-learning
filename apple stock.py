#!/usr/bin/env python
# coding: utf-8

# In[7]:


get_ipython().system('pip install tensorflow')



# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout, Input, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error


# In[2]:


# Set a global seed value
SEED = 42

# For reproducibility
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)


# __Data preprocessing__

# In[4]:


df=pd.read_csv("AAPL.csv")


# In[8]:


df.info()


# In[10]:


df.head()


# In[12]:


df.isna().sum()


# In[14]:


df=df.ffill()


# __Feature selection__

# In[17]:


# Convert the Date column to a datetime format and set it as the index.
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)


# In[19]:


df.shape


# In[21]:


data = df[['Adj Close']].values


# In[23]:


plt.figure(figsize=(12, 6))
plt.plot(df['Adj Close'], label='Adj Close Price')
plt.title('Apple Stock - Adjusted Close Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()


# __normalization__

# In[26]:


scaler = MinMaxScaler()
df['scaled_close'] = scaler.fit_transform(df[['Adj Close']])


# __Create sequences__

# In[29]:


def create_sequences(data, window_size, future_step=1):
    X, y = [], []
    for i in range(len(data) - window_size - future_step + 1):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size+future_step-1])
    return np.array(X), np.array(y)

window_size = 45
X_1, y_1 = create_sequences(df['scaled_close'].values, window_size, future_step=1)
X_5, y_5 = create_sequences(df['scaled_close'].values, window_size, future_step=5)
X_10, y_10 = create_sequences(df['scaled_close'].values, window_size, future_step=10)



# __split datset__

# In[32]:


from sklearn.model_selection import train_test_split

X1_train, X1_test, y1_train, y1_test = train_test_split(X_1, y_1, test_size=0.2, shuffle=False)
X5_train, X5_test, y5_train, y5_test = train_test_split(X_5, y_5, test_size=0.2, shuffle=False)
X10_train, X10_test, y10_train, y10_test = train_test_split(X_10, y_10, test_size=0.2, shuffle=False)


# __model development__

# In[35]:


def create_rnn_model(units=64, dropout=0.2, lr=0.001):
    model = Sequential([
        SimpleRNN(units, activation='tanh', input_shape=(window_size, 1)),
        Dropout(dropout),
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mse')
    return model

def create_lstm_model(units=128, dropout=0.2, lr=0.001):
    model = Sequential([
        LSTM(units, activation='tanh', input_shape=(window_size, 1)),
        Dropout(dropout),
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mse')
    return model


# __Train and evaluation__

# In[38]:


def train_and_evaluate(X_train, y_train, X_test, y_test, time_horizon):
    X_train = X_train.reshape(-1, window_size, 1)
    X_test = X_test.reshape(-1, window_size, 1)

    rnn = create_rnn_model()
    lstm = create_lstm_model()

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    rnn_ckpt_path = f"rnn_best_model_{time_horizon}.keras"
    lstm_ckpt_path = f"lstm_best_model_{time_horizon}.keras"

    rnn_checkpoint = ModelCheckpoint(rnn_ckpt_path, save_best_only=True, monitor='val_loss')
    lstm_checkpoint = ModelCheckpoint(lstm_ckpt_path, save_best_only=True, monitor='val_loss')

    rnn.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1,
            callbacks=[early_stop, rnn_checkpoint], shuffle=False, verbose=0)

    lstm.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1,
             callbacks=[early_stop, lstm_checkpoint], shuffle=False, verbose=0)

    rnn.load_weights(rnn_ckpt_path)
    lstm.load_weights(lstm_ckpt_path)

    rnn_preds = rnn.predict(X_test)
    lstm_preds = lstm.predict(X_test)

    rnn_preds_inv = scaler.inverse_transform(rnn_preds)
    lstm_preds_inv = scaler.inverse_transform(lstm_preds)
    actual_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

    rnn_mse = mean_squared_error(actual_inv, rnn_preds_inv)
    lstm_mse = mean_squared_error(actual_inv, lstm_preds_inv)
    rnn_rmse = np.sqrt(rnn_mse)
    lstm_rmse = np.sqrt(lstm_mse)

    print(f"{time_horizon}-Day Prediction RNN -> MSE: {rnn_mse:.2f}, RMSE: {rnn_rmse:.2f}")
    print(f"{time_horizon}-Day Prediction LSTM -> MSE: {lstm_mse:.2f}, RMSE: {lstm_rmse:.2f}")

    plt.figure(figsize=(10, 5))
    plt.plot(actual_inv, label='Actual')
    plt.plot(rnn_preds_inv, label='RNN Predictions')
    plt.plot(lstm_preds_inv, label='LSTM Predictions')
    plt.title(f'{time_horizon}-Day Ahead Prediction')
    plt.legend()
    plt.show()


# In[40]:


from sklearn.model_selection import train_test_split

window_size = 45

for horizon in [1, 5, 10]:
    print(f"\n{'='*10} {horizon}-Day Forecast {'='*10}")
    X, y = create_sequences(df['scaled_close'].values, window_size, future_step=horizon)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    train_and_evaluate(X_train, y_train, X_test, y_test, time_horizon=horizon)


# In[42]:


train_and_evaluate(X1_train, y1_train, X1_test, y1_test, time_horizon=1)


# __hyper parameter tuning__

# In[44]:


def manual_lstm_tuning(X_train, y_train, window_size, name=""):
    X_train = X_train.reshape(-1, window_size, 1)

    # Simple train/val split
    split_idx = int(0.8 * len(X_train))
    X_tr, X_val = X_train[:split_idx], X_train[split_idx:]
    y_tr, y_val = y_train[:split_idx], y_train[split_idx:]

    # Hyperparameters to test
    units_list = [32, 50, 64, 128]
    dropout_list = [0.2, 0.3]
    lr_list = [0.001, 0.005]
    batch_size_list = [32]
    activation_list = ['relu', 'tanh', 'sigmoid']
    epochs = 100

    best_mse = float('inf')
    best_model = None
    best_params = {}

    for units in units_list:
        for dropout in dropout_list:
            for lr in lr_list:
                for batch_size in batch_size_list:
                    for func in activation_list:

                        print(f"\n🔄 {name} | Trying LSTM: activation Function = {func} units={units}, dropout={dropout}, lr={lr}")
                        model = Sequential([
                            LSTM(units, activation=func, input_shape=(window_size, 1)),
                            Dropout(dropout),
                            Dense(1)
                        ])
                        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mse')

                        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                        checkpoint = ModelCheckpoint(f"best_lstm_{name}.keras", monitor='val_loss', save_best_only=True)

                        model.fit(
                            X_tr, y_tr,
                            validation_data=(X_val, y_val),
                            epochs=epochs,
                            batch_size=batch_size,
                            callbacks=[early_stop, checkpoint],
                            verbose=0
                        )

                        val_preds = model.predict(X_val)
                        val_mse = mean_squared_error(y_val, val_preds)

                        if val_mse < best_mse:
                            best_mse = val_mse
                            best_model = model
                            best_params = {
                                'units': units,
                                'dropout': dropout,
                                'lr': lr,
                                'batch_size': batch_size
                            }
                            print(f"✅ New Best MSE for {name}: {val_mse:.4f} with {best_params}")
                        else:
                            print(f"✖ MSE: {val_mse:.4f}")

    return best_model, best_params, best_mse


# In[46]:


best_lstm_1d, best_params_1d, val_mse_1d = manual_lstm_tuning(X1_train, y1_train, window_size, name="1day")
best_lstm_5d, best_params_5d, val_mse_5d = manual_lstm_tuning(X5_train, y5_train, window_size, name="5day")
best_lstm_10d, best_params_10d, val_mse_10d = manual_lstm_tuning(X10_train, y10_train, window_size, name="10day")


# In[48]:


def evaluate_lstm(model, X_test, y_test, scaler, window_size, name=""):
    X_test = X_test.reshape(-1, window_size, 1)
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    mse = mean_squared_error(actual, predictions)
    rmse = np.sqrt(mse)

    print(f"\n📊 {name} Prediction:")
    print(f"Test MSE: {mse:.2f}")
    print(f"Test RMSE: {rmse:.2f}")

    plt.figure(figsize=(10, 5))
    plt.plot(actual, label='Actual')
    plt.plot(predictions, label='LSTM Predictions')
    plt.title(f'{name} Prediction')
    plt.legend()
    plt.show()

    return mse, rmse


# In[50]:


# Evaluate all models
evaluate_lstm(best_lstm_1d, X1_test, y1_test, scaler, window_size, name="1-Day")
evaluate_lstm(best_lstm_5d, X5_test, y5_test, scaler, window_size, name="5-Day")
evaluate_lstm(best_lstm_10d, X10_test, y10_test, scaler, window_size, name="10-Day")


# In[ ]:




