import mplfinance as mpf
import pandas as pd
import ta
import optuna
import time
import numpy as np
from multiprocessing import Pool
from itertools import combinations, chain
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import warnings
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")

def powerset(s):
    return chain.from_iterable(combinations(s,r) for r in range(1,len(s)+1))

def file_features_tr(data, ds_type: str):
    data1=pd.DataFrame()
    #Calcular indicadores tecnicos
    data1["Open"] = data["Open"]
    data1["Close"] = data["Close"]
    data1["High"] = data["High"]
    data1["Low"] = data["Low"]
    data1["Volume"] = data["Volume"]
    
    cmf_data = ta.volume.ChaikinMoneyFlowIndicator(data.High, data.Low, data.Close, data.Volume, window = 14)
    rsi_data = ta.momentum.RSIIndicator(data.Close, window=14)
    stoch_indicator = ta.momentum.StochasticOscillator(high=data.High, low=data.Low, close=data.Close, window=14, smooth_window=3)

    data1["CMF"] = cmf_data.chaikin_money_flow()
    data1["CMF_1"] = ta.volume.ChaikinMoneyFlowIndicator(data.High, data.Low, data.Close, data.Volume, window = 15).chaikin_money_flow()
    data1["CMF_2"] = ta.volume.ChaikinMoneyFlowIndicator(data.High, data.Low, data.Close, data.Volume, window = 16).chaikin_money_flow()
    data1["CMF_3"] = ta.volume.ChaikinMoneyFlowIndicator(data.High, data.Low, data.Close, data.Volume, window = 17).chaikin_money_flow()
    data1["CMF_4"] = ta.volume.ChaikinMoneyFlowIndicator(data.High, data.Low, data.Close, data.Volume, window = 18).chaikin_money_flow()
    data1["CMF_5"] = ta.volume.ChaikinMoneyFlowIndicator(data.High, data.Low, data.Close, data.Volume, window = 19).chaikin_money_flow()

    
    data1["RSI"] = rsi_data.rsi()
    data1["RSI_1"] = ta.momentum.RSIIndicator(data.Close, window=15).rsi()
    data1["RSI_2"] = ta.momentum.RSIIndicator(data.Close, window=16).rsi()
    data1["RSI_3"] = ta.momentum.RSIIndicator(data.Close, window=17).rsi()
    data1["RSI_4"] = ta.momentum.RSIIndicator(data.Close, window=18).rsi()
    data1["RSI_5"] = ta.momentum.RSIIndicator(data.Close, window=19).rsi()

    
    data1["MACD"] = ta.trend.macd(data.Close)
    data1["MACD_1"] = ta.trend.macd(data.Close, window_slow=26, window_fast=12)
    data1["MACD_2"] = ta.trend.macd(data.Close, window_slow=28, window_fast=14)
    data1["MACD_3"] = ta.trend.macd(data.Close, window_slow=30, window_fast=16)
    data1["MACD_4"] = ta.trend.macd(data.Close, window_slow=32, window_fast=18)
    data1["MACD_5"] = ta.trend.macd(data.Close, window_slow=34, window_fast=19)

    
    data1["SMA_20"] = ta.trend.sma_indicator(data.Close, window=20)
    data1["SMA_20_1"] = ta.trend.sma_indicator(data.Close, window=21)
    data1["SMA_20_2"] = ta.trend.sma_indicator(data.Close, window=22)
    data1["SMA_20_3"] = ta.trend.sma_indicator(data.Close, window=23)
    data1["SMA_20_4"] = ta.trend.sma_indicator(data.Close, window=24)
    data1["SMA_20_5"] = ta.trend.sma_indicator(data.Close, window=25)

    
    data1["ADX"] = ta.trend.adx(data.High, data.Low, data.Close, window=14)
    data1["ADX_1"] = ta.trend.adx(data.High, data.Low, data.Close, window=15)
    data1["ADX_2"] = ta.trend.adx(data.High, data.Low, data.Close, window=16)
    data1["ADX_3"] = ta.trend.adx(data.High, data.Low, data.Close, window=17)
    data1["ADX_4"] = ta.trend.adx(data.High, data.Low, data.Close, window=18)
    data1["ADX_5"] = ta.trend.adx(data.High, data.Low, data.Close, window=19)

    
    data1["CCI"] = ta.trend.cci(data.High, data.Low, data.Close, window=14)
    data1["CCI_1"] = ta.trend.cci(data.High, data.Low, data.Close, window=15)
    data1["CCI_2"] = ta.trend.cci(data.High, data.Low, data.Close, window=16)
    data1["CCI_3"] = ta.trend.cci(data.High, data.Low, data.Close, window=17)
    data1["CCI_4"] = ta.trend.cci(data.High, data.Low, data.Close, window=18)
    data1["CCI_5"] = ta.trend.cci(data.High, data.Low, data.Close, window=19)

    data1["SO"] = stoch_indicator.stoch()
    data1["OBV"] = ta.volume.on_balance_volume(data.Close, data.Volume)
    
    data1["ATR"] = ta.volatility.average_true_range(data.High, data.Low, data.Close, window=14)
    data1["ATR_1"] = ta.volatility.average_true_range(data.High, data.Low, data.Close, window=15)
    data1["ATR_2"] = ta.volatility.average_true_range(data.High, data.Low, data.Close, window=16)
    data1["ATR_3"] = ta.volatility.average_true_range(data.High, data.Low, data.Close, window=17)
    data1["ATR_4"] = ta.volatility.average_true_range(data.High, data.Low, data.Close, window=18)
    data1["ATR_5"] = ta.volatility.average_true_range(data.High, data.Low, data.Close, window=19)


    # Calcular la volatilidad
    data1['Volatility'] = data['High'] - data['Low']
    
    data1 = data1.drop(data1.index[:36])
    data1 = data1.drop(data1.index[-36:])
    data1.reset_index(drop=True, inplace=True)
    
    return data1

def model_transformer(data_train, data_test):
    train_mean = data_train.loc[:, ["Open", "High", "Low", "Close", "Volume", "CMF", "CMF_1", "CMF_2", "CMF_3", "CMF_4", "CMF_5", "RSI", "RSI_1", "RSI_2", "RSI_3", "RSI_4", "RSI_5", "MACD", "MACD_1", "MACD_2", "MACD_3", "MACD_4", "MACD_5", "SMA_20", "SMA_20_1", "SMA_20_2", "SMA_20_3", "SMA_20_4", "SMA_20_5", "ADX", "ADX_1", "ADX_2", "ADX_3", "ADX_4", "ADX_5", "CCI", "CCI_1", "CCI_2", "CCI_3", "CCI_4", "CCI_5", "ATR", "ATR_1", "ATR_2", "ATR_3", "ATR_4", "ATR_5", "SO", "OBV", "Volatility"]].mean()
    train_std = data_train.loc[:, ["Open", "High", "Low", "Close", "Volume", "CMF", "CMF_1", "CMF_2", "CMF_3", "CMF_4", "CMF_5", "RSI", "RSI_1", "RSI_2", "RSI_3", "RSI_4", "RSI_5", "MACD", "MACD_1", "MACD_2", "MACD_3", "MACD_4", "MACD_5", "SMA_20", "SMA_20_1", "SMA_20_2", "SMA_20_3", "SMA_20_4", "SMA_20_5", "ADX", "ADX_1", "ADX_2", "ADX_3", "ADX_4", "ADX_5", "CCI", "CCI_1", "CCI_2", "CCI_3", "CCI_4", "CCI_5", "ATR", "ATR_1", "ATR_2", "ATR_3", "ATR_4", "ATR_5", "SO", "OBV", "Volatility"]].std()

    norm_data_train = (data_train.loc[:, ["Open", "High", "Low", "Close", "Volume", "CMF", "CMF_1", "CMF_2", "CMF_3", "CMF_4", "CMF_5", "RSI", "RSI_1", "RSI_2", "RSI_3", "RSI_4", "RSI_5", "MACD", "MACD_1", "MACD_2", "MACD_3", "MACD_4", "MACD_5", "SMA_20", "SMA_20_1", "SMA_20_2", "SMA_20_3", "SMA_20_4", "SMA_20_5", "ADX", "ADX_1", "ADX_2", "ADX_3", "ADX_4", "ADX_5", "CCI", "CCI_1", "CCI_2", "CCI_3", "CCI_4", "CCI_5", "ATR", "ATR_1", "ATR_2", "ATR_3", "ATR_4", "ATR_5", "SO", "OBV", "Volatility"]] - train_mean) / train_std
    norm_data_test = (data_test.loc[:, ["Open", "High", "Low", "Close", "Volume", "CMF", "CMF_1", "CMF_2", "CMF_3", "CMF_4", "CMF_5", "RSI", "RSI_1", "RSI_2", "RSI_3", "RSI_4", "RSI_5", "MACD", "MACD_1", "MACD_2", "MACD_3", "MACD_4", "MACD_5", "SMA_20", "SMA_20_1", "SMA_20_2", "SMA_20_3", "SMA_20_4", "SMA_20_5", "ADX", "ADX_1", "ADX_2", "ADX_3", "ADX_4", "ADX_5", "CCI", "CCI_1", "CCI_2", "CCI_3", "CCI_4", "CCI_5", "ATR", "ATR_1", "ATR_2", "ATR_3", "ATR_4", "ATR_5", "SO", "OBV", "Volatility"]] - train_mean) / train_std

    lags = 5

    X_train = pd.DataFrame()
    X_test = pd.DataFrame()

    for lag in range(lags):
        X_train[f"Open_{lag}"] = norm_data_train.Open.shift(lag)
        X_train[f"High_{lag}"] = norm_data_train.High.shift(lag)
        X_train[f"Low_{lag}"] = norm_data_train.Low.shift(lag)
        X_train[f"Close_{lag}"] = norm_data_train.Close.shift(lag)
        X_train[f"Volume_{lag}"] = norm_data_train.Volume.shift(lag)
        X_train[f"SO_{lag}"] = norm_data_train.SO.shift(lag)
        X_train[f"OBV_{lag}"] = norm_data_train.OBV.shift(lag)
        X_train[f"Volatility_{lag}"] = norm_data_train.Volatility.shift(lag)   
        X_train[f"CMF_{lag}"] = norm_data_train.CMF.shift(lag)
        X_train[f"RSI_{lag}"] = norm_data_train.RSI.shift(lag)
        X_train[f"MACD_{lag}"] = norm_data_train.MACD.shift(lag)
        X_train[f"SMA_20_{lag}"] = norm_data_train.SMA_20.shift(lag)
        X_train[f"ADX_{lag}"] = norm_data_train.ADX.shift(lag)
        X_train[f"CCI_{lag}"] = norm_data_train.CCI.shift(lag)
        X_train[f"ATR_{lag}"] = norm_data_train.ATR.shift(lag)
        X_test[f"Open_{lag}"] = norm_data_test.Open.shift(lag)
        X_test[f"High_{lag}"] = norm_data_test.High.shift(lag)
        X_test[f"Low_{lag}"] = norm_data_test.Low.shift(lag)
        X_test[f"Close_{lag}"] = norm_data_test.Close.shift(lag)
        X_test[f"Volume_{lag}"] = norm_data_test.Volume.shift(lag)
        X_test[f"SO_{lag}"] = norm_data_test.SO.shift(lag)
        X_test[f"OBV_{lag}"] = norm_data_test.OBV.shift(lag)
        X_test[f"Volatility_{lag}"] = norm_data_test.Volatility.shift(lag)
        X_test[f"CMF_{lag}"] = norm_data_test.CMF.shift(lag)
        X_test[f"RSI_{lag}"] = norm_data_test.RSI.shift(lag)
        X_test[f"MACD_{lag}"] = norm_data_test.MACD.shift(lag)
        X_test[f"SMA_20_{lag}"] = norm_data_test.SMA_20.shift(lag)
        X_test[f"ADX_{lag}"] = norm_data_test.ADX.shift(lag)
        X_test[f"CCI_{lag}"] = norm_data_test.CCI.shift(lag)
        X_test[f"ATR_{lag}"] = norm_data_test.ATR.shift(lag)

    Y_train = (X_train.Close_0 * (1 + 0.01) < X_train.Close_0.shift(-1)).astype(float)
    Y_test = (X_test.Close_0 * (1 + 0.01) < X_test.Close_0.shift(-1)).astype(float)

    X_train = X_train.iloc[5:-1, :].values
    X_test = X_test.iloc[5:-1, :].values

    Y_train = Y_train.iloc[5:-1].values.reshape(-1, 1)
    Y_test = Y_test.iloc[5:-1].values.reshape(-1, 1)

    features = X_train.shape[1]

    X_train = X_train.reshape(-1, features, 1)
    X_test = X_test.reshape(-1, features, 1)

    input_shape = X_train.shape[1:]
    # Hyperparams
    head_size = 256
    num_heads = 4
    num_transformer_blocks = 4
    dnn_dim = 4
    units = 128
    # Defining input_shape as Input layer
    input_layer = tf.keras.layers.Input(input_shape)
    # Creating our transformers based on the input layer
    transformer_layers = input_layer
    for _ in range(num_transformer_blocks):
        # Stacking transformers
        transformer_layers = create_transformer(inputs=transformer_layers,
                                                head_size=head_size,
                                                num_heads=num_heads,
                                                dnn_dim=dnn_dim)
    # Adding global pooling
    pooling_layer = tf.keras.layers.GlobalAveragePooling1D(data_format="channels_last")\
                                                          (transformer_layers)
    # Adding MLP layers
    l1 = tf.keras.layers.Dense(units=128, activation="leaky_relu")(pooling_layer)
    l2 = tf.keras.layers.Dropout(0.3)(l1)
    l3 = tf.keras.layers.Dense(units=128, activation="leaky_relu")(l2)
    # Last layer, units = 2 for True and False values
    outputs = tf.keras.layers.Dense(units=2, activation="softmax")(l3)
    # Model
    model = tf.keras.Model(inputs=input_layer,
                           outputs=outputs,
                           name="transformers_classification")
    metric = tf.keras.metrics.SparseCategoricalAccuracy()
    adam_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    #callbacks = [tf.keras.callbacks.EarlyStopping(monitor="loss",
    #                                              patience=10,
    #                                              restore_best_weights=True)]
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=adam_optimizer,
        metrics=[metric])
    model.summary()
    model.fit(
        X_train,
        Y_train,
        epochs=20,
        batch_size=64)
    model.save("transformer_classifier.keras")
    y_hat_train = model.predict(X_train)

    return y_hat_train

def buy_sell_tr(data):
    buy_signals = pd.DataFrame(data, columns=['Probabilidad_0', 'Probabilidad_1'])
    buy_signals['buy_signals'] = buy_signals.apply(lambda row: 1 if row['Probabilidad_0'] < row['Probabilidad_1'] else 0, axis=1)
    buy_signals = buy_signals.drop(columns=['Probabilidad_0', 'Probabilidad_1'])
    sell_signals = buy_signals.copy()  # Copiar el DataFrame original
    sell_signals['sell_signals'] = sell_signals['buy_signals'].apply(lambda x: 1 if x == 0 else 0)  # Invertir los valores de buy_signalls
    sell_signals = sell_signals.drop(columns=['buy_signals'])
    
    return buy_signals, sell_signals 

def backtest_tr(data, buy_signals, sell_signals, stop_loss, take_profit, n_shares):
            history = []
            active_operations = []
            cash = 1_000_000
            com = 1.25 / 100

            for i, row in data.iterrows():
                # close active operation
                active_op_temp = []
                for operation in active_operations:
                    if operation["stop_loss"] > row.Close:
                        cash += (row.Close * operation["n_shares"]) * (1 - com)
                    elif operation["take_profit"] < row.Close:
                        cash += (row.Close * operation["n_shares"]) * (1 - com)
                    else:
                        active_op_temp.append(operation)
                active_operations = active_op_temp

                # check if we have enough cash
                if cash < (row.Close * (1 + com)):
                    asset_vals = sum([operation["n_shares"] * row.Close for operation in active_operations])
                    portfolio_value = cash + asset_vals
                    continue

                # Apply buy signals
                if buy_signals.loc[i].any():
                    active_operations.append({
                        "bought": row.Close,
                        "n_shares": n_shares,
                        "stop_loss": row.Close * stop_loss,
                        "take_profit": row.Close * take_profit
                    })

                    cash -= row.Close * (1 + com) * n_shares

                # Apply sell signals
                if sell_signals.loc[i].any():
                    active_op_temp = []
                    for operation in active_operations:
                        if operation["take_profit"] < row.Close or operation["stop_loss"] > row.Close:
                            cash += (row.Close * operation["n_shares"]) * (1 - com)
                        else:
                            active_op_temp.append(operation)
                    active_operations = active_op_temp

                asset_vals = sum([operation["n_shares"] * row.Close for operation in active_operations])
                portfolio_value = cash + asset_vals

            return portfolio_value

def create_transformer(inputs, head_size, num_heads, dnn_dim):
    # Stacking layers
    l1 = tf.keras.layers.MultiHeadAttention(key_dim=head_size,
                                            num_heads=num_heads,
                                            dropout=0.2)(inputs, inputs)
    l2 = tf.keras.layers.Dropout(0.2)(l1)
    l3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(l2)
    
    res = l3 + inputs
    
    # Traditional DNN
    l4 = tf.keras.layers.Conv1D(filters=4, kernel_size=1, activation="relu")(res)
    l5 = tf.keras.layers.Dropout(0.2)(l4)
    l6 = tf.keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(l5)
    l7 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(l6)
    return l7 + res

def plot_candle_chart(file_path: str):
    # Leer el archivo CSV
    df = pd.read_csv(file_path)
    # Convertir el índice a tipo datetime si es necesario
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    # Crear el gráfico de velas
    mpf.plot(df, type='candle', style='charles', title='Candlestick Chart', ylabel='Price')
    
def create_and_compile_dnn_model(input_shape, params):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(input_shape,)))
    for i in range(params['num_layers']):
        units = params[f'num_units_layer_{i}']
        activation = params[f'activation_layer_{i}']
        model.add(tf.keras.layers.Dense(units=units, activation=activation))
    model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

def buy_signals_1d_dl(data):
    buy_signals = pd.DataFrame()
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    input_shape = X.shape[1]
    threshold = 0.5
    # Crear y compilar modelo DNN1
    best_dnn1_model = create_and_compile_dnn_model(input_shape, {'num_layers': 5, 'num_units_layer_0': 99, 'num_units_layer_1': 255, 'num_units_layer_2': 99, 'num_units_layer_3': 238, 'num_units_layer_4': 157, 'activation_layer_0': 'sigmoid', 'activation_layer_1': 'tanh', 'activation_layer_2': 'tanh', 'activation_layer_3': 'tanh', 'activation_layer_4': 'tanh', 'learning_rate': 0.00018386130982200606})
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    best_dnn1_model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
    y_pred_dnn1 = best_dnn1_model.predict(X)
    y_pred_rounded_dnn1 = (y_pred_dnn1 > threshold).astype('int32')
    # Crear y compilar modelo DNN2
    best_dnn2_model = create_and_compile_dnn_model(input_shape, {'num_layers': 6, 'num_units_layer_0': 120, 'num_units_layer_1': 109, 'num_units_layer_2': 158, 'num_units_layer_3': 104, 'num_units_layer_4': 178, 'num_units_layer_5': 164, 'activation_layer_0': 'selu', 'activation_layer_1': 'relu', 'activation_layer_2': 'selu', 'activation_layer_3': 'relu', 'activation_layer_4': 'selu', 'activation_layer_5': 'relu', 'learning_rate': 0.00011545356469724032})
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    best_dnn2_model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
    y_pred_dnn2 = best_dnn2_model.predict(X)
    y_pred_rounded_dnn2 = (y_pred_dnn2 > threshold).astype('int32')
    # Crear y compilar modelo DNN3
    best_dnn3_model = create_and_compile_dnn_model(input_shape, {'num_layers': 6, 'num_units_layer_0': 253, 'num_units_layer_1': 200, 'num_units_layer_2': 132, 'num_units_layer_3': 169, 'num_units_layer_4': 338, 'num_units_layer_5': 495, 'activation_layer_0': 'selu', 'activation_layer_1': 'relu', 'activation_layer_2': 'selu', 'activation_layer_3': 'elu', 'activation_layer_4': 'relu', 'activation_layer_5': 'selu', 'learning_rate': 0.0007162616068136376})
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    best_dnn3_model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
    y_pred_dnn3 = best_dnn3_model.predict(X)
    y_pred_rounded_dnn3 = (y_pred_dnn3 > threshold).astype('int32')
    # Agregar las predicciones como nuevas columnas al conjunto de datos original
    buy_signals['predicciones_dnn1'] = pd.Series(y_pred_rounded_dnn1.flatten())
    buy_signals['predicciones_dnn2'] = pd.Series(y_pred_rounded_dnn2.flatten())
    buy_signals['predicciones_dnn3'] = pd.Series(y_pred_rounded_dnn3.flatten())
    return buy_signals

def sell_signals_1d_dl(data):
    sell_signals = pd.DataFrame()
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    input_shape = X.shape[1]
    threshold = 0.5
    # Crear y compilar modelo DNN1
    best_dnn1_model = create_and_compile_dnn_model(input_shape, {'num_layers': 1, 'num_units_layer_0': 68, 'activation_layer_0': 'sigmoid', 'learning_rate': 0.012725657117983725})
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    best_dnn1_model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
    y_pred_dnn1 = best_dnn1_model.predict(X)
    y_pred_rounded_dnn1 = (y_pred_dnn1 > threshold).astype('int32')
    # Crear y compilar modelo DNN2
    best_dnn2_model = create_and_compile_dnn_model(input_shape, {'num_layers': 3, 'num_units_layer_0': 191, 'num_units_layer_1': 153, 'num_units_layer_2': 185, 'activation_layer_0': 'sigmoid', 'activation_layer_1': 'selu', 'activation_layer_2': 'selu', 'learning_rate': 0.0023909850276938767})
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    best_dnn2_model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
    y_pred_dnn2 = best_dnn2_model.predict(X)
    y_pred_rounded_dnn2 = (y_pred_dnn2 > threshold).astype('int32')
    # Crear y compilar modelo DNN3
    best_dnn3_model = create_and_compile_dnn_model(input_shape, {'num_layers': 4, 'num_units_layer_0': 386, 'num_units_layer_1': 196, 'num_units_layer_2': 505, 'num_units_layer_3': 422, 'activation_layer_0': 'elu', 'activation_layer_1': 'relu', 'activation_layer_2': 'relu', 'activation_layer_3': 'selu', 'learning_rate': 0.0022799789776606844})
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    best_dnn3_model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
    y_pred_dnn3 = best_dnn3_model.predict(X)
    y_pred_rounded_dnn3 = (y_pred_dnn3 > threshold).astype('int32')
    # Agregar las predicciones como nuevas columnas al conjunto de datos original
    sell_signals['predicciones_dnn1'] = pd.Series(y_pred_rounded_dnn1.flatten())
    sell_signals['predicciones_dnn2'] = pd.Series(y_pred_rounded_dnn2.flatten())
    sell_signals['predicciones_dnn3'] = pd.Series(y_pred_rounded_dnn3.flatten())
    return sell_signals

def buy_signals_1h_dl(data):
    buy_signals = pd.DataFrame()
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    input_shape = X.shape[1]
    threshold = 0.5
    # Crear y compilar modelo DNN1
    best_dnn1_model = create_and_compile_dnn_model(input_shape, {'num_layers': 5, 'num_units_layer_0': 195, 'num_units_layer_1': 197, 'num_units_layer_2': 166, 'num_units_layer_3': 36, 'num_units_layer_4': 38, 'activation_layer_0': 'relu', 'activation_layer_1': 'relu', 'activation_layer_2': 'tanh', 'activation_layer_3': 'sigmoid', 'activation_layer_4': 'sigmoid', 'learning_rate': 0.00022097125380689752})
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    best_dnn1_model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
    y_pred_dnn1 = best_dnn1_model.predict(X)
    y_pred_rounded_dnn1 = (y_pred_dnn1 > threshold).astype('int32')
    # Crear y compilar modelo DNN2
    best_dnn2_model = create_and_compile_dnn_model(input_shape, {'num_layers': 1, 'num_units_layer_0': 134, 'activation_layer_0': 'selu', 'learning_rate': 0.0002152972267593075})
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    best_dnn2_model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
    y_pred_dnn2 = best_dnn2_model.predict(X)
    y_pred_rounded_dnn2 = (y_pred_dnn2 > threshold).astype('int32')
    # Crear y compilar modelo DNN3
    best_dnn3_model = create_and_compile_dnn_model(input_shape, {'num_layers': 4, 'num_units_layer_0': 380, 'num_units_layer_1': 242, 'num_units_layer_2': 432, 'num_units_layer_3': 271, 'activation_layer_0': 'relu', 'activation_layer_1': 'selu', 'activation_layer_2': 'selu', 'activation_layer_3': 'elu', 'learning_rate': 0.00043095287802657085})
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    best_dnn3_model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
    y_pred_dnn3 = best_dnn3_model.predict(X)
    y_pred_rounded_dnn3 = (y_pred_dnn3 > threshold).astype('int32')
    # Agregar las predicciones como nuevas columnas al conjunto de datos original
    buy_signals['predicciones_dnn1'] = pd.Series(y_pred_rounded_dnn1.flatten())
    buy_signals['predicciones_dnn2'] = pd.Series(y_pred_rounded_dnn2.flatten())
    buy_signals['predicciones_dnn3'] = pd.Series(y_pred_rounded_dnn3.flatten())
    return buy_signals

def sell_signals_1h_dl(data):
    sell_signals = pd.DataFrame()
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    input_shape = X.shape[1]
    threshold = 0.5
    # Crear y compilar modelo DNN1
    best_dnn1_model = create_and_compile_dnn_model(input_shape, {'num_layers': 1, 'num_units_layer_0': 252, 'activation_layer_0': 'relu', 'learning_rate': 0.00017388198812466542})
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    best_dnn1_model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
    y_pred_dnn1 = best_dnn1_model.predict(X)
    y_pred_rounded_dnn1 = (y_pred_dnn1 > threshold).astype('int32')
    # Crear y compilar modelo DNN2
    best_dnn2_model = create_and_compile_dnn_model(input_shape, {'num_layers': 5, 'num_units_layer_0': 80, 'num_units_layer_1': 196, 'num_units_layer_2': 152, 'num_units_layer_3': 65, 'num_units_layer_4': 206, 'activation_layer_0': 'selu', 'activation_layer_1': 'relu', 'activation_layer_2': 'selu', 'activation_layer_3': 'relu', 'activation_layer_4': 'sigmoid', 'learning_rate': 0.04260341650994791})
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    best_dnn2_model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
    y_pred_dnn2 = best_dnn2_model.predict(X)
    y_pred_rounded_dnn2 = (y_pred_dnn2 > threshold).astype('int32')
    # Crear y compilar modelo DNN3
    best_dnn3_model = create_and_compile_dnn_model(input_shape, {'num_layers': 2, 'num_units_layer_0': 405, 'num_units_layer_1': 501, 'activation_layer_0': 'relu', 'activation_layer_1': 'selu', 'learning_rate': 0.002503022188822917})
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    best_dnn3_model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
    y_pred_dnn3 = best_dnn3_model.predict(X)
    y_pred_rounded_dnn3 = (y_pred_dnn3 > threshold).astype('int32')
    # Agregar las predicciones como nuevas columnas al conjunto de datos original
    sell_signals['predicciones_dnn1'] = pd.Series(y_pred_rounded_dnn1.flatten())
    sell_signals['predicciones_dnn2'] = pd.Series(y_pred_rounded_dnn2.flatten())
    sell_signals['predicciones_dnn3'] = pd.Series(y_pred_rounded_dnn3.flatten())
    return sell_signals

def buy_signals_1m_dl(data):
    buy_signals = pd.DataFrame()
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    input_shape = X.shape[1]
    threshold = 0.5
    # Crear y compilar modelo DNN1
    best_dnn1_model = create_and_compile_dnn_model(input_shape, {'num_layers': 2, 'num_units_layer_0': 64, 'num_units_layer_1': 108, 'activation_layer_0': 'relu', 'activation_layer_1': 'sigmoid', 'learning_rate': 0.00022834017269211617})
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    best_dnn1_model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
    y_pred_dnn1 = best_dnn1_model.predict(X)
    y_pred_rounded_dnn1 = (y_pred_dnn1 > threshold).astype('int32')
    # Crear y compilar modelo DNN2
    best_dnn2_model = create_and_compile_dnn_model(input_shape, {'num_layers': 2, 'num_units_layer_0': 132, 'num_units_layer_1': 218, 'activation_layer_0': 'relu', 'activation_layer_1': 'relu', 'learning_rate': 0.00030806549927249493})
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    best_dnn2_model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
    y_pred_dnn2 = best_dnn2_model.predict(X)
    y_pred_rounded_dnn2 = (y_pred_dnn2 > threshold).astype('int32')
    # Crear y compilar modelo DNN3
    best_dnn3_model = create_and_compile_dnn_model(input_shape, {'num_layers': 2, 'num_units_layer_0': 84, 'num_units_layer_1': 199, 'activation_layer_0': 'relu', 'activation_layer_1': 'selu', 'learning_rate': 0.0025599165777107466})
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    best_dnn3_model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
    y_pred_dnn3 = best_dnn3_model.predict(X)
    y_pred_rounded_dnn3 = (y_pred_dnn3 > threshold).astype('int32')
    # Agregar las predicciones como nuevas columnas al conjunto de datos original
    buy_signals['predicciones_dnn1'] = pd.Series(y_pred_rounded_dnn1.flatten())
    buy_signals['predicciones_dnn2'] = pd.Series(y_pred_rounded_dnn2.flatten())
    buy_signals['predicciones_dnn3'] = pd.Series(y_pred_rounded_dnn3.flatten())
    return buy_signals

def sell_signals_1m_dl(data):
    sell_signals = pd.DataFrame()
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    input_shape = X.shape[1]
    threshold = 0.5
    # Crear y compilar modelo DNN1
    best_dnn1_model = create_and_compile_dnn_model(input_shape, {'num_layers': 1, 'num_units_layer_0': 42, 'activation_layer_0': 'relu', 'learning_rate': 0.0003419209078643625})
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    best_dnn1_model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
    y_pred_dnn1 = best_dnn1_model.predict(X)
    y_pred_rounded_dnn1 = (y_pred_dnn1 > threshold).astype('int32')
    # Crear y compilar modelo DNN2
    best_dnn2_model = create_and_compile_dnn_model(input_shape, {'num_layers': 2, 'num_units_layer_0': 208, 'num_units_layer_1': 77, 'activation_layer_0': 'relu', 'activation_layer_1': 'relu', 'learning_rate': 0.0012215742539122853})
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    best_dnn2_model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
    y_pred_dnn2 = best_dnn2_model.predict(X)
    y_pred_rounded_dnn2 = (y_pred_dnn2 > threshold).astype('int32')
    # Crear y compilar modelo DNN3
    best_dnn3_model = create_and_compile_dnn_model(input_shape, {'num_layers': 4, 'num_units_layer_0': 498, 'num_units_layer_1': 476, 'num_units_layer_2': 93, 'num_units_layer_3': 497, 'activation_layer_0': 'elu', 'activation_layer_1': 'selu', 'activation_layer_2': 'elu', 'activation_layer_3': 'relu', 'learning_rate': 2.413838373740613e-05})
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    best_dnn3_model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
    y_pred_dnn3 = best_dnn3_model.predict(X)
    y_pred_rounded_dnn3 = (y_pred_dnn3 > threshold).astype('int32')
    # Agregar las predicciones como nuevas columnas al conjunto de datos original
    sell_signals['predicciones_dnn1'] = pd.Series(y_pred_rounded_dnn1.flatten())
    sell_signals['predicciones_dnn2'] = pd.Series(y_pred_rounded_dnn2.flatten())
    sell_signals['predicciones_dnn3'] = pd.Series(y_pred_rounded_dnn3.flatten())
    return sell_signals

def buy_signals_5m_dl(data):
    buy_signals = pd.DataFrame()
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    input_shape = X.shape[1]
    threshold = 0.5
    # Crear y compilar modelo DNN1
    best_dnn1_model = create_and_compile_dnn_model(input_shape, {'num_layers': 1, 'num_units_layer_0': 48, 'activation_layer_0': 'sigmoid', 'learning_rate': 0.00011910937340114574})
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    best_dnn1_model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
    y_pred_dnn1 = best_dnn1_model.predict(X)
    y_pred_rounded_dnn1 = (y_pred_dnn1 > threshold).astype('int32')
    # Crear y compilar modelo DNN2
    best_dnn2_model = create_and_compile_dnn_model(input_shape, {'num_layers': 5, 'num_units_layer_0': 201, 'num_units_layer_1': 229, 'num_units_layer_2': 83, 'num_units_layer_3': 171, 'num_units_layer_4': 70, 'activation_layer_0': 'selu', 'activation_layer_1': 'selu', 'activation_layer_2': 'selu', 'activation_layer_3': 'relu', 'activation_layer_4': 'sigmoid', 'learning_rate': 0.002526805879974853})
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    best_dnn2_model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
    y_pred_dnn2 = best_dnn2_model.predict(X)
    y_pred_rounded_dnn2 = (y_pred_dnn2 > threshold).astype('int32')
    # Crear y compilar modelo DNN3
    best_dnn3_model = create_and_compile_dnn_model(input_shape, {'num_layers': 2, 'num_units_layer_0': 173, 'num_units_layer_1': 463, 'activation_layer_0': 'elu', 'activation_layer_1': 'elu', 'learning_rate': 1.4196702540553977e-05})
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    best_dnn3_model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
    y_pred_dnn3 = best_dnn3_model.predict(X)
    y_pred_rounded_dnn3 = (y_pred_dnn3 > threshold).astype('int32')
    # Agregar las predicciones como nuevas columnas al conjunto de datos original
    buy_signals['predicciones_dnn1'] = pd.Series(y_pred_rounded_dnn1.flatten())
    buy_signals['predicciones_dnn2'] = pd.Series(y_pred_rounded_dnn2.flatten())
    buy_signals['predicciones_dnn3'] = pd.Series(y_pred_rounded_dnn3.flatten())
    return buy_signals

def sell_signals_5m_dl(data):
    sell_signals = pd.DataFrame()
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    input_shape = X.shape[1]
    threshold = 0.5
    # Crear y compilar modelo DNN1
    best_dnn1_model = create_and_compile_dnn_model(input_shape, {'num_layers': 2, 'num_units_layer_0': 178, 'num_units_layer_1': 115, 'activation_layer_0': 'relu', 'activation_layer_1': 'relu', 'learning_rate': 0.0011702775156435393})
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    best_dnn1_model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
    y_pred_dnn1 = best_dnn1_model.predict(X)
    y_pred_rounded_dnn1 = (y_pred_dnn1 > threshold).astype('int32')
    # Crear y compilar modelo DNN2
    best_dnn2_model = create_and_compile_dnn_model(input_shape, {'num_layers': 5, 'num_units_layer_0': 97, 'num_units_layer_1': 136, 'num_units_layer_2': 150, 'num_units_layer_3': 242, 'num_units_layer_4': 99, 'activation_layer_0': 'relu', 'activation_layer_1': 'sigmoid', 'activation_layer_2': 'relu', 'activation_layer_3': 'selu', 'activation_layer_4': 'relu', 'learning_rate': 0.07182865605616788})
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    best_dnn2_model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
    y_pred_dnn2 = best_dnn2_model.predict(X)
    y_pred_rounded_dnn2 = (y_pred_dnn2 > threshold).astype('int32')
    # Crear y compilar modelo DNN3
    best_dnn3_model = create_and_compile_dnn_model(input_shape, {'num_layers': 5, 'num_units_layer_0': 89, 'num_units_layer_1': 195, 'num_units_layer_2': 127, 'num_units_layer_3': 199, 'num_units_layer_4': 130, 'activation_layer_0': 'selu', 'activation_layer_1': 'elu', 'activation_layer_2': 'relu', 'activation_layer_3': 'selu', 'activation_layer_4': 'relu', 'learning_rate': 0.00036584504741541676})
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    best_dnn3_model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
    y_pred_dnn3 = best_dnn3_model.predict(X)
    y_pred_rounded_dnn3 = (y_pred_dnn3 > threshold).astype('int32')
    # Agregar las predicciones como nuevas columnas al conjunto de datos original
    sell_signals['predicciones_dnn1'] = pd.Series(y_pred_rounded_dnn1.flatten())
    sell_signals['predicciones_dnn2'] = pd.Series(y_pred_rounded_dnn2.flatten())
    sell_signals['predicciones_dnn3'] = pd.Series(y_pred_rounded_dnn3.flatten())
    return sell_signals

def plot_tr(buy_signals, sell_signals, prices):
    merged_df = pd.concat([prices['Close'], buy_signals, sell_signals], axis=1)
    # Graficar los precios Close
    plt.plot(merged_df['Close'], label='Close Price')
    # Marcar los puntos de compra y venta en el gráfico
    plt.scatter(merged_df.index[merged_df['buy_signals'] == 1], 
                merged_df['Close'][merged_df['buy_signals'] == 1], 
                color='green', marker='^', label='Buy Signal')
    plt.scatter(merged_df.index[merged_df['sell_signals'] == 1], 
                merged_df['Close'][merged_df['sell_signals'] == 1], 
                color='red', marker='v', label='Sell Signal')
    # Añadir leyenda y título
    plt.legend()
    plt.title('Close Prices with Buy and Sell Signals')
    plt.xlabel('Time')
    plt.ylabel('Close Price')
    plt.grid(True)
    # Mostrar el gráfico
    plt.show()

def plot_buy_sell_signals_dl(global_buy_signals, global_sell_signals, data_test_long):
    buy_sell_xgboost = pd.DataFrame()
    buy_sell_xgboost['pred_xg_buy'] = global_buy_signals['predicciones_dnn1']
    buy_sell_xgboost['pred_xg_sell'] = global_sell_signals['predicciones_dnn1']
    buy_sell_xgboost['Close'] = data_test_long['Close_Lag0']
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(buy_sell_xgboost['Close'], label='Precio de Cierre', color='black')
    ax.scatter(buy_sell_xgboost.index[buy_sell_xgboost['pred_xg_buy']], buy_sell_xgboost['Close'][buy_sell_xgboost['pred_xg_buy']], marker='^', color='r', label='Compra')
    ax.scatter(buy_sell_xgboost.index[buy_sell_xgboost['pred_xg_sell']], buy_sell_xgboost['Close'][buy_sell_xgboost['pred_xg_sell']], marker='v', color='g', label='Venta')
    ax.set_title('Señales de Compra y Venta')
    ax.set_xlabel('Índice de Tiempo')
    ax.set_ylabel('Precio')
    ax.legend()
    plt.grid(True)
    plt.show()
    
def file_features(data: str, ds_type: str):
    data = pd.read_csv(data)
    data = data.dropna()
    data1 = pd.DataFrame()
    # Calcular indicadores tecnicos
    cmf_data = ta.volume.ChaikinMoneyFlowIndicator(data.High, data.Low, data.Close, data.Volume, window=14)
    rsi_data = ta.momentum.RSIIndicator(data.Close, window=14)

    data1["CMF"] = cmf_data.chaikin_money_flow()
    data1["RSI"] = rsi_data.rsi()
    # Calcular la volatilidad
    data1['Volatility'] = data['High'] - data['Low']
    data1['Close_Lag0'] = data['Close']
    # Calcular las tendencias
    for i in range(1, 5 + 1):
        data1[f'Close_Lag{i}'] = data['Close'].shift(i)
    # Variable ded respuesta
    if ds_type == "buy":
        data1['Response'] = (data['Close'] < data['Close'].shift(-10))
    else:
        data1['Response'] = (data['Close'] > data['Close'].shift(-10))

    data1 = data1.drop(data1.index[:30])
    data1 = data1.drop(data1.index[-30:])
    data1.reset_index(drop=True, inplace=True)

    return data1

def buy_signals_1d(data):
    buy_signals = pd.DataFrame()
    # Selecciona las características
    X = data.iloc[:, :-1]
    # Selecciona la variable objetivo
    y = data.iloc[:, -1]

    # Crear modelos con los mejores parámetros encontrados para cada algoritmo
    best_logistic_model = LogisticRegression(penalty='l1', C=0.010968139046945382, solver='saga')
    best_svm_model = SVC(C=0.43326595131065515, kernel='poly', degree= 2, gamma='scale')
    best_xgboost_model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.01737000818203036, subsample=0.6,
                                       colsample_bytree=0.8)

    # Entrenar los modelos con todo el conjunto de datos original
    best_logistic_model.fit(X, y)
    best_svm_model.fit(X, y)
    best_xgboost_model.fit(X, y)

    # Realizar predicciones en el conjunto de datos original
    predictions_lr = best_logistic_model.predict(X)
    predictions_svm = best_svm_model.predict(X)
    predictions_xgboost = best_xgboost_model.predict(X)
    predictions_xgboost_bool = predictions_xgboost.astype(bool)


    # Agregar las predicciones como nuevas columnas al conjunto de datos original
    buy_signals['predicciones_lr'] = predictions_lr
    buy_signals['predicciones_svm'] = predictions_svm
    buy_signals['predicciones_xgboost'] = predictions_xgboost_bool

    return buy_signals

def sell_signals_1d(data):
    sell_signals = pd.DataFrame()
    # Selecciona las características
    X = data.iloc[:, :-1]
    # Selecciona la variable objetivo
    y = data.iloc[:, -1]

    # Crear modelos con los mejores parámetros encontrados para cada algoritmo
    best_logistic_model = LogisticRegression(penalty='l2', C=0.6431546383805156, solver='liblinear', max_iter=5_000, random_state=123)
    best_svm_model = SVC(C=0.0035454674239493874, kernel='rbf', gamma='scale', max_iter=5_000, random_state=123)
    best_xgboost_model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.030090107781624707, subsample=0.9,
                                       colsample_bytree=0.8, max_iter=5_000, random_state=123)

    # Entrenar los modelos con todo el conjunto de datos original
    best_logistic_model.fit(X, y)
    best_svm_model.fit(X, y)
    best_xgboost_model.fit(X, y)

    # Realizar predicciones en el conjunto de datos original
    predictions_lr = best_logistic_model.predict(X)
    predictions_svm = best_svm_model.predict(X)
    predictions_xgboost = best_xgboost_model.predict(X)
    predictions_xgboost_bool = predictions_xgboost.astype(bool)

    # Agregar las predicciones como nuevas columnas al conjunto de datos original
    sell_signals['predicciones_lr'] = predictions_lr
    sell_signals['predicciones_svm'] = predictions_svm
    sell_signals['predicciones_xgboost'] = predictions_xgboost_bool

    return sell_signals

def buy_signals_1h(data):
    buy_signals = pd.DataFrame()
    # Selecciona las características
    X = data.iloc[:, :-1]
    # Selecciona la variable objetivo
    y = data.iloc[:, -1]

    # Crear modelos con los mejores parámetros encontrados para cada algoritmo
    best_logistic_model = LogisticRegression(penalty='l1', C=0.0021323714279392757, solver='saga', max_iter=5_000, random_state=123)
    best_svm_model = SVC(C=0.04584664364591048, kernel='poly', degree= 2, gamma='auto', max_iter=5_000, random_state=123)
    best_xgboost_model = XGBClassifier(n_estimators=900, max_depth=9, learning_rate=0.2220589262776878, subsample=0.5,
                                       colsample_bytree=0.7, max_iter=5_000, random_state=123)

    # Entrenar los modelos con todo el conjunto de datos original
    best_logistic_model.fit(X, y)
    best_svm_model.fit(X, y)
    best_xgboost_model.fit(X, y)

    # Realizar predicciones en el conjunto de datos original
    predictions_lr = best_logistic_model.predict(X)
    predictions_svm = best_svm_model.predict(X)
    predictions_xgboost = best_xgboost_model.predict(X)
    predictions_xgboost_bool = predictions_xgboost.astype(bool)


    # Agregar las predicciones como nuevas columnas al conjunto de datos original
    buy_signals['predicciones_lr'] = predictions_lr
    buy_signals['predicciones_svm'] = predictions_svm
    buy_signals['predicciones_xgboost'] = predictions_xgboost_bool

    return buy_signals

def sell_signals_1h(data):
    sell_signals = pd.DataFrame()
    # Selecciona las características
    X = data.iloc[:, :-1]
    # Selecciona la variable objetivo
    y = data.iloc[:, -1]

    # Crear modelos con los mejores parámetros encontrados para cada algoritmo
    best_logistic_model = LogisticRegression(penalty='l2', C=0.0028315415752095535, solver='saga', max_iter=5_000, random_state=123)
    best_svm_model = SVC(C=0.2018232497206618, kernel='rbf', gamma='auto', max_iter=5_000, random_state=123)
    best_xgboost_model = XGBClassifier(n_estimators=100, max_depth=10, learning_rate=0.102602018285415, subsample=0.5,
                                       colsample_bytree=0.6, max_iter=5_000, random_state=123)

    # Entrenar los modelos con todo el conjunto de datos original
    best_logistic_model.fit(X, y)
    best_svm_model.fit(X, y)
    best_xgboost_model.fit(X, y)

    # Realizar predicciones en el conjunto de datos original
    predictions_lr = best_logistic_model.predict(X)
    predictions_svm = best_svm_model.predict(X)
    predictions_xgboost = best_xgboost_model.predict(X)
    predictions_xgboost_bool = predictions_xgboost.astype(bool)

    # Agregar las predicciones como nuevas columnas al conjunto de datos original
    sell_signals['predicciones_lr'] = predictions_lr
    sell_signals['predicciones_svm'] = predictions_svm
    sell_signals['predicciones_xgboost'] = predictions_xgboost_bool

    return sell_signals

def buy_signals_1m(data):
    buy_signals = pd.DataFrame()
    # Selecciona las características
    X = data.iloc[:, :-1]
    # Selecciona la variable objetivo
    y = data.iloc[:, -1]

    # Crear modelos con los mejores parámetros encontrados para cada algoritmo
    best_logistic_model = LogisticRegression(penalty='l2', C=0.0015827990076342706, solver='saga', max_iter=5_000, random_state=123)
    best_svm_model = SVC(C=208.31110635195978, kernel='rbf', gamma='auto', max_iter=5_000, random_state=123)
    best_xgboost_model = XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.016335181464666, subsample=0.6,
                                       colsample_bytree=0.5, max_iter=5_000, random_state=123)

    # Entrenar los modelos con todo el conjunto de datos original
    best_logistic_model.fit(X, y)
    best_svm_model.fit(X, y)
    best_xgboost_model.fit(X, y)

    # Realizar predicciones en el conjunto de datos original
    predictions_lr = best_logistic_model.predict(X)
    predictions_svm = best_svm_model.predict(X)
    predictions_xgboost = best_xgboost_model.predict(X)
    predictions_xgboost_bool = predictions_xgboost.astype(bool)


    # Agregar las predicciones como nuevas columnas al conjunto de datos original
    buy_signals['predicciones_lr'] = predictions_lr
    buy_signals['predicciones_svm'] = predictions_svm
    buy_signals['predicciones_xgboost'] = predictions_xgboost_bool

    return buy_signals

def sell_signals_1m(data):
    sell_signals = pd.DataFrame()
    # Selecciona las características
    X = data.iloc[:, :-1]
    # Selecciona la variable objetivo
    y = data.iloc[:, -1]

    # Crear modelos con los mejores parámetros encontrados para cada algoritmo
    best_logistic_model = LogisticRegression(penalty='l2', C=0.002105767490569824, solver='saga', max_iter=5_000, random_state=123)
    best_svm_model = SVC(C=89.33147882881373, kernel='linear', max_iter=5_000, random_state=123)
    best_xgboost_model = XGBClassifier(n_estimators=400, max_depth=3, learning_rate=0.061659645644687434, subsample=0.8,
                                       colsample_bytree=0.7, max_iter=5_000, random_state=123)

    # Entrenar los modelos con todo el conjunto de datos original
    best_logistic_model.fit(X, y)
    best_svm_model.fit(X, y)
    best_xgboost_model.fit(X, y)

    # Realizar predicciones en el conjunto de datos original
    predictions_lr = best_logistic_model.predict(X)
    predictions_svm = best_svm_model.predict(X)
    predictions_xgboost = best_xgboost_model.predict(X)
    predictions_xgboost_bool = predictions_xgboost.astype(bool)

    # Agregar las predicciones como nuevas columnas al conjunto de datos original
    sell_signals['predicciones_lr'] = predictions_lr
    sell_signals['predicciones_svm'] = predictions_svm
    sell_signals['predicciones_xgboost'] = predictions_xgboost_bool

    return sell_signals

def buy_signals_5m(data):
    buy_signals = pd.DataFrame()
    # Selecciona las características
    X = data.iloc[:, :-1]
    # Selecciona la variable objetivo
    y = data.iloc[:, -1]

    # Crear modelos con los mejores parámetros encontrados para cada algoritmo
    best_logistic_model = LogisticRegression(penalty='l2', C=0.002395829495054093, solver='liblinear', max_iter=5_000, random_state=123)
    best_svm_model = SVC(C= 0.002176996974479978, kernel='sigmoid', gamma='scale', max_iter=5_000, random_state=123)
    best_xgboost_model = XGBClassifier(n_estimators=1000, max_depth=9, learning_rate=0.014804455005781484, subsample=0.9,
                                       colsample_bytree=0.7, max_iter=5_000, random_state=123)

    # Entrenar los modelos con todo el conjunto de datos original
    best_logistic_model.fit(X, y)
    best_svm_model.fit(X, y)
    best_xgboost_model.fit(X, y)

    # Realizar predicciones en el conjunto de datos original
    predictions_lr = best_logistic_model.predict(X)
    predictions_svm = best_svm_model.predict(X)
    predictions_xgboost = best_xgboost_model.predict(X)
    predictions_xgboost_bool = predictions_xgboost.astype(bool)


    # Agregar las predicciones como nuevas columnas al conjunto de datos original
    buy_signals['predicciones_lr'] = predictions_lr
    buy_signals['predicciones_svm'] = predictions_svm
    buy_signals['predicciones_xgboost'] = predictions_xgboost_bool

    return buy_signals

def sell_signals_5m(data):
    sell_signals = pd.DataFrame()
    # Selecciona las características
    X = data.iloc[:, :-1]
    # Selecciona la variable objetivo
    y = data.iloc[:, -1]

    # Crear modelos con los mejores parámetros encontrados para cada algoritmo
    best_logistic_model = LogisticRegression(penalty='l1', C=0.0030591725482995644, solver='saga', max_iter=5_000, random_state=123)
    best_svm_model = SVC(C=0.6739747156890435, kernel='linear', max_iter=5_000, random_state=123)
    best_xgboost_model = XGBClassifier(n_estimators=900, max_depth=9, learning_rate=0.08242884326336312, subsample=0.5,
                                       colsample_bytree=1.0, max_iter=5_000, random_state=123)

    # Entrenar los modelos con todo el conjunto de datos original
    best_logistic_model.fit(X, y)
    best_svm_model.fit(X, y)
    best_xgboost_model.fit(X, y)

    # Realizar predicciones en el conjunto de datos original
    predictions_lr = best_logistic_model.predict(X)
    predictions_svm = best_svm_model.predict(X)
    predictions_xgboost = best_xgboost_model.predict(X)
    predictions_xgboost_bool = predictions_xgboost.astype(bool)

    # Agregar las predicciones como nuevas columnas al conjunto de datos original
    sell_signals['predicciones_lr'] = predictions_lr
    sell_signals['predicciones_svm'] = predictions_svm
    sell_signals['predicciones_xgboost'] = predictions_xgboost_bool

    return sell_signals

def plot_buy_sell_signals(global_buy_signals, global_sell_signals, data_test_long):
    buy_sell_xgboost = pd.DataFrame()
    buy_sell_xgboost['pred_xg_buy'] = global_buy_signals['predicciones_xgboost']
    buy_sell_xgboost['pred_xg_sell'] = global_sell_signals['predicciones_xgboost']
    buy_sell_xgboost['Close'] = data_test_long['Close_Lag0']
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(buy_sell_xgboost['Close'], label='Precio de Cierre', color='black')
    ax.scatter(buy_sell_xgboost.index[buy_sell_xgboost['pred_xg_buy']], buy_sell_xgboost['Close'][buy_sell_xgboost['pred_xg_buy']], marker='^', color='r', label='Compra')
    ax.scatter(buy_sell_xgboost.index[buy_sell_xgboost['pred_xg_sell']], buy_sell_xgboost['Close'][buy_sell_xgboost['pred_xg_sell']], marker='v', color='g', label='Venta')
    ax.set_title('Señales de Compra y Venta')
    ax.set_xlabel('Índice de Tiempo')
    ax.set_ylabel('Precio')
    ax.legend()
    plt.grid(True)
    plt.show()
    
def data_fun(file_path: str):
    data = pd.read_csv(file_path)
    data = data.dropna()
    data = data.drop(data.index[:30])
    data = data.drop(data.index[-30:])
    data.reset_index(drop=True, inplace=True)
    return data

def backtest(data, buy_signals, sell_signals, stop_loss, take_profit, n_shares):
    history = []
    active_operations = []
    cash = 1_000_000
    com = 1.25 / 100
    portfolio_values = []
    cash_values = []
    operations_history = []

    for i, row in data.iterrows():
        # close active operation
        active_op_temp = []
        for operation in active_operations:
            if operation["stop_loss"] > row.Close:
                cash += (row.Close * operation["n_shares"]) * (1 - com)
                operations_history.append((i, row.Close, "stop_loss", operation["n_shares"]))
            elif operation["take_profit"] < row.Close:
                cash += (row.Close * operation["n_shares"]) * (1 - com)
                operations_history.append((i, row.Close, "take_profit", operation["n_shares"]))
            else:
                active_op_temp.append(operation)
        active_operations = active_op_temp

        # check if we have enough cash
        if cash < (row.Close * (1 + com)):
            asset_vals = sum([operation["n_shares"] * row.Close for operation in active_operations])
            portfolio_value = cash + asset_vals
            portfolio_values.append(portfolio_value)
            cash_values.append(cash)
            continue

        # Apply buy signals
        if buy_signals.loc[i].any():
            active_operations.append({
                "bought": row.Close,
                "n_shares": n_shares,
                "stop_loss": row.Close * stop_loss,
                "take_profit": row.Close * take_profit
            })

            cash -= row.Close * (1 + com) * n_shares
            operations_history.append((i, row.Close, "buy", n_shares))

        # Apply sell signals
        if sell_signals.loc[i].any():
            active_op_temp = []
            for operation in active_operations:
                if operation["take_profit"] < row.Close or operation["stop_loss"] > row.Close:
                    cash += (row.Close * operation["n_shares"]) * (1 - com)
                    operations_history.append((i, row.Close, "sell", operation["n_shares"]))
                else:
                    active_op_temp.append(operation)
            active_operations = active_op_temp

        asset_vals = sum([operation["n_shares"] * row.Close for operation in active_operations])
        portfolio_value = cash + asset_vals
        portfolio_values.append(portfolio_value)
        cash_values.append(cash)

    return portfolio_values, cash_values, operations_history

def plot_operations_history(operations_history):
    data = operations_history[:80]
    transacciones = [t[0] for t in data]
    precios = [t[1] for t in data]
    acciones = [t[2] for t in data]
    identificadores = [t[3] for t in data]
    # Graficar los precios
    plt.figure(figsize=(10, 6))
    plt.plot(precios, label='Precio', marker='o', color='blue', linestyle='-')
    # Etiquetar las acciones en los puntos correspondientes
    for i, accion in enumerate(acciones):
        plt.text(i, precios[i], accion, fontsize=9, ha='right', va='bottom', rotation=45)
    # Etiquetas y título
    plt.xlabel('Transacción')
    plt.ylabel('Precio')
    plt.title('Gráfico de precios con acciones asociadas')
    # Mostrar leyenda
    plt.legend()
    # Mostrar la gráfica
    plt.grid(True)
    plt.show()
    
def port_value_plot(portfolio_values):
    periodo_tiempo = range(1, len(portfolio_values) + 1)
    # Graficar los valores del portafolio
    plt.plot(periodo_tiempo, portfolio_values, marker='o', linestyle='-')
    # Etiquetas de los ejes
    plt.xlabel('Periodo de Tiempo')
    plt.ylabel('Valor del Portafolio')
    # Título del gráfico
    plt.title('Evolución del Valor del Portafolio')
    # Mostrar la gráfica
    plt.grid(True)
    plt.show()
    
def plot_cash(cash_values, title):
    plt.plot(cash_values)
    plt.title(title)
    plt.xlabel('Tiempo')
    plt.ylabel('Valor en Efectivo')
    plt.grid(True)
    plt.show()
    
def cash_portvalue_plot(cash_values, portfolio_values):
    plt.plot(cash_values, label='Cash')
    plt.plot(portfolio_values, label='Portfolio Value')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Cash and Portfolio Value over Time')
    plt.legend()
    plt.show()
    
def pasive_portvalue_plot(portfolio_values):
    plt.plot(portfolio_values, label='Portfolio Value')
    plt.plot([0,len(portfolio_values)], [1000000,1000000], label="Pasive_Strategy")
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Estrategia Pasiva VS Estrategia Trading')
    plt.legend()
    plt.show()
    
def plot_signals(buy_signals_df, sell_signals_df, data_df):
    # Asegurarse de que los dataframes tengan el mismo índice
    buy_signals_df.index = data_df.index
    sell_signals_df.index = data_df.index
    # Crear una figura y ejes
    fig, ax = plt.subplots(figsize=(10, 6))
    # Graficar el precio de cierre
    ax.plot(data_df['Close_Lag0'], label='Precio de cierre', color='black')
    # Marcar las señales de compra como puntos verdes
    ax.scatter(data_df.index[buy_signals_df['predicciones_dnn1'] == 1],
               data_df['Close_Lag0'][buy_signals_df['predicciones_dnn1'] == 1],
               label='Compra', color='green', marker='^', s=100)
    # Marcar las señales de venta como puntos rojos
    ax.scatter(data_df.index[sell_signals_df['predicciones_dnn1'] == 1],
               data_df['Close_Lag0'][sell_signals_df['predicciones_dnn1'] == 1],
               label='Venta', color='red', marker='v', s=100)
    # Añadir leyenda y título
    ax.legend()
    ax.set_title('Señales de compra y venta en AAPL')
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Precio de cierre')
    # Rotar las fechas para una mejor visualización
    plt.xticks(rotation=45)
    # Mostrar la gráfica
    plt.tight_layout()
    plt.show()
    
def backtest1(data, buy_signals, sell_signals, stop_loss, take_profit, n_shares, commission=1.25/100):

    capital = 1000000 # Capital inicial en USD
    shares = 0       # Número de acciones en nuestra posesión
    in_position = False  # Indica si estamos en una posición (comprados)
    
    portfolio_value = []  # Lista para almacenar el valor del portafolio a lo largo del tiempo
    capital_value = []    # Lista para almacenar el valor del capital a lo largo del tiempo
    transactions = []     # Lista para almacenar información sobre las transacciones

    for i in range(len(data)):
        price = data.iloc[i]['Close']
        buy_signal = buy_signals.iloc[i][0] if i < len(buy_signals) else False
        sell_signal = sell_signals.iloc[i][0] if i < len(sell_signals) else False

        # Si hay una señal de compra y no estamos en posición
        if buy_signal and not in_position:
            shares_to_buy = min(n_shares, capital // price)  # Compramos la cantidad de acciones que podemos permitirnos
            shares += shares_to_buy
            capital -= shares_to_buy * price * (1 + commission)  # Descontamos la comisión
            in_position = True
            transactions.append(['Compra', price])

        # Si hay una señal de venta y estamos en posición
        elif sell_signal and in_position:
            capital += shares * price * (1 - commission)  # Vendemos todas las acciones al precio actual
            shares = 0
            in_position = False
            transactions.append(['Venta', price])

        # Revisamos si debemos activar el stop loss o take profit
        if in_position:
            current_value = shares * price
            stop_loss_value = current_value * stop_loss
            take_profit_value = current_value * take_profit

            if current_value <= stop_loss_value:
                capital += current_value * (1 - commission)
                shares = 0
                in_position = False
                transactions.append(['Venta', price])

            elif current_value >= take_profit_value:
                capital += current_value * (1 - commission)
                shares = 0
                in_position = False
                transactions.append(['Venta', price])

        # Almacenamos el valor del portafolio y del capital en cada paso
        portfolio_value.append(capital + shares * price)
        capital_value.append(capital)

    # Calculamos la ganancia total
    profit = capital - 10000  # Restamos el capital inicial

    # Creamos los DataFrames de pandas para el valor del portafolio, el valor del capital y las transacciones
    portfolio_value_df = pd.DataFrame({'portfolio_value': portfolio_value})
    capital_value_df = pd.DataFrame({'capital_value': capital_value})
    transactions_df = pd.DataFrame(transactions, columns=['Transaction', 'Price'])

    return portfolio_value_df, capital_value_df, transactions_df

def buy_sell(global_buy_signals_1d_long, global_sell_signals_1d_short):
    buy_signals = pd.DataFrame()
    buy_signals["predicciones_dnn1"] = global_buy_signals_1d_long["predicciones_dnn1"]
    sell_signals = pd.DataFrame()
    sell_signals["predicciones_dnn1"] = global_sell_signals_1d_short["predicciones_dnn1"]
    return buy_signals, sell_signals

def plot_operations_history(operations_history):
    data = operations_history[:35]
    transacciones = [t[0] for t in data]
    precios = [t[1] for t in data]
    acciones = [t[2] for t in data]
    identificadores = [t[3] for t in data]
    # Graficar los precios
    plt.figure(figsize=(10, 6))
    plt.plot(precios, label='Precio', marker='o', color='blue', linestyle='-')
    # Etiquetar las acciones en los puntos correspondientes
    for i, accion in enumerate(acciones):
        plt.text(i, precios[i], accion, fontsize=9, ha='right', va='bottom', rotation=45)
    # Etiquetas y título
    plt.xlabel('Transacción')
    plt.ylabel('Precio')
    plt.title('Gráfico de precios con acciones asociadas')
    # Mostrar leyenda
    plt.legend()
    # Mostrar la gráfica
    plt.grid(True)
    plt.show()
    
def port_value_plot(portfolio_values):
    periodo_tiempo = range(1, len(portfolio_values) + 1)
    # Graficar los valores del portafolio
    plt.plot(periodo_tiempo, portfolio_values, marker='o', linestyle='-')
    # Etiquetas de los ejes
    plt.xlabel('Periodo de Tiempo')
    plt.ylabel('Valor del Portafolio')
    # Título del gráfico
    plt.title('Evolución del Valor del Portafolio')
    # Mostrar la gráfica
    plt.grid(True)
    plt.show()
    
def plot_cash(cash_values):
    periodo_tiempo = range(1, len(cash_values) + 1)
    # Graficar los valores del portafolio
    plt.plot(periodo_tiempo, cash_values, marker='o', linestyle='-')
    # Etiquetas de los ejes
    plt.xlabel('Periodo de Tiempo')
    plt.ylabel('Valor del Portafolio')
    # Título del gráfico
    plt.title('Dinero atraves del Tiempo')
    # Mostrar la gráfica
    plt.grid(True)
    
def cash_portvalue_plot(cash_values, portfolio_values):
    plt.plot(cash_values, label='Cash')
    plt.plot(portfolio_values, label='Portfolio Value')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Cash and Portfolio Value over Time')
    plt.legend()
    plt.show()
    
def pasive_portvalue_plot(portfolio_values):
    plt.plot(portfolio_values, label='Portfolio Value')
    plt.plot([0,len(portfolio_values)], [1000000,1000000], label="Pasive_Strategy")
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Estrategia Pasiva VS Estrategia Trading')
    plt.legend()
    plt.show()
    
def plot_transacciones(df):
    # Lista de precios y transacciones
    precios = df['Price']
    transacciones = df['Transaction']
    # Crear el gráfico de línea
    plt.figure(figsize=(10, 6))
    plt.plot(precios,color='blue')
    # Resaltar puntos de compra y venta
    for i in range(len(precios)):
        if transacciones[i] == 'Compra':
            plt.scatter(i, precios[i], color='green', s=100, zorder=3)
        elif transacciones[i] == 'Venta':
            plt.scatter(i, precios[i], color='red', s=100, zorder=3)

    # Configuración del gráfico
    plt.xlabel('Índice')
    plt.ylabel('Precio')
    plt.title('Gráfico de Precio con Compras y Ventas')
    plt.grid(True)
    # Mostrar el gráfico
    plt.show()