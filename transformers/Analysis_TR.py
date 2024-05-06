import pandas as pd
import ta 
import optuna 
from itertools import combinations, chain 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import tensorflow as tf

def powerset(s):
    return chain.from_iterable(combinations(s,r) for r in range(1,len(s)+1))

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
    
    data1 = data1.drop(data1.index[:30])
    data1 = data1.drop(data1.index[-30:])
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
        
data_1 = pd.read_csv("aapl_5m_train.csv").dropna()
data_2 = pd.read_csv("aapl_5m_test.csv").dropna()
data_train = file_features_tr(data_1, "buy")
data_test = file_features_tr(data_2, "buy")
data = model_transformer(data_train, data_test)
buy_signals, sell_signals = buy_sell_tr(data)
Try_Few_variables = backtest(data_2, buy_signals, sell_signals, 0.6809, 1.0994972027471122, 45)