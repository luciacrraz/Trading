import pandas as pd
import ta
import optuna
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
def file_features(data, ds_type: str):
    data 1 =pd.DataFrame()
    # Calcular indicadores tecnicos
    cmf_data = ta.volume.ChaikinMoneyFlowIndicator(data.High, data.Low, data.Close, data.Volume, window = 14)
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

def objective_dnn_1(trial, data, threshold=0.5):
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X = data.iloc[:, :-1]
    # Selecciona la variable objetivo
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    # Definir los parámetros a optimizar
    num_layers = trial.suggest_int('num_layers', 1, 5)
    num_units = [trial.suggest_int(f'num_units_layer_{i}', 32, 256) for i in range(num_layers)]
    activations = [trial.suggest_categorical(f'activation_layer_{i}', ['relu', 'sigmoid', 'tanh']) for i in range(num_layers)]
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-1)
    # Crear el modelo de red neuronal
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(X_train.shape[1],)))
    for i in range(num_layers):
        model.add(tf.keras.layers.Dense(units=num_units[i], activation=activations[i]))
    model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    # Compilar el modelo
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    # Definir Early Stopping para evitar overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
    # Entrenar el modelo
    model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
    # Calcular las predicciones en el conjunto de prueba
    y_pred = model.predict(X_test)
    # Redondear las predicciones si superan el umbral
    y_pred_rounded = (y_pred > threshold).astype('int32')
    # Calcular la precisión en el conjunto de prueba
    accuracy = accuracy_score(y_test, y_pred_rounded)
    return accuracy

def objective_dnn_2(trial, data, threshold=0.5):
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X = data.iloc[:, :-1]
    # Selecciona la variable objetivo
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    # Definir los parámetros a optimizar
    num_layers = trial.suggest_int('num_layers', 1, 6)
    num_units = [trial.suggest_int(f'num_units_layer_{i}', 64, 256) for i in range(num_layers)]
    activations = [trial.suggest_categorical(f'activation_layer_{i}', ['relu', 'sigmoid', 'selu']) for i in range(num_layers)]
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-1)
    # Crear el modelo de red neuronal
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(X_train.shape[1],)))
    for i in range(num_layers):
        model.add(tf.keras.layers.Dense(units=num_units[i], activation=activations[i]))
    model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    # Compilar el modelo
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    # Definir Early Stopping para evitar overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
    # Entrenar el modelo
    model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
    # Calcular las predicciones en el conjunto de prueba
    y_pred = model.predict(X_test)
    # Redondear las predicciones si superan el umbral
    y_pred_rounded = (y_pred > threshold).astype('int32')
    # Calcular la precisión en el conjunto de prueba
    accuracy = accuracy_score(y_test, y_pred_rounded)
    return accuracy

def objective_dnn_3(trial, data, threshold=0.5):
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X = data.iloc[:, :-1]
    # Selecciona la variable objetivo
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    # Definir los parámetros a optimizar
    num_layers = trial.suggest_int('num_layers', 2, 6)
    num_units = [trial.suggest_int(f'num_units_layer_{i}', 64, 512) for i in range(num_layers)]
    activations = [trial.suggest_categorical(f'activation_layer_{i}', ['relu', 'elu', 'selu']) for i in range(num_layers)]
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    # Crear el modelo de red neuronal
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(X_train.shape[1],)))
    for i in range(num_layers):
        model.add(tf.keras.layers.Dense(units=num_units[i], activation=activations[i]))
    model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    # Compilar el modelo
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    # Definir Early Stopping para evitar overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
    # Entrenar el modelo
    model.fit(X_train, y_train, epochs=30, batch_size=64, validation_split=0.2, callbacks=[early_stopping])
    # Calcular las predicciones en el conjunto de prueba
    y_pred = model.predict(X_test)
    # Redondear las predicciones si superan el umbral
    y_pred_rounded = (y_pred > threshold).astype('int32')
    # Calcular la precisión en el conjunto de prueba
    accuracy = accuracy_score(y_test, y_pred_rounded)
    return accuracy

def optimize_params_dnn_1(data):
    # Crear un estudio Optuna para la optimización
    study = optuna.create_study(direction='maximize')
    # Función objetivo con el dataset como parámetro fijo
    objective_fn = lambda trial: objective_dnn_1(trial, data)
    # Ejecutar la optimización
    study.optimize(objective_fn, n_trials=20)
    # Obtener los mejores parámetros
    best_params = study.best_params
    best_accuracy = study.best_value
    return best_params, best_accuracy

def optimize_params_dnn_2(data):
    # Crear un estudio Optuna para la optimización
    study = optuna.create_study(direction='maximize')
    # Función objetivo con el dataset como parámetro fijo
    objective_fn = lambda trial: objective_dnn_2(trial, data)
    # Ejecutar la optimización
    study.optimize(objective_fn, n_trials=20)
    # Obtener los mejores parámetros
    best_params = study.best_params
    best_accuracy = study.best_value
    return best_params, best_accuracy

def optimize_params_dnn_3(data):
    # Crear un estudio Optuna para la optimización
    study = optuna.create_study(direction='maximize')
    # Función objetivo con el dataset como parámetro fijo
    objective_fn = lambda trial: objective_dnn_3(trial, data)
    # Ejecutar la optimización
    study.optimize(objective_fn, n_trials=20)
    # Obtener los mejores parámetros
    best_params = study.best_params
    best_accuracy = study.best_value
    return best_params, best_accuracy

def optimize_params(data):
    # Optimización de DNN 1
    best_params_dnn_1, best_accuracy_dnn_1 = optimize_params_dnn_1(data)
    print("Mejores parámetros de DNN 1:", best_params_dnn_1)
    # Optimización de DNN 2
    best_params_dnn_2, best_accuracy_dnn_2 = optimize_params_dnn_2(data)
    print("Mejores parámetros de DNN 2:", best_params_dnn_2)
    # Optimización de DNN 3
    best_params_dnn_3, best_accuracy_dnn_3 = optimize_params_dnn_3(data)
    print("Mejores parámetros de DNN 3:", best_params_dnn_3)
    return best_params_dnn_1, best_params_dnn_2, best_params_dnn_3


def params(data: str):
    data_1d_train = pd.read_csv(data)
    data_1d_train = data_1d_train.dropna()
    dataresult_long_1d_train = file_features(data_1d_train, ds_type="buy")
    dataresult_short_1d_train = file_features(data_1d_train, ds_type="sell")
    best_params_dnn1_long, best_params_dnn2_long, best_params_dnn3_long = optimize_params(dataresult_long_1d_train)
    best_params_dnn1_short, best_params_dnn2_short, best_params_dnn3_short = optimize_params(dataresult_short_1d_train)

    return best_params_dnn1_long, best_params_dnn2_long, best_params_dnn3_long, best_params_dnn1_short, best_params_dnn2_short, best_params_dnn3_short

dnn1_params_1d_long, dnn2_params_1d_long, dnn3_params_1d_long, dnn1_params_1d_short, dnn2_params_1d_short, dnn3_params_1d_short = params("aapl_1d_train.csv")
dnn1_params_1h_long, dnn2_params_1h_long, dnn3_params_1h_long, dnn1_params_1h_short, dnn2_params_1h_short, dnn3_params_1h_short = params("aapl_1h_train.csv")
dnn1_params_1m_long, dnn2_params_1m_long, dnn3_params_1m_long, dnn1_params_1m_short, dnn2_params_1m_short, dnn3_params_1m_short = params("aapl_1m_train.csv")
dnn1_params_5m_long, dnn2_params_5m_long, dnn3_params_5m_long, dnn1_params_5m_short, dnn2_params_5m_short, dnn3_params_5m_short = params("aapl_5m_train.csv")


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


def optimize(trial, strategy, data, global_buy_signals, global_sell_signals):
    portfolio_value = 0

    stop_loss = trial.suggest_float("stop_loss", 0.80, 0.90)
    take_profit = trial.suggest_float("take_profit", 1.01, 1.10)
    n_shares = trial.suggest_int("n_shares", 20, 50)

    strat_params = {}

    buy_signals = pd.DataFrame()
    sell_signals = pd.DataFrame()

    if "dnn1" in strategy:
        buy_signals["dnn1"] = global_buy_signals["predicciones_dnn1"]
        sell_signals["dnn1"] = global_sell_signals["predicciones_dnn1"]

    if "dnn2" in strategy:
        buy_signals["dnn2"] = global_buy_signals["predicciones_dnn2"]
        sell_signals["dnn2"] = global_sell_signals["predicciones_dnn2"]

    if "dnn3" in strategy:
        buy_signals["dnn3"] = global_buy_signals["predicciones_dnn3"]
        sell_signals["dnn3"] = global_sell_signals["predicciones_dnn3"]

    return backtest(data, buy_signals, sell_signals, stop_loss, take_profit, n_shares)

def optimize_file(data):
    data = data.drop(data.index[:30])
    data = data.drop(data.index[-30:])
    data.reset_index(drop=True, inplace=True)
    strategies = list(powerset(["dnn1", "dnn2", "dnn3"]))
    best_strat = None
    best_val = -1
    best_params = None

    for strat in strategies:
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda x: optimize(x, strat, data), n_trials=20)
        value = study.best_value
        if value > best_val:
            best_val = value
            best_strat = strat
            best_params = study.best_params
    print(study.best_value)
    print(best_strat)
    print(best_params)

    return {"file": data,
            "strat": best_strat,
            "value": best_val,
            "params": best_params}