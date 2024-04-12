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
    best_logistic_model = LogisticRegression(penalty='l2', C=0.6431546383805156, solver='liblinear')
    best_svm_model = SVC(C=0.0035454674239493874, kernel='rbf', gamma='scale')
    best_xgboost_model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.030090107781624707, subsample=0.9,
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
    best_logistic_model = LogisticRegression(penalty='l1', C=0.0021323714279392757, solver='saga')
    best_svm_model = SVC(C=0.04584664364591048, kernel='poly', degree= 2, gamma='auto')
    best_xgboost_model = XGBClassifier(n_estimators=900, max_depth=9, learning_rate=0.2220589262776878, subsample=0.5,
                                       colsample_bytree=0.7)

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
    best_logistic_model = LogisticRegression(penalty='l2', C=0.0028315415752095535, solver='saga')
    best_svm_model = SVC(C=0.2018232497206618, kernel='rbf', gamma='auto')
    best_xgboost_model = XGBClassifier(n_estimators=100, max_depth=10, learning_rate=0.102602018285415, subsample=0.5,
                                       colsample_bytree=0.6)

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
    best_logistic_model = LogisticRegression(penalty='l2', C=0.0015827990076342706, solver='saga')
    best_svm_model = SVC(C=208.31110635195978, kernel='rbf', gamma='auto')
    best_xgboost_model = XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.016335181464666, subsample=0.6,
                                       colsample_bytree=0.5)

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
    best_logistic_model = LogisticRegression(penalty='l2', C=0.002105767490569824, solver='saga')
    best_svm_model = SVC(C=89.33147882881373, kernel='linear')
    best_xgboost_model = XGBClassifier(n_estimators=400, max_depth=3, learning_rate=0.061659645644687434, subsample=0.8,
                                       colsample_bytree=0.7)

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
    best_logistic_model = LogisticRegression(penalty='l2', C=0.002395829495054093, solver='liblinear')
    best_svm_model = SVC(C= 0.002176996974479978, kernel='sigmoid', gamma='scale')
    best_xgboost_model = XGBClassifier(n_estimators=1000, max_depth=9, learning_rate=0.014804455005781484, subsample=0.9,
                                       colsample_bytree=0.7)

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
    best_logistic_model = LogisticRegression(penalty='l1', C=0.0030591725482995644, solver='saga')
    best_svm_model = SVC(C=0.6739747156890435, kernel='linear')
    best_xgboost_model = XGBClassifier(n_estimators=900, max_depth=9, learning_rate=0.08242884326336312, subsample=0.5,
                                       colsample_bytree=1.0)

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