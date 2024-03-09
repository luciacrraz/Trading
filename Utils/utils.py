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

warnings.filterwarnings("ignore")

data_1m_test = pd.read_csv("aapl_1d_train.csv")
data_1m_test = data_1m_test.dropna()
def candle_chart(file_path: str):
    warnings.filterwarnings("ignore")
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return mpf.plot(df, type='candle', style='charles', title='Candlestick Chart', ylabel='Price')

candle_test = candle_chart("../data/aapl_1m_test.csv")

def file_features(data, ds_type: str):
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

data_test_long = file_features("../data/aapl_1m_test.csv", ds_type="buy")
data_test_buy = file_features("../data/aapl_1m_test.csv", ds_type="sell")

def buy_signals(data):
    buy_signals = pd.DataFrame()
    # Selecciona las características
    X = data.iloc[:, :-1]
    # Selecciona la variable objetivo
    y = data.iloc[:, -1]

    # Crear modelos con los mejores parámetros encontrados para cada algoritmo
    best_logistic_model = LogisticRegression(penalty='l2', C=0.010405629107406833, solver='liblinear')
    best_svm_model = SVC(C=0.02891729617292502, kernel='linear', gamma='scale')
    best_xgboost_model = XGBClassifier(n_estimators=300, max_depth=9, learning_rate=0.02572836520302602, subsample=0.8,
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
    buy_signals['predicciones_lr'] = predictions_lr
    buy_signals['predicciones_svm'] = predictions_svm
    buy_signals['predicciones_xgboost'] = predictions_xgboost_bool

    return buy_signals

def sell_signals(data):
    sell_signals = pd.DataFrame()
    # Selecciona las características
    X = data.iloc[:, :-1]
    # Selecciona la variable objetivo
    y = data.iloc[:, -1]

    # Crear modelos con los mejores parámetros encontrados para cada algoritmo
    best_logistic_model = LogisticRegression(penalty='l1', C=0.002778605059659017, solver='saga')
    best_svm_model = SVC(C=0.28120622947860485, kernel='rbf', gamma='auto')
    best_xgboost_model = XGBClassifier(n_estimators=400, max_depth=3, learning_rate=0.11719371112820326, subsample=0.7,
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

global_buy_signals = buy_signals(data_test_long)
global_sell_signals = sell_signals(data_test_buy)

buy_sell_xgboost = pd.DataFrame()
buy_sell_xgboost['pred_xg_buy'] = global_buy_signals['predicciones_xgboost']
buy_sell_xgboost['pred_xg_sell'] = global_sell_signals['predicciones_xgboost']
buy_sell_xgboost['Close'] = data_test_long['Close_Lag0']

# Crear una figura y un conjunto de ejes
fig, ax = plt.subplots(figsize=(10, 6))

# Graficar el precio de cierre
ax.plot(buy_sell_xgboost['Close'], label='Precio de Cierre', color='black')

# Marcar las señales de compra y venta
ax.scatter(buy_sell_xgboost.index[buy_sell_xgboost['pred_xg_buy']], buy_sell_xgboost['Close'][buy_sell_xgboost['pred_xg_buy']], marker='^', color='r', label='Compra')
ax.scatter(buy_sell_xgboost.index[buy_sell_xgboost['pred_xg_sell']], buy_sell_xgboost['Close'][buy_sell_xgboost['pred_xg_sell']], marker='v', color='g', label='Venta')

# Agregar leyendas, títulos, y etiquetas de ejes
ax.set_title('Señales de Compra y Venta')
ax.set_xlabel('Índice de Tiempo')
ax.set_ylabel('Precio')
ax.legend()

# Mostrar el gráfico
plt.grid(True)
plt.show()
#("../data/aapl_1m_test.csv")