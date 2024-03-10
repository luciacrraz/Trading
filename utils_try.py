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

def candle_chart(file_path: str):
    warnings.filterwarnings("ignore")
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])  # Cambio aquí el nombre de la columna
    df.set_index('Date', inplace=True)  # Cambio aquí el nombre de la columna
    return mpf.plot(df, type='candle', style='charles', title='Candlestick Chart', ylabel='Price')

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

def data_fun(file_path: str):
    data = pd.read_csv(file_path)
    data = data.dropna()
    data = data.drop(data.index[:30])
    data = data.drop(data.index[-30:])
    data.reset_index(drop=True, inplace=True)
    return data

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
    plt.plot([0,700], [1000000,1000000], label="Pasive_Strategy")
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Estrategia Pasiva VS Estrategia Trading')
    plt.legend()
    plt.show()

file_path = "data/aapl_1d_test.csv"
# explicar el data set
data_1m_test = pd.read_csv(file_path)
data_1m_test = data_1m_test.dropna()
# grafica de vela del precio close sin hacer nada
candle_test = candle_chart(file_path)
# variables que usamos para la prediccion
data_test_long = file_features(data_1m_test, ds_type="buy")
data_test_buy = file_features(data_1m_test, ds_type="sell")
# dataframes de senales de compra
global_buy_signals = buy_signals(data_test_long)
global_sell_signals = sell_signals(data_test_buy)
# grafica de compra venta conforme al precio de cierre
plot_buy_sell_signals(global_buy_signals, global_sell_signals, data_test_long)
#data para el backtest
data_1m = data_fun(file_path)
# valores del portafolio, dinero y parametros
portfolio_values, cash_values, operations_history,  = backtest(data_1m, global_buy_signals["predicciones_xgboost"], global_sell_signals["predicciones_xgboost"], 0.88, 1.05, 39)
# grafica con las operaciones
plot_operations = plot_operations_history(operations_history)
# grafica con el dinero atraves del tiempo
cash_plot = plot_cash(cash_values)
#grafica con el valor del portafolio atraves del tiempo
plot_port_value = port_value_plot(portfolio_values)
# grafica comparando el dinero con el portafolio
cash_port = cash_portvalue_plot(cash_values, portfolio_values)
#comparacion con estrategia pasiva:
comparacion = pasive_portvalue_plot(portfolio_values)