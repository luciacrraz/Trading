import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
def buy_signals(data, best_dnn1_params, best_dnn2_params, best_dnn3_params):
    buy_signals = pd.DataFrame()
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    input_shape = X.shape[1]
    threshold = 0.5

    # Crear y compilar modelo DNN1
    best_dnn1_model = create_and_compile_dnn_model(input_shape, best_dnn1_params)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    best_dnn1_model.fit(X, y, epochs=30, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
    y_pred_dnn1 = best_dnn1_model.predict(X)
    y_pred_rounded_dnn1 = (y_pred_dnn1 > threshold).astype('int32')

    # Crear y compilar modelo DNN2
    best_dnn2_model = create_and_compile_dnn_model(input_shape, best_dnn2_params)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    best_dnn2_model.fit(X, y, epochs=30, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
    y_pred_dnn2 = best_dnn2_model.predict(X)
    y_pred_rounded_dnn2 = (y_pred_dnn2 > threshold).astype('int32')

    # Crear y compilar modelo DNN3
    best_dnn3_model = create_and_compile_dnn_model(input_shape, best_dnn3_params)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    best_dnn3_model.fit(X, y, epochs=30, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
    y_pred_dnn3 = best_dnn3_model.predict(X)
    y_pred_rounded_dnn3 = (y_pred_dnn3 > threshold).astype('int32')

    # Agregar las predicciones como nuevas columnas al conjunto de datos original
    buy_signals['predicciones_dnn1'] = pd.Series(y_pred_rounded_dnn1.flatten())
    buy_signals['predicciones_dnn2'] = pd.Series(y_pred_rounded_dnn2.flatten())
    buy_signals['predicciones_dnn3'] = pd.Series(y_pred_rounded_dnn3.flatten())

    return buy_signals


def sell_signals(data, best_dnn1_params, best_dnn2_params, best_dnn3_params):
    sell_signals = pd.DataFrame()
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    input_shape = X.shape[1]
    threshold = 0.5

    # Crear y compilar modelo DNN1
    best_dnn1_model = create_and_compile_dnn_model(input_shape, best_dnn1_params)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    best_dnn1_model.fit(X, y, epochs=30, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
    y_pred_dnn1 = best_dnn1_model.predict(X)
    y_pred_rounded_dnn1 = (y_pred_dnn1 > threshold).astype('int32')

    # Crear y compilar modelo DNN2
    best_dnn2_model = create_and_compile_dnn_model(input_shape, best_dnn2_params)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    best_dnn2_model.fit(X, y, epochs=30, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
    y_pred_dnn2 = best_dnn2_model.predict(X)
    y_pred_rounded_dnn2 = (y_pred_dnn2 > threshold).astype('int32')

    # Crear y compilar modelo DNN3
    best_dnn3_model = create_and_compile_dnn_model(input_shape, best_dnn3_params)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    best_dnn3_model.fit(X, y, epochs=30, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
    y_pred_dnn3 = best_dnn3_model.predict(X)
    y_pred_rounded_dnn3 = (y_pred_dnn3 > threshold).astype('int32')

    # Agregar las predicciones como nuevas columnas al conjunto de datos original
    sell_signals['predicciones_dnn1'] = pd.Series(y_pred_rounded_dnn1.flatten())
    sell_signals['predicciones_dnn2'] = pd.Series(y_pred_rounded_dnn2.flatten())
    sell_signals['predicciones_dnn3'] = pd.Series(y_pred_rounded_dnn3.flatten())

    return sell_signals