def def_trading_strategy(data, n_shares, stop_loss_close, take_profit_close, Umbral, Indicador):
    cash= 1_000_000
    com = 0.125/100
    history = []
    portfolio_value = []
    active_operations = []
    total_profit_loss = 0

    for i, row in data.iterrows():
        # Close active operation
        active_op_temp = []
        for operation in active_operations:
            if operation["stop_loss"] > row.Close:
                cash += (row.Close * operation["n_shares"]) * (1 - com)
                # Calcular la ganancia o pérdida y actualizar el total
                profit_loss = (row.Close * operation["n_shares"]) * (1 - com) - (operation["bought"] * operation["n_shares"])
                total_profit_loss += profit_loss
                history.append({"timestamp": row.Timestamp, "profit_loss": profit_loss})
            elif operation["take_profit"] < row.Close:
                cash += (row.Close * operation["n_shares"]) * (1 - com)
                # Calcular la ganancia o pérdida y actualizar el total
                profit_loss = (row.Close * operation["n_shares"]) * (1 - com) - (operation["bought"] * operation["n_shares"])
                total_profit_loss += profit_loss
                history.append({"timestamp": row.Timestamp, "profit_loss": profit_loss})
            else:
                active_op_temp.append(operation)
        active_operations = active_op_temp
            
        # ¿Tenemos suficiente efectivo?
        if cash < (row.Close * n_shares * (1 + com)):
            asset_vals = sum([operation["n_shares"] * row.Close for operation in active_operations])
            portfolio_value.append(cash + asset_vals)
            continue
        
        # Analizar la señal larga
        if Indicador[i] <= Umbral:  # If ceil(Op)
            active_operations.append({
                "timestamp": row.Timestamp,
                "bought": row.Close,
                "n_shares": n_shares,
                "type": "long",
                "stop_loss": row.Close * stop_loss_close,
                "take_profit": row.Close * take_profit_close
            })
            cash -= row.Close * n_shares * (1 + com)
        
        asset_vals = sum([operation["n_shares"] * row.Close for operation in active_operations])
        portfolio_value.append(cash + asset_vals)
    
    return total_profit_loss

import optuna

def objective(trial):
    # Define los rangos de búsqueda para los parámetros
    n_shares = trial.suggest_int('n_shares', 1, 50)
    stop_loss_close = trial.suggest_uniform('stop_loss_close', 0.80, 1)
    take_profit_close = trial.suggest_uniform('take_profit_close', 1, 1.5)
    Umbral = trial.suggest_uniform('Umbral', -40, 10)

    # Evalúa la función de trading con los parámetros sugeridos
    profit_loss = def_trading_strategy(data, n_shares, stop_loss_close, take_profit_close, Umbral)
    
    return profit_loss  # Devuelve la métrica que se debe minimizar/maximizar