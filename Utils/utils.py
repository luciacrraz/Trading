import pandas as pd
import mplfinance as mpf
import warnings



def candle_chart(file_path: str):
    warnings.filterwarnings("ignore")
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return mpf.plot(df, type='candle', style='charles', title='Candlestick Chart', ylabel='Price')

candle = candle_chart('../data/aapl_1d_train.csv')


