import yfinance as yf
import pandas as pd

# Coletando dado do BTC-USD
dados = yf.download(
    tickers=["DOGE-USD","BTC-USD"],
    period="2y",
    interval="1d"
)

# Salvando os dados em um CSV
dados.to_csv("dados.csv")