import yfinance as yf
import pandas as pd
import streamlit as st

def download_data(tickers, start_date, end_date):
    """Baixar dados das ações e tratar valores ausentes"""
    try:
        data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)['Adj Close']
        if isinstance(data, pd.Series):
            data = pd.DataFrame(data)
        data = data.dropna(how='all')  # Remover colunas com todos os valores ausentes
        return data
    except Exception as e:
        st.error(f"Erro ao baixar dados: {str(e)}")
        return None