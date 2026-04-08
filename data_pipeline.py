import FinanceDataReader as fdr
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

KR_SECTORS = {
    '삼성전자': '전기 전자', 'SK하이닉스': '전기 전자', 'LG에너지솔루션': '전기 전자', '삼성전기': '전기 전자',
    '현대차': '운송장비 부품', '한화에어로스페이스': '운송장비 부품',
    '두산에너빌리티': '기계 장비', '한미반도체': '기계 장비',
    'LG화학': '화학', '고려아연': '금속',
    '삼성생명': '보험', '삼성물산': '유통',
    '미래에셋증권': '증권', '현대건설': '건설',
    '한국전력': '전기 가스'
}

KR_TICKERS = {
    '삼성전자': '005930', 'SK하이닉스': '000660', 'LG에너지솔루션': '373220', '삼성전기': '009150',
    '현대차': '005380', '한화에어로스페이스': '012450',
    '두산에너빌리티': '034020', '한미반도체': '042700',
    'LG화학': '051910', '고려아연': '010130',
    '삼성생명': '032830', '삼성물산': '028260',
    '미래에셋증권': '006800', '현대건설': '000720',
    '한국전력': '015760'
}

US_SECTORS = {
    'NVDA': 'Electronic Technology', 'AVGO': 'Electronic Technology', 'MU': 'Electronic Technology', 'AAPL': 'Electronic Technology', 'AMD': 'Electronic Technology', 'INTC': 'Electronic Technology',
    'GOOGL': 'Technology Services', 'MSFT': 'Technology Services', 'META': 'Technology Services', 'ORCL': 'Technology Services', 'PLTR': 'Technology Services', 'NET': 'Technology Services',
    'LLY': 'Health Technology', 'JNJ': 'Health Technology',
    'XOM': 'Energy Minerals', 'CVX': 'Energy Minerals',
    'JPM': 'Finance', 'UNH': 'Health Services', 'AMZN': 'Retail Trade', 'TSLA': 'Consumer Durables'
}

US_TICKERS = list(US_SECTORS.keys())

def fetch_kr_prices(start_date="2025-01-01"):
    df_list = []
    for name, code in KR_TICKERS.items():
        try:
            df = fdr.DataReader(code, start_date)
            df = df[['Close']].rename(columns={'Close': name})
            df_list.append(df)
        except Exception as e:
            print(f"Error fetching {name}: {e}")
    
    if not df_list:
        return pd.DataFrame()
    return pd.concat(df_list, axis=1).dropna()

def fetch_us_prices(start_date="2025-01-01"):
    df_list = []
    for ticker in US_TICKERS:
        try:
            df = yf.download(ticker, start=start_date, progress=False)
            if 'Close' in df.columns:
                close_col = df['Close']
                # Handle MultiIndex column if yfinance returns it
                if isinstance(close_col, pd.DataFrame):
                    close_col = close_col.iloc[:, 0]
                close_col.name = ticker
                df_list.append(close_col)
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
            
    if not df_list:
        return pd.DataFrame()
    return pd.concat(df_list, axis=1).dropna()

def fetch_kr_market_caps():
    krx = fdr.StockListing('KRX')
    mcaps = {}
    for name, code in KR_TICKERS.items():
        row = krx[krx['Code'] == code]
        if not row.empty and 'Marcap' in row.columns:
            mcaps[name] = float(row['Marcap'].iloc[0])
        else:
            mcaps[name] = 1.0 # Fallback
    return mcaps

def fetch_us_market_caps():
    mcaps = {}
    for ticker in US_TICKERS:
        try:
            info = yf.Ticker(ticker).info
            mcaps[ticker] = info.get('marketCap', 1.0)
        except:
            mcaps[ticker] = 1.0
    return mcaps
