import FinanceDataReader as fdr
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from curl_cffi import requests as cffi_requests

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

KR_STATIC_MARKET_CAPS = {
    '삼성전자': 400_000_000_000_000.0,
    'SK하이닉스': 120_000_000_000_000.0,
    'LG에너지솔루션': 80_000_000_000_000.0,
    '삼성전기': 10_000_000_000_000.0,
    '현대차': 50_000_000_000_000.0,
    '한화에어로스페이스': 15_000_000_000_000.0,
    '두산에너빌리티': 12_000_000_000_000.0,
    '한미반도체': 10_000_000_000_000.0,
    'LG화학': 25_000_000_000_000.0,
    '고려아연': 15_000_000_000_000.0,
    '삼성생명': 15_000_000_000_000.0,
    '삼성물산': 25_000_000_000_000.0,
    '미래에셋증권': 4_000_000_000_000.0,
    '현대건설': 3_500_000_000_000.0,
    '한국전력': 13_000_000_000_000.0
}

def fetch_kr_market_caps():
    mcaps = {}
    for name, code in KR_TICKERS.items():
        fallback_val = KR_STATIC_MARKET_CAPS.get(name, 1.0)
        try:
            ticker_yf = f"{code}.KS"
            info = yf.Ticker(ticker_yf).info
            mcap = info.get('marketCap')
            if mcap:
                mcaps[name] = float(mcap)
            else:
                mcaps[name] = fallback_val
        except Exception as e:
            print(f"Error fetching KR market cap for {name} ({code}): {e}")
            mcaps[name] = fallback_val
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


def fetch_vkospi(start_date="2025-01-01"):
    """Fetch daily VKOSPI close series from Investing.com."""
    try:
        end_date = datetime.today().strftime("%Y-%m-%d")
        resp = cffi_requests.get(
            "https://api.investing.com/api/financialdata/historical/956761",
            impersonate="chrome",
            headers={"domain-id": "www"},
            params={
                "start-date": start_date,
                "end-date": end_date,
                "time-frame": "Daily",
                "add-missing-rows": "false",
            },
            timeout=10,
        )
        resp.raise_for_status()
        rows = resp.json().get("data") or []
        if not rows:
            return pd.Series(dtype=float, name="VKOSPI")
        df = pd.DataFrame(rows)
        dates = pd.to_datetime(df["rowDateTimestamp"]).dt.tz_localize(None).dt.normalize()
        values = pd.to_numeric(df["last_close"].astype(str).str.replace(",", ""), errors="coerce")
        series = pd.Series(values.values, index=dates, name="VKOSPI").dropna().sort_index()
        return series[~series.index.duplicated(keep="last")]
    except Exception as e:
        print(f"Error fetching VKOSPI: {e}")
        return pd.Series(dtype=float, name="VKOSPI")


def fetch_vix(start_date="2025-01-01"):
    """Fetch daily CBOE VIX (^VIX) close series from Yahoo Finance."""
    try:
        df = yf.download("^VIX", start=start_date, progress=False)
        if df.empty or "Close" not in df.columns:
            return pd.Series(dtype=float, name="VIX")
        close_col = df["Close"]
        if isinstance(close_col, pd.DataFrame):
            close_col = close_col.iloc[:, 0]
        dates = pd.to_datetime(close_col.index).tz_localize(None).normalize()
        series = pd.Series(close_col.values, index=dates, name="VIX").dropna().sort_index()
        return series[~series.index.duplicated(keep="last")]
    except Exception as e:
        print(f"Error fetching VIX: {e}")
        return pd.Series(dtype=float, name="VIX")


def fetch_kospi_index(start_date="2025-01-01"):
    """Fetch daily KOSPI index (KS11) close series."""
    try:
        df = fdr.DataReader("KS11", start_date)
        if df.empty or "Close" not in df.columns:
            return pd.Series(dtype=float, name="KOSPI")
        dates = pd.to_datetime(df.index).tz_localize(None).normalize()
        series = pd.Series(df["Close"].values, index=dates, name="KOSPI").dropna().sort_index()
        return series[~series.index.duplicated(keep="last")]
    except Exception as e:
        print(f"Error fetching KOSPI: {e}")
        return pd.Series(dtype=float, name="KOSPI")


def fetch_nasdaq100_index(start_date="2025-01-01"):
    """Fetch daily NASDAQ 100 index (^NDX) close series."""
    try:
        df = yf.download("^NDX", start=start_date, progress=False)
        if df.empty or "Close" not in df.columns:
            return pd.Series(dtype=float, name="NASDAQ 100")
        close_col = df["Close"]
        if isinstance(close_col, pd.DataFrame):
            close_col = close_col.iloc[:, 0]
        dates = pd.to_datetime(close_col.index).tz_localize(None).normalize()
        series = pd.Series(close_col.values, index=dates, name="NASDAQ 100").dropna().sort_index()
        return series[~series.index.duplicated(keep="last")]
    except Exception as e:
        print(f"Error fetching NASDAQ 100: {e}")
        return pd.Series(dtype=float, name="NASDAQ 100")
