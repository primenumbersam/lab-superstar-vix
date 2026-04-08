import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

def calculate_returns(prices_df):
    """Calculates log returns and fills NaNs"""
    returns = np.log(prices_df / prices_df.shift(1)).dropna()
    # Outlier clipping
    lower = returns.quantile(0.01)
    upper = returns.quantile(0.99)
    returns = returns.clip(lower=lower, upper=upper, axis=1)
    return returns

def initialize_variance(returns_df, initial_window=29):
    """Calculates sample variance for the first initial_window days."""
    if len(returns_df) < initial_window:
        return returns_df.var()
    return returns_df.iloc[:initial_window].var()

def update_ewma(prev_vars, return_t, lambda_factor=0.94):
    """
    Updates EWMA variance for one step.
    prev_vars: pd.Series
    return_t: pd.Series
    """
    return lambda_factor * prev_vars + (1 - lambda_factor) * (return_t ** 2)

def compute_ledoit_wolf_correlation(returns_df):
    """
    Computes Ledoit-Wolf Shrinkage correlation.
    """
    lw = LedoitWolf()
    cov_lw = lw.fit(returns_df).covariance_
    
    # Convert Covariance to Correlation
    v = np.sqrt(np.diag(cov_lw))
    outer_v = np.outer(v, v)
    corr = cov_lw / outer_v
    corr[cov_lw == 0] = 0
    return pd.DataFrame(corr, index=returns_df.columns, columns=returns_df.columns)

def calculate_portfolio_volatility(ewma_vars, correlation_matrix, mcaps_dict):
    """
    Calculates portfolio volatility D * R * D
    """
    tickers = ewma_vars.index.intersection(correlation_matrix.index)
    
    ewma_vars = ewma_vars[tickers]
    correlation_matrix = correlation_matrix.loc[tickers, tickers]
    
    # Create weights
    total_mcap = sum(mcaps_dict.get(t, 1.0) for t in tickers)
    weights = np.array([mcaps_dict.get(t, 1.0) / total_mcap for t in tickers])
    
    # D matrix (diagonal of std devs)
    std_devs = np.sqrt(ewma_vars.values)
    D = np.diag(std_devs)
    
    # Covariance = D * R * D
    R = correlation_matrix.values
    cov_matrix = D @ R @ D
    
    # Port variance
    port_var = weights.T @ cov_matrix @ weights
    port_vol = np.sqrt(port_var)
    
    # Annualize (approx 252 trading days)
    return port_vol * np.sqrt(252)
