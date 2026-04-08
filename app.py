import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from state_manager import StateManager
from data_pipeline import fetch_kr_prices, fetch_us_prices, fetch_kr_market_caps, fetch_us_market_caps, KR_SECTORS, US_SECTORS
from ewma_model import calculate_returns, initialize_variance, update_ewma, compute_ledoit_wolf_correlation, calculate_portfolio_volatility

# Page Layout configuration
st.set_page_config(page_title="Superstar VIX", layout="wide")

# Custom CSS to reduce top margin
st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 0rem;
            padding-left: 5rem;
            padding-right: 5rem;
        }
        [data-testid="stHeader"] {
            height: 3rem;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Superstar VIX (Volatility Index)")

@st.cache_resource
def get_state_manager():
    return StateManager()

state = get_state_manager()

@st.cache_data(show_spinner=False)
def get_cvi_data(market_name, _fetch_prices_func, _fetch_mcaps_func):
    prices = _fetch_prices_func("2025-01-01")
    mcaps = _fetch_mcaps_func()
    
    if prices.empty:
        return None, None, None
        
    returns = calculate_returns(prices)
    if len(returns) < 30:
        return None, None, None
        
    init_vars = initialize_variance(returns, 29)
    cv_index = []
    dates = []
    current_vars = init_vars.copy()
    
    for i in range(29, len(returns)):
        curr_date = returns.index[i]
        ret_t = returns.iloc[i]
        current_vars = update_ewma(current_vars, ret_t)
        window_rets = returns.iloc[i-29:i+1]
        corr_matrix = compute_ledoit_wolf_correlation(window_rets)
        port_vol = calculate_portfolio_volatility(current_vars, corr_matrix, mcaps)
        cvi = port_vol * 100 
        dates.append(curr_date)
        cv_index.append(cvi)
        
    df_cvi = pd.DataFrame({"Date": dates, market_name: cv_index})
    return df_cvi, mcaps, current_vars, returns, prices

def process_market(market_name, df_cvi, mcaps, current_vars, returns, prices, sector_dict):
    st.header(f"{market_name} Superstar VIX")
    
    if df_cvi is None or df_cvi.empty:
        st.error(f"Failed to load or calculate {market_name} data.")
        return
        
    # Latest CVI
    latest_cvi = df_cvi[market_name].iloc[-1]
    st.metric(label=f"Latest {market_name} CVI", value=f"{latest_cvi:.2f}")
    
    # Time series chart
    fig = px.line(
        df_cvi, 
        x="Date", 
        y=market_name, 
        title=f"{market_name} Customized Volatility Index (CVI)",
        range_y=[0, 100],
        labels={market_name: "CVI Index"}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # -----------------------------------
    # Supplemental Visualizations
    # -----------------------------------
    st.subheader(f"{market_name} Market Breakdown (Latest)")
    
    # Pre-calculate data for visualizations
    total_mcap = sum(mcaps.values())
    latest_vols = np.sqrt(current_vars) * np.sqrt(252) * 100 # Annualized volatility per series
    
    # Calculate Mean Annualized Returns (252 days)
    mean_returns = returns.mean() * 252 * 100
    
    chart_data = []
    for ticker, mcap in mcaps.items():
        weight = mcap / total_mcap
        volatility = latest_vols.get(ticker, 0)
        mean_ret = mean_returns.get(ticker, 0)
        sector = sector_dict.get(ticker, "Unknown")
        chart_data.append({
            "Ticker": ticker,
            "Sector": sector,
            "Market Cap": mcap,
            "Weight (%)": weight * 100,
            "Volatility (%)": volatility,
            "Mean Return (%)": mean_ret
        })
    df_chart = pd.DataFrame(chart_data)
    
    col1, col2 = st.columns(2)
    with col1:
        # Treemap
        fig_tree = px.treemap(
            df_chart, 
            path=[px.Constant(f"{market_name} Market"), "Sector", "Ticker"], 
            values="Market Cap",
            color="Weight (%)",
            color_continuous_scale="Blues",
            title="Sector & Market Cap Treemap"
        )
        st.plotly_chart(fig_tree, use_container_width=True)
        
    with col2:
        # Scatterplot: Std Dev vs Mean Return
        fig_scatter = px.scatter(
            df_chart, 
            x="Volatility (%)", 
            y="Mean Return (%)", 
            color="Sector",
            hover_name="Ticker",
            size="Weight (%)",
            title="Risk-Return Profile: Individual Volatility vs Mean Return"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    st.caption(f"**Data Window:** Calculation based on returns from {prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')}.")

# Pre-fetching and calculating all data
with st.spinner("Synchronizing Market Data..."):
    kr_df, kr_mcaps, kr_vars, kr_rets, kr_prices = get_cvi_data("KR", fetch_kr_prices, fetch_kr_market_caps)
    us_df, us_mcaps, us_vars, us_rets, us_prices = get_cvi_data("US", fetch_us_prices, fetch_us_market_caps)

tab1, tab2, tab3 = st.tabs(["변동성 비교", "KR Market", "US Market"])

with tab1:
    st.header("Comparison: KR vs US Volatility")
    
    if kr_df is not None and us_df is not None:
        # Merge for combined plot
        combined_df = pd.merge(kr_df, us_df, on="Date", how="inner")
        fig_combined = px.line(
            combined_df, 
            x="Date", 
            y=["KR", "US"], 
            title="Standardized Superstar VI: KR vs US Stocks",
            range_y=[0, 100],
            color_discrete_map={"KR": "#EF553B", "US": "#636EFA"}
        )
        st.plotly_chart(fig_combined, use_container_width=True)
    else:
        st.warning("Could not generate combined chart. Data synchronization issue.")

    st.divider()
    st.subheader("Methodology")
    
    st.markdown("""
    ### 1. Individual Volatility: Recursive EWMA
    Each asset's volatility is calculated using a **Recursive Exponentially Weighted Moving Average (EWMA)**. 
    Unlike simple historical volatility, EWMA gives more weight to recent data, allowing the index to react faster to sudden market shifts.
    
    $$ \\sigma_{i,t}^2 = \\lambda \\sigma_{i,t-1}^2 + (1 - \\lambda) r_{i,t-1}^2 $$
    
    - **$\lambda$ (Decay Factor):** 0.94 (standard for daily data).
    - **Initialization:** We use the first 29 days of the window to calculate a baseline sample variance.
    
    ### 2. Correlation Matrix: Ledoit-Wolf Shrinkage
    To combine individual volatilities into a portfolio index, we need a correlation matrix ($R_t$). 
    A simple correlation matrix is often "noisy" or mathematically unstable. We apply **Ledoit-Wolf Shrinkage**, which "shrinks" the sample correlation toward a more stable target.
    This ensures that the VI remains robust even when market liquidty or correlations change abruptly.
    
    ### 3. Portfolio Aggregation: Market Cap Weighting
    The final index is aggregated by weighting each asset by its relative **Market Capitalization**. 
    This ensures the VI represents the "Superstar" market as a whole, rather than just an average of tickers.
    
    $$ \\sigma_{port, t} = \\sqrt{w_t^T (D_t R_t D_t) w_t} $$
    
    Where $D_t$ is a diagonal matrix of EWMA standard deviations.
    
    ---
    
    ### Intuitive Interpretation: What does Superstar VI tell you?
    
    - **VI ≈ 15-20 (Quiet):** The market expects "business as usual." Price movements are within normal, historical ranges.
    - **VI > 30 (Warning):** Significant market stress. Investors are pricing in large future price swings. This often correlates with market "bottoming" or high-fear periods.
    - **Standardization:** We scale the annualized volatility to a 0-100 range. A VI of **20** in Korea and **20** in the US represent the **same relative level of expected price fluctuation** (standardized to roughly 20% annualized volatility).
    """)

with tab2:
    process_market("KR", kr_df, kr_mcaps, kr_vars, kr_rets, kr_prices, KR_SECTORS)

with tab3:
    process_market("US", us_df, us_mcaps, us_vars, us_rets, us_prices, US_SECTORS)
