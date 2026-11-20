import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from state_manager import StateManager
from data_pipeline import (
    fetch_kr_prices, fetch_us_prices, fetch_kr_market_caps, fetch_us_market_caps,
    fetch_vkospi, fetch_vix, fetch_kospi_index, fetch_nasdaq100_index,
    KR_SECTORS, US_SECTORS
)
from ewma_model import calculate_returns, initialize_variance, update_ewma, compute_ledoit_wolf_correlation, calculate_portfolio_volatility

# Page Layout configuration
st.set_page_config(page_title="Superstar VIX", layout="wide")

# Custom CSS to control layout and font sizes
st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem;
        }
        /* Customize st.title font size */
        h1 {
            font-size: 1.8rem !important;
        }
        /* Customize st.header font size */
        h2 {
            font-size: 1.6rem !important;
        }
        /* Optional: Customize st.subheader font size */
        h3 {
            font-size: 1.3rem !important;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Volatility Index of Superstar Firms")

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

@st.cache_data(show_spinner=False)
def get_implied_vol(market_name):
    if market_name == "KR":
        return fetch_vkospi("2025-01-01")
    return fetch_vix("2025-01-01")


@st.cache_data(show_spinner=False)
def get_benchmark_index(market_name):
    if market_name == "KR":
        return fetch_kospi_index("2025-01-01")
    return fetch_nasdaq100_index("2025-01-01")


def process_market(market_name, df_cvi, mcaps, current_vars, returns, prices, sector_dict):
    st.header(f"{market_name} Superstar VIX")
    
    if df_cvi is None or df_cvi.empty:
        st.error(f"Failed to load or calculate {market_name} data.")
        return
        
    # Latest CVI
    latest_cvi = df_cvi[market_name].iloc[-1]
    st.metric(label=f"Latest {market_name} CVI", value=f"{latest_cvi:.2f}")
    
    # Time series chart (Dual axis: Left = Benchmark Index, Right = CVI & Official Implied Vol)
    iv_name = "VKOSPI" if market_name == "KR" else "VIX"
    bench_name = "KOSPI" if market_name == "KR" else "NASDAQ 100"
    market_color = "#EF553B" if market_name == "KR" else "#636EFA"
    bench_color = "#FECB52"
    
    iv_series = get_implied_vol(market_name)
    bench_series = get_benchmark_index(market_name)
    
    df_plot = df_cvi.copy()
    df_plot["Date"] = pd.to_datetime(df_plot["Date"]).dt.tz_localize(None).dt.normalize()
    if not iv_series.empty:
        df_plot = df_plot.merge(
            iv_series.rename_axis("Date").reset_index(),
            on="Date",
            how="left",
        )
    if not bench_series.empty:
        df_plot = df_plot.merge(
            bench_series.rename_axis("Date").reset_index(),
            on="Date",
            how="left",
        )
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if bench_name in df_plot.columns and not df_plot[bench_name].dropna().empty:
        fig.add_trace(
            go.Scatter(
                x=df_plot["Date"],
                y=df_plot[bench_name],
                name=bench_name,
                mode="lines",
                line=dict(color=bench_color),
                connectgaps=True,
            ),
            secondary_y=False,
        )
    fig.add_trace(
        go.Scatter(
            x=df_plot["Date"],
            y=df_plot[market_name],
            name=f"{market_name} CVI",
            mode="lines",
            line=dict(color=market_color),
        ),
        secondary_y=True,
    )
    if iv_name in df_plot.columns and not df_plot[iv_name].dropna().empty:
        fig.add_trace(
            go.Scatter(
                x=df_plot["Date"],
                y=df_plot[iv_name],
                name=iv_name,
                mode="lines",
                line=dict(color=market_color, dash="dot"),
                connectgaps=True,
            ),
            secondary_y=True,
        )
    fig.update_layout(
        title=f"{market_name} CVI & {iv_name} vs {bench_name}",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text=f"{bench_name} Index", secondary_y=False)
    fig.update_yaxes(title_text=f"Volatility ({market_name} CVI / {iv_name})", range=[0, 100], secondary_y=True)
    st.plotly_chart(fig, width="stretch")
    
    # -----------------------------------
    # Supplemental Visualizations
    # -----------------------------------
    st.subheader(f"{market_name} Market Breakdown (Latest)")
    
    # Pre-calculate data for visualizations
    total_mcap = sum(mcaps.values())
    latest_vols = np.sqrt(current_vars) * np.sqrt(252) * 100 # Annualized volatility per series
    
    # Calculate Mean Annualized Returns (252 days)
    mean_returns = returns.mean() * 252 * 100
    
    start_date = returns.index[0].strftime('%Y-%m-%d')
    end_date = returns.index[-1].strftime('%Y-%m-%d')
    trading_days = len(returns)
    
    chart_data = []
    for ticker, mcap in mcaps.items():
        weight = (mcap / total_mcap) * 100
        volatility = latest_vols.get(ticker, 0)
        mean_ret = mean_returns.get(ticker, 0)
        sector = sector_dict.get(ticker, "Unknown")
        chart_data.append({
            "Ticker": ticker,
            "Sector": sector,
            "Market Cap": mcap,
            "Relative Market Cap Weight (%)": weight,
            "Annualized Volatility (%)": volatility,
            "Annualized Mean Return (%)": mean_ret
        })
    df_chart = pd.DataFrame(chart_data)
    
    col1, col2 = st.columns(2)
    with col1:
        # Treemap: Size = Relative Cap Weight, Color = Annualized Mean Return
        fig_tree = px.treemap(
            df_chart, 
            path=[px.Constant(f"{market_name} Market"), "Sector", "Ticker"], 
            values="Relative Market Cap Weight (%)",
            color="Annualized Mean Return (%)",
            color_continuous_scale="RdYlGn",
            color_continuous_midpoint=0.0,
            hover_data={
                "Relative Market Cap Weight (%)": ":.2f",
                "Annualized Mean Return (%)": ":.2f",
                "Annualized Volatility (%)": ":.2f"
            },
            title=f"{market_name} Treemap: Cap Weight (Size) & Mean Return (Color) [{start_date} ~ {end_date}]"
        )
        st.plotly_chart(fig_tree, width="stretch")
        
    with col2:
        # Scatterplot: Std Dev vs Mean Return
        fig_scatter = px.scatter(
            df_chart, 
            x="Annualized Volatility (%)", 
            y="Annualized Mean Return (%)", 
            color="Sector",
            hover_name="Ticker",
            size="Relative Market Cap Weight (%)",
            labels={
                "Annualized Volatility (%)": f"Annualized Volatility (%) [As of {end_date}]",
                "Annualized Mean Return (%)": f"Annualized Mean Return (%) [{start_date} ~ {end_date}]",
                "Relative Market Cap Weight (%)": "Relative Cap Weight (%)"
            },
            title=f"Risk-Return Profile ({start_date} ~ {end_date}, {trading_days} Trading Days)"
        )
        st.plotly_chart(fig_scatter, width="stretch")
    
    st.caption(
        fr"**Calculation Window:** Evaluated from {start_date} to {end_date} ({trading_days} trading days). "
        fr"Annualized Mean Return represents average log return scaled by 252. "
        fr"Annualized Volatility is calculated using Recursive EWMA ($\lambda=0.94$) scaled by $\sqrt{{252}}$ as of {end_date}. "
        fr"Tile size reflects the relative market capitalization weight within the {market_name} Superstar universe."
    )

# Pre-fetching and calculating all data
with st.spinner("Synchronizing Market Data..."):
    kr_df, kr_mcaps, kr_vars, kr_rets, kr_prices = get_cvi_data("KR", fetch_kr_prices, fetch_kr_market_caps)
    us_df, us_mcaps, us_vars, us_rets, us_prices = get_cvi_data("US", fetch_us_prices, fetch_us_market_caps)

tab1, tab2, tab3 = st.tabs(["Comparison", "KR Market", "US Market"])

with tab1:
    st.header("KR vs US Superstar VIX")
    
    if kr_df is not None and us_df is not None:
        # Merge for combined plot (single axis: KR CVI, VKOSPI, US CVI, VIX)
        combined_df = pd.merge(kr_df, us_df, on="Date", how="inner")
        combined_df["Date"] = pd.to_datetime(combined_df["Date"]).dt.tz_localize(None).dt.normalize()
        vkospi_series = get_implied_vol("KR")
        vix_series = get_implied_vol("US")
        if not vkospi_series.empty:
            combined_df = combined_df.merge(vkospi_series.rename_axis("Date").reset_index(), on="Date", how="left")
        if not vix_series.empty:
            combined_df = combined_df.merge(vix_series.rename_axis("Date").reset_index(), on="Date", how="left")

        fig_combined = go.Figure()
        fig_combined.add_trace(go.Scatter(
            x=combined_df["Date"], y=combined_df["KR"],
            name="KR CVI", mode="lines", line=dict(color="#EF553B"),
        ))
        if "VKOSPI" in combined_df.columns and not combined_df["VKOSPI"].dropna().empty:
            fig_combined.add_trace(go.Scatter(
                x=combined_df["Date"], y=combined_df["VKOSPI"],
                name="VKOSPI", mode="lines", line=dict(color="#EF553B", dash="dot"), connectgaps=True,
            ))
        fig_combined.add_trace(go.Scatter(
            x=combined_df["Date"], y=combined_df["US"],
            name="US CVI", mode="lines", line=dict(color="#636EFA"),
        ))
        if "VIX" in combined_df.columns and not combined_df["VIX"].dropna().empty:
            fig_combined.add_trace(go.Scatter(
                x=combined_df["Date"], y=combined_df["VIX"],
                name="VIX", mode="lines", line=dict(color="#636EFA", dash="dot"), connectgaps=True,
            ))
        fig_combined.update_layout(
            title="Standardized Superstar VI: KR vs US Stocks",
            hovermode="x unified",
            yaxis=dict(title="Volatility Index", range=[0, 100]),
            xaxis=dict(title="Date"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_combined, width="stretch")
    else:
        st.warning("Could not generate combined chart. Data synchronization issue.")

    st.divider()
    st.subheader("Methodology")
    
    st.markdown(r"""
    ### 1. Individual Volatility: Recursive EWMA
    Each asset's volatility is calculated using a **Recursive Exponentially Weighted Moving Average (EWMA)**. 
    Unlike simple historical volatility, EWMA gives more weight to recent data, allowing the index to react faster to sudden market shifts.
    
    $$ \sigma_{i,t}^2 = \lambda \sigma_{i,t-1}^2 + (1 - \lambda) r_{i,t-1}^2 $$
    
    - **$\lambda$ (Decay Factor):** 0.94 (standard for daily data).
    - **Initialization:** We use the first 29 days of the window to calculate a baseline sample variance.
    
    ### 2. Correlation Matrix: Ledoit-Wolf Shrinkage
    To combine individual volatilities into a portfolio index, we need a correlation matrix ($R_t$). 
    A simple correlation matrix is often "noisy" or mathematically unstable. We apply **Ledoit-Wolf Shrinkage**, which "shrinks" the sample correlation toward a more stable target.
    This ensures that the VI remains robust even when market liquidty or correlations change abruptly.
    
    ### 3. Portfolio Aggregation: Market Cap Weighting
    The final index is aggregated by weighting each asset by its relative **Market Capitalization**. 
    This ensures the VI represents the "Superstar" market as a whole, rather than just an average of tickers.
    
    $$ \sigma_{port, t} = \sqrt{w_t^T (D_t R_t D_t) w_t} $$
    
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
