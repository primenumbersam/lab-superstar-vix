# Superstar VIX: Customized Volatility Index (CVI)

Welcome to **Project Superstar VIX**, a financial dashboard built with Streamlit. This mini app calculates a **Customized Volatility Index (CVI)** for high-performing "Superstar" stocks in both the **Republic of Korea (KR)** and the **United States (US)** markets.

## 🚀 Deployment
This application is designed to be hosted on **Streamlit Community Cloud**. 

## 🛠 Features
- **Real-time CVI Calculation:** Using a Recursive EWMA model ($\lambda=0.94$).
- **Stateful Engine:** SQLite-backed state management for efficient incremental updates.
- **Robust Covariance:** Implements **Ledoit-Wolf Shrinkage** for stable correlation matrices.
- **Hierarchical Visualization:** Sector-based Treemaps and Risk-Return Profile scatterplots.
- **Standardized Comparison:** Fixed Y-axis (0-100) for direct cross-market comparison.

## 📊 Methodology
### 1. Volatility (EWMA)
Individual stock volatility is calculated using the Recursive Exponentially Weighted Moving Average formula:
$$\sigma_{i,t}^2 = \lambda \sigma_{i,t-1}^2 + (1 - \lambda) r_{i,t-1}^2$$

### 2. Aggregation
The index is aggregated using **Market Capitalization weights** ($w_t$):
$$\sigma_{port, t} = \sqrt{w_t^T (D_t R_t D_t) w_t}$$
Where $R_t$ is the Ledoit-Wolf shrunk correlation matrix.

## 💻 Tech Stack
- **Framework:** Streamlit
- **Data:** FinanceDataReader (KR), yfinance (US)
- **Math/Stats:** Pandas, NumPy, Scikit-learn (Ledoit-Wolf)
- **Viz:** Plotly

## ⚙️ Local Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/lab-superstar-vix.git
   cd lab-superstar-vix
   ```
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the app:
   ```bash
   streamlit run app.py
   ```

## 📄 License
MIT License. Feel free to use and modify for your own research.
