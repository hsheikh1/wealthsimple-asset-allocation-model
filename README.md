# Wealthsimple Replication: Robust Asset Allocation Engine

## 📌 Overview
This project is a quantitative research case study designed to replicate and enhance institutional asset allocation frameworks for retail investors. 

Using **Python (Pandas, NumPy, SciPy)**, I constructed a custom risk engine to backtest two advanced portfolio construction methods against a standard 60/40 benchmark:
1. **Maximum Sharpe Ratio (MSR):** Optimizing for risk-adjusted returns.
2. **Risk Parity (Equal Risk Contribution):** Minimizing tail risk and drawdown depth.

## 🚀 Key Findings
* **Drawdown Protection:** The Risk Parity model demonstrated a significant reduction in maximum drawdown during the 2022 volatility regime compared to the traditional 60/40 portfolio.
* **Efficiency:** While MSR provided higher theoretical returns, the Risk Parity approach offered a smoother equity curve, which is critical for behavioral retention of retail clients.

## 🛠️ Technical Stack
* **Core Logic:** `risk_analytics.py` - A proprietary library containing robust risk metrics (Gaussian VaR, CVaR) and convex optimization functions for portfolio construction.
* **Analysis:** `main_analysis.ipynb` - The driver notebook that pulls live data via `yfinance`, cleans the dataset, and performs the backtest.
* **Data Source:** Real-time adjusted close prices for SPY, EFA, EEM, TLT, and VNQ (Yahoo Finance API).

## 📉 Usage
1. Clone the repository.
2. Ensure dependencies are installed (`pip install yfinance pandas numpy scipy`).
3. Run `main_analysis.ipynb` to pull fresh data and generate the performance report.

---
*Curated by: Haroon Sheikh, CFA*
