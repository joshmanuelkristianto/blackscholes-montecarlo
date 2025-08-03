import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta, datetime
from models.Black_Scholes_Merton_Model import BlackScholesMertonModel
from models.montecarlosimulation import MonteCarloOptionPricer

# Set page layout
st.set_page_config(layout="wide", page_title="Black-Scholes PnL Heatmap")

st.title("Option Pricing with PnL Heatmap (Black-Scholes Model)")

# Create sidebar inputs
with st.sidebar:
    st.header("📊 Option Parameters")
    spot_price = st.number_input("Current Asset Price", value=100.0, min_value=0.01)
    strike_price = st.number_input("Strike Price", value=100.0, min_value=0.01)
    maturity_days = st.number_input("Time to Maturity (Days)", value=365, min_value=1)
    volatility = st.slider("Volatility (σ)", 0.01, 1.0, 0.2)
    risk_free_rate = st.slider("Risk-Free Rate (r)", 0.0, 0.2, 0.01)

    st.header("🧮 PnL Calculation")
    call_purchase_price = st.number_input("Call Option Purchase Price", value=5.0, min_value=0.0)
    put_purchase_price = st.number_input("Put Option Purchase Price", value=5.0, min_value=0.0)

    # Calculate button
    calculate = st.button("💡 Calculate!")

    st.divider()
    st.header("Heatmap Parameters")
    default_min_spot = round(spot_price - 30, 2)
    default_max_spot = round(spot_price + 30, 2)
    min_spot = st.number_input("Min Spot Price", value=default_min_spot, min_value=0.01)
    max_spot = st.number_input("Max Spot Price", value=default_max_spot, min_value=0.01)

    min_vol = st.slider("Min Volatility for Heatmap", 0.01, 1.0, 0.1)
    max_vol = st.slider("Max Volatility for Heatmap", 0.01, 1.0, 0.6)

if calculate:
    with st.spinner("Calculating..."):
        import time
        time.sleep(1)  # Loading Simulation

        # Generate heatmap
        n_steps = 20
        spot_range = np.linspace(min_spot, max_spot, n_steps)
        vol_range = np.linspace(min_vol, max_vol, n_steps)

        spot_grid, vol_grid = np.meshgrid(spot_range, vol_range)
        call_pnl_grid = np.zeros_like(spot_grid)
        put_pnl_grid = np.zeros_like(spot_grid)

        days_to_maturity = int(maturity_days)
        T = days_to_maturity / 365

        # Calculate call and put PnL
        for i in range(n_steps):
            for j in range(n_steps):
                S = spot_grid[i, j]
                sigma = vol_grid[i, j]
                model = BlackScholesMertonModel(S, strike_price, days_to_maturity, risk_free_rate, sigma)
                call_price = model.calculate_option_price("Call Option")
                put_price = model.calculate_option_price("Put Option")
                call_pnl_grid[i, j] = call_price - call_purchase_price
                put_pnl_grid[i, j] = put_price - put_purchase_price

        # PnL color map: green (profit) to red (loss)
        cmap = plt.get_cmap("RdYlGn")  # red=loss, green=profit

        # Show metrics
        col1, col2 = st.columns(2)
        model = BlackScholesMertonModel(spot_price, strike_price, days_to_maturity, risk_free_rate, volatility)
        call_price = model.calculate_option_price("Call Option")
        put_price = model.calculate_option_price("Put Option")
        delta_call = call_price - call_purchase_price
        delta_put = put_price - put_purchase_price

        # Format the value change display with custom arrow and color
        def custom_metric(label, value, pnl):
            arrow = "🡅" if pnl > 0 else "🡇"
            color = "green" if pnl > 0 else "red"
            sign = "  + " if pnl > 0 else "  - "
            delta = f"{arrow} {sign}${abs(pnl):.2f}"

            # Custom HTML display with a small rounded rectangle
            html = f"""
            <div style='padding: 10px 0;'>
                <div style='font-size: 20px; color: white; opacity: 0.85;'>{label}</div>
                <div style='font-size: 32px; font-weight: bold; color: white;'>${value:.2f}</div>
                <div style='display: inline-block; background-color: {color}; color: white;
                            padding: 4px 10px; border-radius: 20px; font-size: 16px; opacity: 0.9;
                            margin-top: 4px;'>
                    {delta}
                </div>
            </div>
            """
            return html

        # Show in columns
        col1, col2 = st.columns(2)
        col1.markdown(custom_metric("Call Option Value", call_price, delta_call), unsafe_allow_html=True)
        col2.markdown(custom_metric("Put Option Value", put_price, delta_put), unsafe_allow_html=True)

        st.subheader("📉 Options Price - PnL Heatmaps")

        # Plot heatmaps
        fig, axs = plt.subplots(1, 2, figsize=(16, 6))
        sns.heatmap(call_pnl_grid, xticklabels=np.round(spot_range, 1), yticklabels=np.round(vol_range, 2),
                    ax=axs[0], cmap=cmap, center=0, cbar_kws={'label': 'Call PnL'})
        axs[0].set_title("Call Option PnL Heatmap")
        axs[0].set_xlabel("Spot Price")
        axs[0].set_ylabel("Volatility")

        sns.heatmap(put_pnl_grid, xticklabels=np.round(spot_range, 1), yticklabels=np.round(vol_range, 2),
                    ax=axs[1], cmap=cmap, center=0, cbar_kws={'label': 'Put PnL'})
        axs[1].set_title("Put Option PnL Heatmap")
        axs[1].set_xlabel("Spot Price")
        axs[1].set_ylabel("Volatility")

        st.pyplot(fig)

        # Set parameters from input
        mc = MonteCarloOptionPricer(
            spot_price=spot_price,
            strike_price=strike_price,
            days_to_maturity=maturity_days,
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            num_simulations=10000
        )

        # Run simulation
        mc.simulate_price_paths()

        # Calculate Monte Carlo prices
        call_price_mc = mc.price_option(option_type="call")
        put_price_mc = mc.price_option(option_type="put")

        # Show in Streamlit
        st.markdown(f"**Monte Carlo Estimated Call Price:** ${call_price_mc:.2f}")
        st.markdown(f"**Monte Carlo Estimated Put Price:** ${put_price_mc:.2f}")

        # Plot simulated price paths
        st.pyplot(mc.plot_simulations(num_paths_to_plot=100))


import yfinance as yf

# Title
st.title("📈 Asset Price Analysis and Summary Statistics")

# Ticker and date inputs
ticker = st.text_input("Enter stock ticker", value="NVDA")

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=pd.to_datetime("2024-01-01"))
with col2:
    end_date = st.date_input("End Date", value=pd.to_datetime("today"))

# Download price data from yfinance
df = yf.download(ticker, start=start_date, end=end_date)
df.reset_index(inplace=True)
df['Date'] = pd.to_datetime(df['Date'])  # set into datetime format
    # Set Date as index

if not df.empty:
    latest_price = float(df['Close'].iloc[-1])
    st.success(f"**Latest {ticker} Price:** ${latest_price:.2f}")

    spot_price = st.number_input("Current Asset Price", value=float(latest_price), min_value=0.01)

    st.subheader(f"{ticker} Price Action")
    st.line_chart(df['Close'])

    # Compute statistical
    df['LogRet'] = np.log(df['Close']/df['Close'].shift(1))
    daily_std = df['LogRet'].std()
    hist_vol = daily_std * np.sqrt(252)
    sharpe_ratio = (df['LogRet'].mean() * 252 - risk_free_rate) / (daily_std * np.sqrt(252))

    # Risk free rate from ^TNX (10-year treasury yield) from yf
    # Short term (1‑month T‑bill proxy) and long-term rate from yf
    irx = yf.Ticker("^IRX").history(period="1d")["Close"].iloc[-1] / 100
    tnx = yf.Ticker("^TNX").history(period="1d")["Close"].iloc[-1] / 100
    try:
        st.markdown(f"<div style='font-size:23px;'><strong>• 1‑Month Treasury Bill Rate (≈ 3 Months): </strong> {irx:.2%}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:23px;'><strong>• 10‑Year Treasury Yield: </strong> {tnx:.2%}</div>", unsafe_allow_html=True)
    except:
        irx, tnx = None
        irx, tnx = "N/A"

    # Picked Stock Statistic
    st.subheader("📊 Summary Statistics (Daily Close)")
    st.markdown(f"""
    - **Period:** `{start_date.strftime('%Y-%m-%d')}` to `{end_date.strftime('%Y-%m-%d')}`
    - **Historical Volatility (per selected range of time):**`{hist_vol:.2%}`
    - **Sharpe Ratio:**`{sharpe_ratio:.5}`
    - **Log Returns Mean:** `{df['LogRet'].mean():.4f}`
    - **Log Returns Std Dev:** `{daily_std:.4f}`
    """)

    # Log Returns Plot
    log_returns = np.log(df["Close"] / df["Close"].shift(1)).dropna()
    
    plt.figure(figsize=(8, 4))
    sns.histplot(log_returns, bins=50, kde=True, color="teal")
    plt.title(f"Distribution of Daily Log Returns of {ticker}")
    plt.xlabel("Log Return")
    plt.ylabel("Frequency")
    st.pyplot(plt)

    cumulative_returns = (1 + log_returns).cumprod()

    plt.figure(figsize=(10, 4))
    plt.plot(df["Date"].iloc[1:], cumulative_returns, color="green")
    plt.title(f"Cumulative Log Returns of {ticker}")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.grid(True)
    st.pyplot(plt)


    # 21-Day Rolling Volatility Plot
    rolling_vol = log_returns.rolling(window=21).std() * np.sqrt(252)  # 1-month rolling vol annualized

    plt.figure(figsize=(10, 4))
    plt.plot(df["Date"].iloc[1:], rolling_vol, color="orange")
    plt.title(f"21-Day Rolling Annualized Volatility of {ticker}")
    plt.xlabel("Date")
    plt.ylabel("Volatility (%)")
    plt.grid(True)
    st.pyplot(plt)



else:
    st.warning("No data found for the selected ticker and date range.")