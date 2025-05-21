
# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import norm
from datetime import datetime, timedelta
from py_vollib.black_scholes import black_scholes as bs
from py_vollib.black_scholes.greeks.analytical import delta, gamma, vega, theta, rho
from fredapi import Fred
from prophet import Prophet
import plotly.express as px
import arch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import altair as alt



#site Title
st.header('PortfoliOluso')
# Function to determine trading days using the NYSE calendar
def get_trading_days(start_date, end_date):
    nyse = mcal.get_calendar('XNYS')
    return nyse.valid_days(start_date=start_date, end_date=end_date)
# Function to load stock data
def load_stock_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end)
    return data
    # Filter out non-trading days
    trading_days = get_trading_days(start_date=start_date, end_date=train_end_date)
    stock_data = stock_data[stock_data['Date'].isin(trading_days)]
# Function to download stock data
def download_stock_data(stock_symbol, start_date, end_date):
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
    stock_data.reset_index(inplace=True)
    return stock_data
# Sidebar inputs
st.sidebar.header("Select Analysis")
analysis_option = st.sidebar.radio("Select Analysis", ["Portfolio Risk", "Hedging"])
# Define initial variables
selected_tickers = []
custom_tickers = ""
weights = []
moving_average_period = None


# Function to calculate individual holding volatility and returns
def calculate_individual_holdings_stats(returns, weights):
    individual_volatility = np.std(returns, axis=0)
    individual_returns = np.mean(returns, axis=0)
    weighted_volatility = np.dot(individual_volatility, weights)
    weighted_returns = np.dot(individual_returns, weights)
    return individual_volatility, individual_returns, weighted_volatility, weighted_returns
    
if analysis_option == "Portfolio Risk":
    st.sidebar.header("Value at Risk (VaR) Analysis")
    # List of stock tickers in the portfolio
    predefined_tickers = [
        "MMM", "AOS", "ABT", "ABBV", "ABMD", "ACN", "ATVI", "ADM", "ADBE", "ADP", "AAP", 
        "AES", "AFL", "A", "APD", "AKAM", "ALK", "ALB", "ARE", "ALGN", "ALLE", "LNT", 
        "ALL", "GOOGL", "GOOG", "MO", "AMZN", "AMCR", "AMD", "AEE", "AAL", "AEP", "AXP", 
        "AIG", "AMT", "AWK", "AMP", "ABC", "AME", "AMGN", "APH", "ADI", "ANSS", "AON", "APA", 
        "AAPL", "AMAT", "APTV", "ANET", "AJG", "AIZ", "T", "ATO", "ADSK", "AZO", "AVB", "AVY", "BKR", 
        "BALL", "BAC", "BBWI", "BAX", "BDX", "WRB", "BRK.B", "BBY", "BIO", "TECH", "BIIB", "BLK", "BK", 
        "BA", "BKNG", "BWA", "BXP", "BSX", "BMY", "AVGO", "BR", "BRO", "BF.B", "CHRW", "CDNS", 
        "CZR", "CPT", "CPB", "COF", "CAH", "KMX", "CCL", "CARR", "CTLT", "CAT", "CBOE", 
        "CBRE", "CDW", "CE", "CNC", "CNP", "CDAY", "CF", "CRL", "SCHW", "CHTR", "CVX", 
        "CMG", "CB", "CHD", "CI", "CINF", "CTAS", "CSCO", "C", "CFG", "CLX", 
        "CME", "CMS", "KO", "CTSH", "CL", "CMCSA", "CMA", "CAG", "COP", "ED", 
        "STZ", "CEG", "COO", "CPRT", "GLW", "CTVA", "CSGP", "COST", "CTRA", "CCI", 
        "CSX", "CMI", "CVS", "DHI", "DHR", "DRI", "DVA", "DE", "DAL", "XRAY", 
        "DVN", "DXCM", "FANG", "DLR", "DFS", "DISH", "DIS", "DG", "DLTR", "D", "DPZ",
        "DOV", "DOW", "DTE", "DUK", "DD", "DXC", "EMN", "ETN", "EBAY", "ECL", "EIX", 
        "EW", "EA", "ELV", "LLY", "EMR", "ENPH", "ETR", "EOG", "EPAM", "EQT", "EFX", 
        "EQIX", "EQR", "ESS", "EL", "ETSY", "RE", "EVRG", "ES", "EXC", "EXPE", "EXPD", 
        "EXR", "XOM", "FFIV", "FDS", "FAST", "FRT", "FDX", "FITB", "FRC", "FE", "FIS", 
        "FISV", "FLT", "FMC", "F", "FTNT", "FTV", "FBHS", "FOXA", "FOX", 
        "BEN", "FCX", "GRMN", "IT", "GNRC", "GD", "GE", "GIS", "GM", "GPC", "GILD",
        "GL", "GPN", "GS", "HAL", "HIG", "HAS", "HCA", "PEAK", "HSIC", "HSY", "HES", 
        "HPE", "HLT", "HOLX", "HD", "HON", "HRL", "HST", "HWM", "HPQ", "HUM", "HBAN",
        "HII", "IBM", "IEX", "IDXX", "ITW", "ILMN", "INCY", "IR", "INTC", "ICE", "IP", 
        "IPG", "IFF", "INTU", "ISRG", "IVZ", "INVH", "IQV", "IRM", "JBHT", "JKHY", "J", 
        "JNJ", "JCI", "JPM", "JNPR", "K", "KDP", "KEY", "KEYS", "KMB", "KIM", "KMI", "KLAC", 
        "KHC", "KR", "LHX", "LH", "LRCX", "LW", "LVS", "LDOS", "LEN", "LNC", "LIN", "LYV", "LKQ", 
        "LMT", "L", "LOW", "LUMN", "LYB", "MTB", "MRO", "MPC", "MKTX", "MAR", "MMC", "MLM", "MAS", 
        "MA", "MTCH", "MKC", "MCD", "MCK", "MDT", "MRK", "META", "MET", "MTD", "MGM", "MCHP", "MU",
        "MSFT", "MAA", "MRNA", "MHK", "MOH", "TAP", "MDLZ", "MPWR", "MNST", "MCO", "MS", "MOS", "MSI",
        "MSCI", "NDAQ", "NTAP", "NFLX", "NWL", "NEM", "NWSA", "NWS", "NEE", "NKE", "NI", "NDSN", "NSC",
        "NTRS", "NOC", "NLOK", "NCLH", "NRG", "NUE", "NVDA", "NVR", "NXPI", "ORLY", "OXY", "ODFL", "OMC", 
        "ON", "OKE", "ORCL", "OGN", "OTIS", "PCAR", "PKG", "PARA", "PH", "PAYX", "PAYC", "PYPL", "PNR", "PEP", 
        "PKI", "PFE", "PCG", "PM", "PSX", "PNW", "PXD", "PNC", "POOL", "PPG", "PPL", "PFG", "PG", "PGR", 
        "PLD", "PRU", "PEG", "PTC", "PSA", "PHM", "QRVO", "PWR", "QCOM", "DGX", "RL", 
        "RJF", "RTX", "O", "REG", "REGN", "RF", "RSG", "RMD", "RHI", "ROK", "ROL", 
        "ROP", "ROST", "RCL", "SPGI", "CRM", "SBAC", "SLB", "STX", "SEE", "SRE", 
        "NOW", "SHW", "SBNY", "SPG", "SWKS", "SJM", "SNA", "SEDG", "SO", "LUV", 
        "SWK", "SBUX", "STT", "STE", "SYK", "SIVB", "SYF", "SNPS", "SYY", "TMUS",
        "TROW", "TTWO", "TPR", "TRGP", "TGT", "TEL", "TDY", "TFX", "TER", "TSLA", 
        "TXN", "TXT", "TMO", "TJX", "TSCO", "TT", "TDG", "TRV", "TRMB", "TFC", "TWTR", 
        "TYL", "TSN", "USB", "UDR", "ULTA", "UNP", "UAL", "UPS", "URI", "UNH", "UHS", "VLO",
        "VTR", "VRSN", "VRSK", "VZ", "VRTX", "VFC", "VTRS", "VICI", "V", 
        "VNO", "VMC", "WAB", "WBA", "WMT", "WBD", "WM", "WAT", "WEC", "WFC", "WELL",
        "WST", "WDC", "WRK", "WY", "WHR", "WMB", "WTW", "GWW", "WYNN", "XEL", "XYL", 
        "YUM", "ZBRA", "ZBH", "ZION", "ZTS"
    ]
    # Multiselect for selecting multiple stock tickers
    selected_tickers = st.sidebar.multiselect("Select Stock Tickers", predefined_tickers)
    # Input custom tickers
    custom_tickers = st.sidebar.text_area("Enter Custom Tickers (comma-separated)", "").strip()
    # Combine predefined and custom tickers
    all_tickers = selected_tickers + [ticker.strip().upper() for ticker in custom_tickers.split(",") if ticker.strip()]
    if not all_tickers:
        st.warning("Please select or enter at least one stock ticker.")
    else:
        # Radio button for selecting equal or custom weights
        weight_option = st.radio("Select Weight Option", ["Equal Weights", "Custom Weights"])
        # Initialize an empty list to store weights
        weights = []
        if weight_option == "Equal Weights":
            weights = np.array([1 / len(all_tickers)] * len(all_tickers))
        else:
            for ticker in all_tickers:
                weight = st.text_input(f"Weight for {ticker}", value = ".05", key=ticker)
                weights.append(float(weight))  # Convert the input to float
            # Normalize custom weights to ensure they sum up to 1
            weight_sum = sum(weights)
            weights = [weight / weight_sum for weight in weights]
        # Input date range for VaR calculation
        var_start_date = st.sidebar.text_input("Start Date (YYYY-MM-DD)", '1980-01-01')
        var_end_date = st.sidebar.text_input("End Date (YYYY-MM-DD)", '2023-01-01')
        # Portfolio value
        portfolio_value = st.sidebar.number_input("Portfolio Starting Value", min_value=1.0, value=100000.0)
        # Define the scaling function
        def scale(x):
            return x / np.sum(np.abs(x))
        # Download historical stock data for the specified date range for selected tickers and S&P 500
        all_tickers.append('^GSPC')  # Adding S&P 500 to the list of tickers
        data = yf.download(all_tickers, start=var_start_date, end=var_end_date)
        # Calculate daily returns for the selected tickers and S&P 500
        returns = data['Adj Close'].pct_change()
        returns = returns.dropna()
         # Center returns around zero
        returns_centered = returns - returns.mean()    
    
        # Scale the initial weights
        portfolio_weights = scale(weights)
        # Calculate portfolio returns by multiplying the returns of each stock by their respective weights
        portfolio_returns = np.dot(returns.iloc[:, :-1], portfolio_weights)  # Exclude the last column (S&P 500)
        
        # Calculate the S&P 500 returns
        sp500_returns = returns['^GSPC']
        # Add S&P 500 returns to the portfolio returns
        portfolio_returns_with_sp500 = portfolio_returns + sp500_returns
        # Calculate the dollar value of the portfolio based on the input portfolio value
        portfolio_dollar_value = portfolio_value * (1 + portfolio_returns_with_sp500).cumprod()
        # Display portfolio returns with S&P 500
        st.subheader("Portfolio Returns vs S&P 500")
        portfolio_chart_data = pd.DataFrame({
            'Portfolio': portfolio_dollar_value,
            'S&P 500': (portfolio_value * (1 + sp500_returns).cumprod())
        })
        st.line_chart(portfolio_chart_data, use_container_width=True)
        # Calculate cumulative returns for individual holdings
        cumulative_returns = (1 + returns.iloc[:, :-1]).cumprod() - 1
        # Calculate volatility for individual holdings
        volatility = returns_centered.iloc[:, :-1].std()
        # Display cumulative returns in a table
        st.subheader("Returns vs Volatility")
        cumulative_returns_table_data = pd.DataFrame({
            'Ticker': all_tickers[:-1],
            'Cumulative Return': cumulative_returns.iloc[-1] * 100,  # Display as percentage
            'Volatility': volatility * 100  # Display as percentage
        })
        # Ensure all values are positive
        #cumulative_returns_table_data['Cumulative Return'] = cumulative_returns_table_data['Cumulative Return'].abs()
        # Set the 'Ticker' column as the index
        cumulative_returns_table_data['Ticker'] = all_tickers[:-1]
        cumulative_returns_table_data.set_index('Ticker', inplace=True)
        # Display cumulative returns table
        cumulative_returns_table_data = cumulative_returns_table_data.rename_axis('Ticker')
        st.table(cumulative_returns_table_data)


        # Download Returns as CSV
        cumulative_returns_table_data_csv = cumulative_returns_table_data.to_csv(index=True)
        st.download_button("Download Returns as CSV", data=cumulative_returns_table_data_csv, file_name="cumulative_returns_table_data.csv")  
        
        # Calculate VaR for different confidence levels
        confidence_levels = [0.9, 0.95, 0.99]
        # Calculate Historical VaR
        historical_var_percentiles = [np.percentile(portfolio_returns_with_sp500, 100 - level*100) for level in confidence_levels]
        historical_var_dollars = [portfolio_value * percentile for percentile in historical_var_percentiles]
        # Calculate Variance-Covariance VaR
        m = portfolio_returns_with_sp500.mean()
        std_dev = portfolio_returns_with_sp500.std()
        var_covar_percentiles = [norm.ppf(1 - level, m, std_dev) for level in confidence_levels]
        var_covar_dollars = [portfolio_value * percentile for percentile in var_covar_percentiles]
        # Perform Monte Carlo Simulation for VaR
        num_simulations = 10_000
        simulation_results = []
        for _ in range(num_simulations):
            daily_returns = np.random.normal(m, std_dev, len(portfolio_returns_with_sp500))
            simulation_results.append(daily_returns[-1])  # Keep only the last day's return
        simulation_results = np.array(simulation_results)
        monte_carlo_var_percentiles = [np.percentile(simulation_results, 100 - level*100) for level in confidence_levels]
        monte_carlo_var_dollars = [portfolio_value * percentile for percentile in monte_carlo_var_percentiles]
        # Display VaR results
        st.subheader("VaR Analysis")
        for i, level in enumerate(confidence_levels):
            st.write(f'Historical VaR at {int(level*100)}%: ${historical_var_dollars[i]:,.2f}')
            st.write(f'Variance-Covariance VaR at {int(level*100)}%: ${var_covar_dollars[i]:,.2f}')
            st.write(f'Monte Carlo VaR at {int(level*100)}%: ${monte_carlo_var_dollars[i]:,.2f}')
elif analysis_option == "Hedging":
    st.sidebar.header("Options Pricing")
    # User input for stock symbol, strike price, days to expiration, and risk-free rate
    stock_name = st.text_input("Enter Ticker", 'AAPL')
    strike_price = st.number_input("Enter Strike Price", value=0, key="strike_price")
    days_to_expiration = st.number_input("Enter Days to Expiration", 1, 365, 30)  # Adjust the range as needed
    risk_free_rate = st.number_input("Enter Risk-Free Rate", 0.01, 0.10, 0.0433, 0.0001, key="risk_free_rate")
    # Define start_date and end_date as user inputs
    end_date = st.date_input("Select End Date", datetime.today())
    n_years = st.number_input("Enter Number of Years", 1)
    start_date = end_date - timedelta(days=n_years * 365)
    # Download stock data
    stock_data = yf.download(tickers=stock_name, start=start_date, end=end_date)
    stock_prices = stock_data['Adj Close']
    # Calculate log returns and volatility
    log_returns = np.log(stock_prices / stock_prices.shift(10)).dropna()
    trading_days_year = 252
    trading_days_month = 21
    volatility = log_returns.rolling(window=trading_days_month).std() * np.sqrt(trading_days_year)
    # Create a DataFrame with both sets of data
    data = pd.DataFrame({'Stock Prices': stock_prices, 'Volatility': volatility})
    fig, ax = plt.subplots()
    ax.plot(stock_prices, color = 'red')
    ax.set_xlabel("Date", fontsize = 14)
    ax.set_ylabel("Stock price", color = 'red', fontsize = 14)
    ax2 = ax.twinx()
    ax2.plot(volatility, color = 'blue')
    ax2.set_ylabel("Volatility", color = 'blue', fontsize = 14)
    st.pyplot(plt)
    # BS Call option pricing
    st.subheader("Black-Scholes Call Option Pricing")
    S0_call = stock_prices.iloc[-1]
    T_call = days_to_expiration / 365
    vol_call = volatility.iloc[-1]
    call_option_price = bs("c", S0_call, strike_price, T_call, risk_free_rate, vol_call)
    st.write("Call Option price:", call_option_price)
    # Greeks for Call option
    st.subheader("Call Option Greeks")
    delta_option_call = delta("c", S0_call, strike_price, T_call, risk_free_rate, vol_call)
    gamma_option_call = gamma("c", S0_call, strike_price, T_call, risk_free_rate, vol_call)
    vega_option_call = vega("c", S0_call, strike_price, T_call, risk_free_rate, vol_call)
    rho_option_call = rho("c", S0_call, strike_price, T_call, risk_free_rate, vol_call)
    theta_option_call = theta("c", S0_call, strike_price, T_call, risk_free_rate, vol_call)
    st.write("Delta (Call):", round(delta_option_call, 3))
    st.write("Gamma (Call):", round(gamma_option_call, 3))
    st.write("Vega (Call):", round(vega_option_call, 3))
    st.write("Rho (Call):", round(rho_option_call, 3))
    st.write("Theta (Call):", round(theta_option_call, 3))
    # BS Put option pricing
    st.subheader("Black-Scholes Put Option Pricing")
    S0_put = stock_prices.iloc[-1]
    vol_put = volatility.iloc[-1]
    put_option_price = bs("p", S0_put, strike_price, T_call, risk_free_rate, vol_put)
    st.write("Put Option price:", put_option_price)
    # Greeks for Put option
    st.subheader("Put Option Greeks")
    delta_option_put = delta("p", S0_put, strike_price, T_call, risk_free_rate, vol_put)
    gamma_option_put = gamma("p", S0_put, strike_price, T_call, risk_free_rate, vol_put)
    vega_option_put = vega("p", S0_put, strike_price, T_call, risk_free_rate, vol_put)
    rho_option_put = rho("p", S0_put, strike_price, T_call, risk_free_rate, vol_put)
    theta_option_put = theta("p", S0_put, strike_price, T_call, risk_free_rate, vol_put)
    st.write("Delta (Put):", round(delta_option_put, 3))
    st.write("Gamma (Put):", round(gamma_option_put, 3))
    st.write("Vega (Put):", round(vega_option_put, 3))
    st.write("Rho (Put):", round(rho_option_put, 3))
    st.write("Theta (Put):", round(theta_option_put, 3))
    
st.sidebar.write("Data provided by Yahoo Finance.")
