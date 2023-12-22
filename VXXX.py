

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
st.header('_P1_')
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
# Function to load economic data
def load_economic_data(indicator, start_date, end_date):
    fred = Fred(api_key='a3c314b9096130db0731f91c2d8001a5')  # Replace 'your_api_key' with your actual API key
    economic_data = fred.get_series(indicator, start_date=start_date, end_date=end_date)
    return economic_data
# Function to preprocess and align the data
def preprocess_data(stock_data, economic_data):
    # Align economic data with stock data
    aligned_data = pd.concat([stock_data, economic_data], axis=1)
    # Remove missing values
    aligned_data = aligned_data.dropna()
    return aligned_data
# Function to download stock data
def download_stock_data(stock_symbol, start_date, end_date):
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
    stock_data.reset_index(inplace=True)
    return stock_data
# Sidebar inputs
st.sidebar.header("Select Analysis")
analysis_option = st.sidebar.radio("Select Analysis", ["Economic Data", "Portfolio Risk", "Hedging", "Equity Price Forecast"])
# Define initial variables
selected_tickers = []
custom_tickers = ""
weights = []
moving_average_period = None
# Function to load stock data and create a forecast using fbProphet
def load_stock_data_and_forecast_fbprophet(stock_symbol, start_date, end_date, train_end_date, test_start_date, prediction_range):
    # Retrieve stock price data from Yahoo Finance
    stock_data = yf.download(stock_symbol, start=start_date, end=train_end_date)
    stock_data.reset_index(inplace=True)
    # Create a line plot for the stock's 'Date' and 'Adj Close' price
    st.plotly_chart(px.line(stock_data, x='Date', y='Adj Close', title=f'{stock_symbol} Stock Price'))
    # Rename columns for Prophet compatibility
    stock_data = stock_data.rename(columns={'Date': 'ds', 'Adj Close': 'y'})
    # Initialize and fit Prophet model
    m1 = Prophet()
    m1.fit(stock_data)
    # Create a DataFrame for future predictions
    future = m1.make_future_dataframe(periods=prediction_range)
    # Predict future values
    forecast = m1.predict(future)
    # Display the forecasted values
    st.write("Forecasted Stock Prices (fbProphet):")
    st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
def load_stock_data_and_forecast_garch(stock_symbol, start_date, end_date, train_end_date, test_start_date, prediction_range):
    # Retrieve stock price data from Yahoo Finance
    stock_data = yf.download(stock_symbol, start=start_date, end=train_end_date)
    stock_data.reset_index(inplace=True)
    # Create a line plot for the stock's 'Date' and 'Adj Close' price
    st.plotly_chart(px.line(stock_data, x='Date', y='Adj Close', title=f'{stock_symbol} Stock Price'))
    # Calculate log returns
    stock_data['log_returns'] = np.log(stock_data['Adj Close'] / stock_data['Adj Close'].shift(1))
    # Fit a GARCH(1, 1) model
    model = arch.arch_model(stock_data['log_returns'][1:], vol='Garch', p=1, q=1)
    results = model.fit()
    # Create a DataFrame for future predictions
    forecast_horizon = prediction_range
    forecasts = results.forecast(start=stock_data.index[-1], horizon=forecast_horizon)
    # Display the forecasted volatility
    st.write("Forecasted Stock Price Volatility (GARCH Model):")
    for i in range(forecast_horizon):
        st.write(f"Day {i + 1}: {forecasts.variance.values[-1, i]:.6f}")
def load_stock_data_and_forecast_xgboost(stock_symbol, start_date, end_date, train_end_date, test_start_date):
    # Download historical stock data
    train_data = yf.download(stock_symbol, start=start_date, end=train_end_date)
    test_data = yf.download(stock_symbol, start=test_start_date, end=end_date)
    # Data Visualization
    def visualize_data(data, title, target_column):
        plt.figure(figsize=(15, 5))
        plt.plot(data.index, data[target_column], '.', color=sns.color_palette()[0])
        plt.title(title)
        plt.show()
    visualize_data(train_data, f'{stock_symbol} Training Data', 'Adj Close')
    # Model Building
    def train_xgboost_model(train, features, target):
        X_train = train[features]
        y_train = train[target]
        reg = xgb.XGBRegressor(
            n_estimators=1000,
            early_stopping_rounds=50,
            objective='reg:linear',
            max_depth=3,
            learning_rate=0.01
        )
        reg.fit(X_train, y_train,
                eval_set=[(X_train, y_train)],
                verbose=100)
        return reg
    FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume']
    TARGET = 'Adj Close'
    model = train_xgboost_model(train_data, FEATURES, TARGET)
    # Feature Importance
    def plot_feature_importance(model):
        fi = pd.DataFrame(data=model.feature_importances_, index=FEATURES, columns=['importance'])
        fi.sort_values('importance').plot(kind='barh', title='Feature Importance')
    plot_feature_importance(model)
    # Forecasting for test data
    def plot_predictions_vs_truth(data, title):
        ax = data['Adj Close'].plot(figsize=(15, 5))
        data['prediction'].plot(ax=ax, style='.')
        plt.legend(['Truth Data', 'Predictions'])
        ax.set_title(title)
    test_data['prediction'] = model.predict(test_data[FEATURES])
    plot_predictions_vs_truth(test_data, f'{stock_symbol} Stock Price Prediction (Test Data)')
    # Model Evaluation
    def evaluate_model(test, target_col, prediction_col):
        rmse = np.sqrt(mean_squared_error(test[target_col], test[prediction_col]))
        print(f'RMSE Score on Test set: {rmse:0.2f}')
    evaluate_model(test_data, TARGET, 'prediction')
# Function to calculate individual holding volatility and returns
def calculate_individual_holdings_stats(returns, weights):
    individual_volatility = np.std(returns, axis=0)
    individual_returns = np.mean(returns, axis=0)
    weighted_volatility = np.dot(individual_volatility, weights)
    weighted_returns = np.dot(individual_returns, weights)
    return individual_volatility, individual_returns, weighted_volatility, weighted_returns
# Add options for different functionalities
if analysis_option == "Equity Price Forecast":
    st.sidebar.header("Equity Price Forecast")
    option = st.sidebar.selectbox("Select an Option", ["Stock Price Forecast using fbProphet", "Garch Model", "XGBoost Model"])
    stock_symbol = st.sidebar.text_input("Enter Stock Symbol")
    start_date = st.sidebar.date_input("Start Date")
    train_end_date = st.sidebar.date_input("Train End Date")
    test_start_date = st.sidebar.date_input("Test Start Date")
    end_date = st.sidebar.date_input("End Date")
    prediction_range = st.sidebar.number_input("Prediction Range (days)", 1, 365, 30)
    if st.sidebar.button("Generate Forecast"):
        if option == "Stock Price Forecast using fbProphet":
            load_stock_data_and_forecast_fbprophet(stock_symbol, start_date, end_date, train_end_date, test_start_date, prediction_range)
        elif option == "Garch Model":
            load_stock_data_and_forecast_garch(stock_symbol, start_date, end_date, train_end_date, test_start_date, prediction_range)
        elif option == "XGBoost Model":
            load_stock_data_and_forecast_xgboost(stock_symbol, start_date, end_date, train_end_date, test_start_date)
elif analysis_option == "Portfolio Risk":
    st.sidebar.header("Value at Risk (VaR) Analysis")
    # List of stock tickers in the portfolio
    predefined_tickers = [
        'MMM', 'T', 'ABBV', 'ABT', 'ACN', 'ALL', 'GOOGL', 'GOOG', 'MO', 'AMZN',
        'BRK-B', 'BIIB', 'BLK', 'BA', 'BMY', 'CVS', 'COF', 'CAT', 'CVX', 'CSCO',
        'COP', 'DHR', 'DUK', 'DD', 'EMC', 'EMR', 'EXC', 'XOM', 'META', 'FDX',
        'GS', 'HAL', 'HD', 'HON', 'INTC', 'IBM', 'JPM', 'JNJ', 'KMI', 'LLY',
        'MET', 'MSFT', 'MS', 'NKE', 'NEE', 'OXY', 'ORCL', 'PYPL', 'PEP', 'PFE',
        'SLB', 'SPG', 'SO', 'SBUX', 'TGT', 'TXN', 'BK', 'USB', 'UNP', 'UPS',
        'WBA', 'DIS', 'WFC'
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
                weight = st.text_input(f"Weight for {ticker}", value="0.00199601", key=ticker)
                weights.append(float(weight))  # Convert the input to float
            # Normalize custom weights to ensure they sum up to 1
            weight_sum = sum(weights)
            weights = [weight / weight_sum for weight in weights]
        # Input date range for VaR calculation
        var_start_date = st.sidebar.text_input("Start Date (YYYY-MM-DD)", '2010-01-01')
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
        
        # Ensure all values are positive (optional based on your requirement)
        # cumulative_returns_table_data['Cumulative Return'] = cumulative_returns_table_data['Cumulative Return'].abs()
        
        # Set the 'Ticker' column as the index
        cumulative_returns_table_data.set_index('Ticker', inplace=True)
        
        # User option to reorder the table
        order_by = st.selectbox("Order by", ["Cumulative Return", "Volatility"])
        
        # Define the arrow symbol and the corresponding sorting options for Cumulative Return
        arrow_return = "↑" if st.button("Ascending (Return)") else "↓" if st.button("Descending (Return)") else ""
        ascending_return = arrow_return == "↑"
        
        # Define the arrow symbol and the corresponding sorting options for Volatility
        arrow_volatility = "↑" if st.button("Ascending (Volatility)") else "↓" if st.button("Descending (Volatility)") else ""
        ascending_volatility = arrow_volatility == "↑"
        
        # Sort the table based on user selection
        if order_by == "Cumulative Return":
            cumulative_returns_table_data = cumulative_returns_table_data.sort_values(by=order_by, ascending=ascending_return)
        elif order_by == "Volatility":
            cumulative_returns_table_data = cumulative_returns_table_data.sort_values(by=order_by, ascending=ascending_volatility)
        
        # Display cumulative returns table with clickable arrows
        st.table(cumulative_returns_table_data.style.format({"Cumulative Return": lambda x: f"{x:.2f}%{arrow_return}",
                                                             "Volatility": lambda x: f"{x:.2f}%{arrow_volatility}"},
                                                            na_rep="-"))
        
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
    risk_free_rate = st.number_input("Enter Risk-Free Rate", 0.01, 0.10, 0.0433, 0.001, key="risk_free_rate")
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
elif analysis_option == "Economic Data":
    st.sidebar.header("Economic Data Analysis")

    try:
        # Initialize the Fred API with your API key
        fred = Fred(api_key='a3c314b9096130db0731f91c2d8001a5')

        # Input date range for economic data
        economic_start_date = st.sidebar.text_input("Start Date (YYYY-MM-DD)", '2010-01-01')
        economic_end_date = st.sidebar.text_input("End Date (YYYY-MM-DD)", '2023-01-01')
        
        # List of available economic indicators
        economic_indicators = {
            'Consumer Price Index (CPI)': 'CPALTT01USM657N',
            'Unemployment Rate': 'UNRATE',
            '10-Year Treasury Yield': 'DGS10',
            'Housing Prices': 'CSUSHPINSA',
            'Trade Balance': 'BOPGSTB',
            'Money Supply': 'M2SL',
            'Consumer Sentiment Index': 'UMCSENT',
        }

        # Multiselect for selecting economic indicators
        selected_indicators = st.sidebar.multiselect("Select Economic Indicators", list(economic_indicators.keys()))

        if not selected_indicators:
            st.warning("Please select at least one economic indicator.")
        else:
            # Create a DataFrame to store the selected economic indicators' data
            economic_data = pd.DataFrame()

            # Retrieve data for selected economic indicators
            for indicator_name in selected_indicators:
                indicator_code = economic_indicators[indicator_name]
                indicator_data = fred.get_series(indicator_code, start=economic_start_date, end=economic_end_date)
                economic_data[indicator_name] = indicator_data

            # Display data only for the selected date range
            economic_data = economic_data.loc[economic_start_date:economic_end_date]

            # Create graphs for selected economic indicators
            for indicator_name in selected_indicators:
                st.subheader(f"{indicator_name} Data")
                st.line_chart(economic_data[indicator_name])

    except Exception as e:
        st.exception(f"ERROR! VERIFY PARAMETERS AND RERUN. Details: {str(e)}")

st.sidebar.write("Data provided by Yahoo Finance. This product uses the FRED® API but is not endorsed or certified by the Federal Reserve Bank of St. Louis.")
