# This script analyzes various volatilities 
# Closing prices for daily, weekly, monthly
# Volatility based on max/min 
# Volatility analysis for around option expiration days (Friday activity)
# Volatility based on volume
# Volatility based on earnings report


# This data will be used to calculate the right stocks for volatility option trading strategies like Iron Condor

import stock_scanner as ss
import price_target_scanner as pts
import options_scanner as ops


# Function to find closing price 
# IEX or RH or yfinance or other API can be used

