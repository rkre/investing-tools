# This script analyzes various volatilities 
# Closing prices for daily, weekly, monthly
# Volatility based on max/min 
# Volatility analysis for around option expiration days (Friday activity)
# Volatility based on volume
# Volatility based on earnings report


# This data will be used to calculate the right stocks for volatility option trading strategies like Iron Condor


from secrets import IEX_CLOUD_API_TOKEN
import numpy as np
import pandas as pd
import requests #for http requests


# Function to find closing price 
# IEX or RH or yfinance or other API can be used

# Getting the following from the IEX API
# https://cloud.iexapis.com/ 

market_api_url = 'https://cloud.iexapis.com/stable/stock/market/list'
stock_api_url = 'https://cloud.iexapis.com/stable/stock'

def get_1yr_data(ticker):
    url_1yr = f'{stock_api_url}/{ticker}/chart/6m?token={IEX_CLOUD_API_TOKEN}'
    year_data = requests.get(url_1yr).json()
    year_df = pd.DataFrame(year_data)
    pd.set_option("display.max_rows", None, "display.max_columns", None)

    return year_df


def get_daily_highlows(ticker):
    "Gets the max and min price for the day"
    year_data = get_1yr_data(ticker)
    highlows = year_data[['date','high','low']]
    return highlows

def daily_volatility(ticker):
    "Returns the % of volatility based on the high and low price of the day"
    highlows = get_daily_highlows(ticker)
    points_change = highlows['high'].sub(highlows['low'])
    highlows.insert(3,"points",points_change)
    percent_change = points_change.div(highlows['low'])*100
    highlows.insert(4,"%",percent_change)
    return highlows

def highest_volatility(ticker):
    "Returns the date annd prices for the highest volatility of a stock"
    highlows = daily_volatility(ticker)
    highest_vol = highlows.nlargest(5,"%")
    return highest_vol

def highest_price(ticker):
    "Returns highest price in a given time period"
    prices = get_daily_highlows(ticker)
    highest_price = prices.nlargest(2,"high")
    return highest_price

def lowest_price(ticker):
    "Returns lowest price in a given time period"
    prices = get_daily_highlows(ticker)
    lowest_price = prices.nsmallest(2,"high")
    return lowest_price

# ticker = 'MSFT'
# highlow = highest_volatility(ticker)

# print(highest_price(ticker))
# print(lowest_price(ticker))
# print(highlow)




