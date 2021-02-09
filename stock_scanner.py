import numpy as np
import pandas as pd
import requests #for http requests
import xlsxwriter
import math
import datetime
import time
from short_seller_checker import short_seller_checker
from tradingview_data import tradingview_recommendation
from secrets import IEX_CLOUD_API_TOKEN
from scipy import stats


# Getting the following from the IEX API
# https://cloud.iexapis.com/ 
market_api_url = 'https://cloud.iexapis.com/stable/stock/market/list'
stock_api_url = 'https://cloud.iexapis.com/stable/stock'

def latest_iex_news(ticker):
    "Shows the latest news from the IEX Cloud"
    
    #print(f"Here are the news and stats for {ticker}: ")

    # Specific Ticker News
    ticker_news_url = f'{stock_api_url}/{ticker}/news/last/5?token={IEX_CLOUD_API_TOKEN}'
    ticker_news = requests.get(ticker_news_url).json()

    # Get the headlines
    print(f"Headlines for {ticker}")
    for i in range(len(ticker_news)):
        print(ticker_news[i]['headline'])
        # Summary 
        print("")

    # Converts to Dataframe for readability
    # Convert datetime to something humans can easily read
    for i in range(len(ticker_news)):
        
        epochtime = ticker_news[i]['datetime']
        # Converting to date and time since original value is in epoch milliseconds
        dt = datetime.datetime.fromtimestamp(epochtime / 1000.0, tz=datetime.timezone.utc)
        ticker_news[i]['datetime'] = dt
        print("")

    headlines_df = pd.DataFrame(ticker_news)
    headlines_df = headlines_df.drop(columns=['image','lang','hasPaywall'])
    print(headlines_df)
    print()

def ceo_compensation(ticker):
    "CEO Compensation from IEX Cloud"
    # Adding error check since some companies don't resport CEO compensation
    try:
        ceo_comp_url = f'{stock_api_url}/{ticker}/ceo-compensation?token={IEX_CLOUD_API_TOKEN}'
        ceo_comp = requests.get(ceo_comp_url).json()
        ceo_comp = pd.DataFrame(ceo_comp)
        print("CEO Compensation: \n", ceo_comp)
        print("")
        
    except Exception as error:
        print("No reported CEO compensation so ", error.__class__, "occurred.")
        print()
    
def fund_ownership(ticker):
    "Fund Ownership from IEX Cloud"
    fund_ownership_url = f'{stock_api_url}/{ticker}/fund-ownership?token={IEX_CLOUD_API_TOKEN}'
    fund_ownership = requests.get(fund_ownership_url).json()
    # Use Pandas DataFrame for easy reading
    fund_df = pd.DataFrame(fund_ownership)
    print("Fund Ownership Info: \n", fund_df)
    print("")

def day50MA(ticker):
    # 50 Day Moving Average
    day50_MA_url = f'{stock_api_url}/{ticker}/stats/day50movingavg?token={IEX_CLOUD_API_TOKEN}'
    day50_MA = requests.get(day50_MA_url).json()
    print("50 Day MA: ", day50_MA['day50MovingAvg'])
    print("")

def day200MA(ticker):
    # 200 Day Moving Average
    day200_MA_url = f'{stock_api_url}/{ticker}/stats/day200movingavg?token={IEX_CLOUD_API_TOKEN}'
    day200_MA = requests.get(day200_MA_url).json()
    print("200 Day MA: ", day200_MA['day200MovingAvg'])
    print("")

def latest_price(ticker):
    # Latest Price
    latest_price_url = f'{stock_api_url}/{ticker}/quote/latestPrice?token={IEX_CLOUD_API_TOKEN}'
    latest_price = requests.get(latest_price_url).json()
    print("Latest Price: ", latest_price)
    print("")

    # Market Cap
    # market_cap_url = f'{stock_api_url}/{ticker}/marketcap?token={IEX_CLOUD_API_TOKEN}'
    # market_cap = requests.get(market_cap_url).json()
    # print("Market Cap: ", market_cap)
    # print("")
    
def short_seller_information(ticker):
    "Short Seller Information"
    # Short Seller Checker
    short_seller_checker(ticker)

def recommendation_list(ticker):
    "Assortment of recommendations"
    
    # TradingView Recommendations
    tradingview_recommendation(ticker)

def largest_trade_information(ticker):
    "Largest Trade Information"

    largest_trade_url = f'{stock_api_url}/{ticker}/largest-trades?token={IEX_CLOUD_API_TOKEN}'
    largest_trade = requests.get(largest_trade_url).json()

    # Error check
    try: 
        epochtime = largest_trade[:]['time']
        print(epochtime)
        print(type(epochtime))
        # Converting to date and time since original value is in epoch milliseconds
        dt = datetime.datetime.fromtimestamp(epochtime / 1000.0, tz=datetime.timezone.utc)
        largest_trade[:]['time'] = dt
        print("Largest Trade Stats: ")
        print(largest_trade)
        largest_trade = pd.DataFrame(largest_trade)
    except:    
        print("Largest Trading Data Unavilable: ", largest_trade)
        print("")