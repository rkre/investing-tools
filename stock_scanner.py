# Functions included:
# Latest price (IEX)
# Latest IEX news

import numpy as np
import pandas as pd
import requests #for http requests
import xlsxwriter
import math
import datetime
import time
from short_seller_checker import short_seller_checker

from secrets import IEX_CLOUD_API_TOKEN
from scipy import stats


# Getting the following from the IEX API
# https://cloud.iexapis.com/ 
market_api_url = 'https://cloud.iexapis.com/stable/stock/market/list'
stock_api_url = 'https://cloud.iexapis.com/stable/stock'


def get_stats(ticker):
    stats_url = f'{stock_api_url}/{ticker}/advanced-stats?token={IEX_CLOUD_API_TOKEN}'
    stats = requests.get(stats_url).json()
    stats_df = pd.DataFrame(stats, index=[0])
    pd.set_option("display.max_rows", None, "display.max_columns", None)

    return stats_df

def latest_iex_news(ticker):
    "Shows the latest news from the IEX Cloud"
    
    #print(f"Here are the news and stats for {ticker}: ")

    # Specific Ticker News
    ticker_news_url = f'{stock_api_url}/{ticker}/news/last/5?token={IEX_CLOUD_API_TOKEN}'
    ticker_news = requests.get(ticker_news_url).json()

    # Get the headlines [OLD WAY]
    print(f"Headlines for {ticker}")
    for i in range(len(ticker_news)):
        print(ticker_news[i]['url'])
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

    # Prints as a dataframe
    headlines_df = pd.DataFrame(ticker_news)
    headlines_df = headlines_df.drop(columns=['image','lang','hasPaywall','related'])
    pd.set_option("display.max_rows", None,"display.max_columns",None)
    print(headlines_df)
    print(headlines_df['url'])
    print()

def bid_ask_data(ticker):
    "Gets the bids and asks from IEX"
    api_url = f'https://cloud.iexapis.com/stable/deep/book?symbols={ticker}&token={IEX_CLOUD_API_TOKEN}'
    bid_ask_data = requests.get(api_url).json()
    bid_ask_df = pd.DataFrame(bid_ask_data)
    pd.set_option("display.max_rows", None,"display.max_columns",None)

    print(bid_ask_data)

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
    # print("Latest Price: ", latest_price)
    # print("")
    return float(latest_price)

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
    from tradingview_data import tradingview_recommendation
    # TradingView Recommendations
    # Include error checker since TradingView doesn't always have the ticker
    try:
        tradingview_recommendation(ticker)
    except: 
        print(f"No recommendations from TradingView for {ticker} \n")

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

def company_financials(ticker):
    "Returns financial data from IEX"
    stats_df = get_stats(ticker)
    financials_df = stats_df[['marketcap', 'totalCash', 'currentDebt', 'revenue', 'grossProfit', 'totalRevenue']]
    # Make figures more readable ($1000000 = $1M)
    financials_df = financials_df[:]/1000000

    print("Financials: \n", financials_df)
    print("")

def employee_information(ticker):
    "Employee info: # of employees, earnings per employee"
    stats_df = get_stats(ticker)
    employees = stats_df[['employees']]
    revenue_per_employee = stats_df['revenuePerEmployee']
    print("Employees: \n", employees)
    print("Revenue per Employee: \n", revenue_per_employee)
    print()

def stock_ratios(ticker):
    "Commonly used ratios: P/E, EBITDA, Put/Call Ratio"
    stats_df = get_stats(ticker)
    ratios = stats_df[['peRatio','EBITDA','putCallRatio']]
    print("Ratios: \n", ratios)
    print()

def stock_highlow_price(ticker):
    "52 Week High/Low Prices"
    
    stats_df = get_stats(ticker)
    high52 = stats_df[['week52high']]
    print("52-Week High Price: \n", high52)
    print()

def stock_percent_change(ticker):
    "Change Percentages"

    stats_df = get_stats(ticker)
    percent_changes = stats_df[['week52change','day5ChangePercent','day30ChangePercent','month1ChangePercent','month3ChangePercent','month6ChangePercent','year1ChangePercent','year2ChangePercent','year5ChangePercent','maxChangePercent']]
    # Make it a percent
    percent_changes = percent_changes[:]*100
    print("52-Week Percent Change: \n", percent_changes)
    print()



print("Done!")