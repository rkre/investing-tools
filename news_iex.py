#
# Get general market information and news/stats for a ticker from IEX Cloud

import numpy as np
import pandas as pd
import requests #for http requests
import xlsxwriter
import math
from secrets import IEX_CLOUD_API_TOKEN
from scipy import stats


# Getting the following from the IEX API
# https://cloud.iexapis.com/ 

# Most Active
most_active_url = f'https://cloud.iexapis.com/stable/stock/market/list/mostactive?token={IEX_CLOUD_API_TOKEN}'
most_active = requests.get(most_active_url).json()

# Gainers
gainers_url = f'https://cloud.iexapis.com/stable/stock/market/list/gainers?token={IEX_CLOUD_API_TOKEN}'
gainers = requests.get(gainers_url).json()

# Losers
losers_url = f'https://cloud.iexapis.com/stable/stock/market/list/losers?token={IEX_CLOUD_API_TOKEN}'
losers = requests.get(losers_url).json()

# IEX Volume
iex_volume_url = f'https://cloud.iexapis.com/stable/stock/market/list/iexvolume?token={IEX_CLOUD_API_TOKEN}'
iex_volume = requests.get(iex_volume_url).json()

# IEX Percent
iex_percent_url = f'https://cloud.iexapis.com/stable/stock/market/list/iexpercent?token={IEX_CLOUD_API_TOKEN}'
iex_percent = requests.get(iex_percent_url).json()

# Sector Performance
# sector_perf_url = f'https://cloud.iexapis.com/stable/stock/market/sector-performance?token={IEX_CLOUD_API_TOKEN}'
# sector_perf = requests.get(sector_perf_url).json()

# Market Upcoming Events
# market_events_url = f'https://cloud.iexapis.com/stable/stock/market/upcoming-events?token={IEX_CLOUD_API_TOKEN}'
# market_events = requests.get(market_events_url).json()


# Market Upcoming Earnings # NOT SUPPORTED YET
# upcoming_earnings_url = f'https://cloud.iexapis.com/stable/stock/market/upcoming-earnings?token={IEX_CLOUD_API_TOKEN}'
# upcoming_earnings = requests.get(upcoming_earnings_url).json()


# Market Upcoming IPOs # NOT SUPPORTED BY IEX YET
# market_events_url = f'https://cloud.iexapis.com/stable/stock/market/upcoming-ipos?token={IEX_CLOUD_API_TOKEN}'
# market_events = requests.get(sector_perf_url.json())

# Market Upcoming Splits
# market_events_url = f'https://cloud.iexapis.com/stable/stock/market/upcoming-splits?token={IEX_CLOUD_API_TOKEN}'
# market_events = requests.get(sector_perf_url.json())

# Market Upcoming Dividends
# market_events_url = f'https://cloud.iexapis.com/stable/stock/market/upcoming-dividends?token={IEX_CLOUD_API_TOKEN}'
# market_events = requests.get(sector_perf_url.json())

# Now we make an HHTP request and store it into a variable
# We use the requests library because it's apparently the best one
# Also .json() will make it json format and you can access it as a Python dictionary (key : value)
# sp500_news = requests.get(api_url).json()


# Get the most active
print("Most Active Companies: ")
for i in range(len(most_active)):
    print(most_active[i]['companyName'])
    
print("")

# List Gainers
print("Gainers: ")
for i in range(len(gainers)):
    print(gainers[i]['companyName'])
    
print("")

# List Losers
print("Losers: ")
for i in range(len(losers)):
    print(losers[i]['companyName'])
    
print("")

# List IEX Volume
print("IEX Volume: ")
for i in range(len(iex_volume)):
    print(iex_volume[i]['companyName'])
    
print("")

# List IEX Percent
print("IEX Percent: ")
for i in range(len(iex_percent)):
    print(iex_percent[i]['companyName'])
    
print("")


# Upcoming Events NOT SUPPORTED YET
# print("Upcoming Market Events: ")
# for i in range(len(market_events)):
#     print(market_events[i]['companyName'])
    
print("")


# Enter ticker/symbol of the company to see stats and news
print("___________________________")
print("  ")
ticker = input("Enter ticker of a company to see the stats and news: ")
print("____________________________")
print(f"Here are the news and stats for {ticker}: ")

# Specific Ticker News
ticker_news_url = f'https://cloud.iexapis.com/stable/stock/{ticker}/news/last/5?token={IEX_CLOUD_API_TOKEN}'
ticker_news = requests.get(ticker_news_url).json()

# Get the headlines
print(f"Headlines for {ticker}")
for i in range(len(ticker_news)):
    print(ticker_news[i]['headline'])
    # Summary 
    print("")


# CEO Compensation
ceo_comp_url = f'https://cloud.iexapis.com/stable/stock/{ticker}/ceo-compensation?token={IEX_CLOUD_API_TOKEN}'
ceo_comp = requests.get(ceo_comp_url).json()
print("CEO Compensation: \n", ceo_comp)
print("")

# Fund Ownership
fund_ownership_url = f'https://cloud.iexapis.com/stable/stock/{ticker}/fund-ownership?token={IEX_CLOUD_API_TOKEN}'
fund_ownership = requests.get(fund_ownership_url).json()
print("Fund Ownership Info: \n", fund_ownership)
print("")

# 50 Day Moving Average
day50_MA_url = f'https://cloud.iexapis.com/stable/stock/{ticker}/stats/day50movingavg?token={IEX_CLOUD_API_TOKEN}'
day50_MA = requests.get(day50_MA_url).json()
print("50 Day MA: ", day50_MA['day50MovingAvg'])
print("")

# 200 Day Moving Average
day200_MA_url = f'https://cloud.iexapis.com/stable/stock/{ticker}/stats/day200movingavg?token={IEX_CLOUD_API_TOKEN}'
day200_MA = requests.get(day200_MA_url).json()
print("200 Day MA: ", day200_MA['day200MovingAvg'])
print("")

# Market Cap
# market_cap_url = f'https://cloud.iexapis.com/stable/stock/{ticker}/marketcap?token={IEX_CLOUD_API_TOKEN}'
# market_cap = requests.get(market_cap_url).json()
# print("Market Cap: ", market_cap)
# print("")

# Largest Trade Price
largest_trade_url = f'https://cloud.iexapis.com/stable/stock/{ticker}/largest-trades?token={IEX_CLOUD_API_TOKEN}'
largest_trade = requests.get(largest_trade_url).json()
print("Largest Trade Stats: ")
print(largest_trade)
print("")


# Watchlist News


# Enter ticker for news


# Parse news
