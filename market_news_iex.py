#
# Get general market information and news/stats for a ticker from IEX Cloud

import numpy as np
import pandas as pd
import requests #for http requests
import xlsxwriter
import math
import datetime
import time
from short_seller_checker import short_seller_checker
#from tradingview_data import tradingview_recommendation
from secrets import IEX_CLOUD_API_TOKEN
from scipy import stats


# Getting the following from the IEX API
# https://cloud.iexapis.com/ 
market_api_url = 'https://cloud.iexapis.com/stable/stock/market/list'

def most_active():
    # Most Active
    most_active_url = f'{market_api_url}/mostactive?token={IEX_CLOUD_API_TOKEN}'
    most_active = requests.get(most_active_url).json()
    # Get the most active
    print("Most Active Companies: ")
    for i in range(len(most_active)):
        print(most_active[i]['companyName'])
        
    print("")

def gainers():
    # Gainers
    gainers_url = f'{market_api_url}/gainers?token={IEX_CLOUD_API_TOKEN}'
    gainers = requests.get(gainers_url).json()

    # List Gainers
    print("Gainers: ")
    for i in range(len(gainers)):
        print(gainers[i]['companyName'])
        
    print("")

def losers():
    # Losers
    losers_url = f'{market_api_url}/losers?token={IEX_CLOUD_API_TOKEN}'
    losers = requests.get(losers_url).json()

    # List Losers
    print("Losers: ")
    for i in range(len(losers)):
        print(losers[i]['companyName'])
        
    print("")
    
def iex_volume():
    # IEX Volume
    iex_volume_url = f'{market_api_url}/list/iexvolume?token={IEX_CLOUD_API_TOKEN}'
    iex_volume = requests.get(iex_volume_url).json()

    # List IEX Volume
    print("IEX Volume: ")
    for i in range(len(iex_volume)):
        print(iex_volume[i]['companyName'])
        
    print("")

def iex_percent():
    # IEX Percent
    iex_percent_url = f'{market_api_url}/iexpercent?token={IEX_CLOUD_API_TOKEN}'
    iex_percent = requests.get(iex_percent_url).json()

    # List IEX Percent
    print("IEX Percent: ")
    for i in range(len(iex_percent)):
        print(iex_percent[i]['companyName'])
        
    print("")



# Sector Performance
# sector_perf_url = f'{market_api_url}/sector-performance?token={IEX_CLOUD_API_TOKEN}'
# sector_perf = requests.get(sector_perf_url).json()

# Market Upcoming Events
# market_events_url = f'{market_api_url}/upcoming-events?token={IEX_CLOUD_API_TOKEN}'
# market_events = requests.get(market_events_url).json()


# Market Upcoming Earnings # NOT SUPPORTED YET
# upcoming_earnings_url = f'{market_api_url}/upcoming-earnings?token={IEX_CLOUD_API_TOKEN}'
# upcoming_earnings = requests.get(upcoming_earnings_url).json()


# Market Upcoming IPOs # NOT SUPPORTED BY IEX YET
# market_events_url = f'{market_api_url}/upcoming-ipos?token={IEX_CLOUD_API_TOKEN}'
# market_events = requests.get(sector_perf_url.json())

# Market Upcoming Splits
# market_events_url = f'{market_api_url}/upcoming-splits?token={IEX_CLOUD_API_TOKEN}'
# market_events = requests.get(sector_perf_url.json())

# Market Upcoming Dividends
# market_events_url = f'{market_api_url}/upcoming-dividends?token={IEX_CLOUD_API_TOKEN}'
# market_events = requests.get(sector_perf_url.json())

# Upcoming Events NOT SUPPORTED YET
# print("Upcoming Market Events: ")
# for i in range(len(market_events)):
#     print(market_events[i]['companyName'])

# Watchlist News


# Enter ticker for news


# Parse news
