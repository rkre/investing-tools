# Calculates the profitability of various Iron Corridor strategies


from secrets import IEX_CLOUD_API_TOKEN
import requests
import pandas as pd
from stock_scanner import latest_price
import volatility_scanner as vs

# Getting the following from the IEX API
# https://cloud.iexapis.com/ 
market_api_url = 'https://cloud.iexapis.com/stable/stock/market/list'
stock_api_url = 'https://cloud.iexapis.com/stable/stock'

def iron_condor(ticker):
    "Iron corridor strike price calculator"
    # Current price of the stock
    current_price = latest_price(ticker)
    current_price = float(current_price)
    # Average volatility 
    avg_vol = .03 # 10%
    max_vol = vs.highest_volatility(ticker)

    # Calculate strike prices to buy_put, sell_put, sell_call, buy_call
    sell_put_strike = -(current_price * max_vol) + current_price
    buy_put_strike = -(sell_put_strike * (max_vol-avg_vol)) + sell_put_strike
    sell_call_strike = (current_price * max_vol) + current_price
    buy_call_strike = (sell_call_strike * (max_vol-avg_vol)) + sell_call_strike
    # sell_put_strike = -(current_price * avg_vol) + current_price
    # buy_put_strike = -(sell_put_strike * (max_vol-avg_vol)) + sell_put_strike
    # sell_call_strike = (current_price * avg_vol) + current_price
    # buy_call_strike = (sell_call_strike * (max_vol-avg_vol)) + sell_call_strike

    # Find premium fees for selected strike prices expiring the same week
    print(sell_put_strike,sell_call_strike)

    IC_strikes = [sell_put_strike, buy_put_strike, sell_call_strike, buy_call_strike]
    return IC_strikes

ticker = 'CAT'
IC_strikes = iron_condor(ticker)
print(IC_strikes)
print("Current price: ", latest_price(ticker))
    
