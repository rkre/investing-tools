# Tracking Portfolio:
# Profits
# Shares held and % of portfolio allocation
# Stop loss, price target


############################
# Portfolio Sheet
# Ticker | % Allocated | Profit % | Price Target | Stop Loss

import numpy as np
import pandas as pd
# import requests #for http requests
import xlsxwriter
import math
# import datetime
# import time
# from secrets import IEX_CLOUD_API_TOKEN
# from scipy import stats

portfolio_columns = ['Ticker', '% Allocated', 'Profit %', 'Price Target', 'Stop Loss']
portfolio_df = pd.DataFrame(columns = portfolio_columns)
print(portfolio_df)

# Add ticker, amount bought, price target, stop loss, buy price
print("1 - Buying or 2 - Selling?")
menu_option = input("Enter: ")

if menu_option == '1':
    # Buy shares selected
    # Prompt for buy info
    ticker = input("Ticker: ")
    buy_amt = input("$ Amount Bought: ")
    price_target = input("Price Target: ")
    stop_loss = input("Stop Loss: ")
    buy_data = [ticker, buy_amt, price_target, stop_loss]
    # Add ticker and info to portfolio
    portfolio = portfolio.append(buy_data)
    loop_option = input("Return to 1 - menu or 2 - exit: ")

elif menu_option == '2':
    # Sell shares selected
    ticker = input("Ticker: ")
    # Check if stock is currently owned in the portfolio

    sell_amt = input("$ Amount to sell: ")
    # Check if the amount is within range of amount owned

    loop_option = input("Return to 1 - menu or 2 - exit: ")

print(portfolio)
    
