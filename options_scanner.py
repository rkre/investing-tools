# Scans options based on a watchlist to output expiration dates in order of expiration
# Calculates % profit for premium fee if selling put

import numpy as np
import pandas as pd
import requests #for http requests
import xlsxwriter
import math
from secrets import IEX_CLOUD_API_TOKEN, RH_MFA_CODE, RH_PASSWORD, RH_USER_EMAIL
from scipy import stats
import datetime
import pyotp
import robin_stocks as rh



# Upload or input watchlist/tickers from CSV file

#stocks = pd.read_csv('genomics_stocks.csv')
ticker = 'MARA'


def next_expiration_date():
    today = datetime.date.today()
    print("Today's date:", today)
    next_exp = today
    count = 0
    while count != 6:
        count += 1
        if next_exp.weekday() < 4:
            next_exp = next_exp + datetime.timedelta(1)
        if next_exp.weekday() > 4:
            next_exp = next_exp + datetime.timedelta(1)
        else:
            next_exp = next_exp
    # Format to YYYYMMDD to input into API url in other functions
    next_exp_string = next_exp.strftime('%Y%m%d')
    print('Next options expiration date: ', next_exp_string)
    return next_exp_string

def options_expiration_dates(ticker):
    "Find expiration dates for tickers from IEX Cloud (YYMMDD)"
    api_url = f'https://cloud.iexapis.com/stable/stock/{ticker}/options/?token={IEX_CLOUD_API_TOKEN}'
    data = requests.get(api_url).json()
    print(data)

def expiring_this_week(ticker):
    "Outputs tickers that are expiring this week"
    # Find the date YYMMDD for upcoming Friday

def robinhood_login():
    totp = pyotp.TOTP(RH_MFA_CODE).now()
    # print("Current OTP:", totp)
    login = rh.login(RH_USER_EMAIL, RH_PASSWORD, mfa_code=totp)

def robinhood_put_option_data(ticker, expiration):
    robinhood_login()
    
    strike_price = rh.find_options_by_expiration([ticker],
              expirationDate='2021-03-12',optionType='put',info='strike_price')
    bid_price = rh.find_options_by_expiration([ticker],
              expirationDate='2021-03-12',optionType='put',info='bid_price')
    ask_price = rh.find_options_by_expiration([ticker],
              expirationDate='2021-03-12',optionType='put',info='ask_price')
    volume = rh.find_options_by_expiration([ticker],
              expirationDate='2021-03-12',optionType='put',info='volume')
    open_interest = rh.find_options_by_expiration([ticker],
              expirationDate='2021-03-12',optionType='put',info='open_interest')
    tradability = rh.find_options_by_expiration([ticker],
              expirationDate='2021-03-12',optionType='put',info='tradability')
    #option_data_df = pd.DataFrame(option_data)

    option_data = [strike_price, bid_price, ask_price, volume, open_interest, tradability]
    

    strike_price = [float(i) for i in strike_price] 
    bid_price = [float(i) for i in bid_price] 
    ask_price = [float(i) for i in ask_price] 
    open_interest = [float(i) for i in open_interest] 
    volume = [float(i) for i in volume] 
    
    return(strike_price,bid_price,ask_price,open_interest,volume)



def iex_option_data(ticker, expiration):
    "IEX option data"

    api_url = f'https://cloud.iexapis.com/stable/stock/{ticker}/options/{expiration}/put/?token={IEX_CLOUD_API_TOKEN}'
    print(api_url)
    # Find strike price, premium fee
    stats = requests.get(api_url).json()
    stats_df = pd.DataFrame(stats)
    strike_price_df = stats_df[['strikePrice']]
    ask = stats_df[['ask']]
    bid = stats_df[['bid']]
    open_interest = stats_df[['openInterest']]
    volume = stats_df[['volume']]
    expiration = stats_df[['expirationDate']]
    last_updated = stats_df[['lastUpdated']]
    options_df = pd.concat([ask, bid, strike_price_df, expiration, open_interest, volume, last_updated], axis=1)
    pd.set_option("display.max_rows", None, "display.max_columns", None)

    print(options_df)
    print()

def sell_option_fee_percent(strike_price,fee):
    fee_percent = [i / j for i, j in zip(fee,strike_price)] 
    # for i in range(len(strike_price)):
    #     fee_percent[i] = fee[i]/strike_price[i]
        #collateral = strike_price*100
    print(fee_percent)
    return fee_percent


exp_date = next_expiration_date()
#put_premium_percent(ticker, exp_date)
strike_price, bid, ask, open_interest, volume = robinhood_put_option_data(ticker,exp_date)
# strike_price = rh_options.strike_price
# fee = rh_options.bid_price
sell_option_fee_percent(strike_price,bid)


