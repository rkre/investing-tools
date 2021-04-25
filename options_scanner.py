# Scans options based on a watchlist to output expiration dates in order of expiration
# Calculates % profit for premium fee if selling put
# Robinhood latest price
# 

#import numpy as np
import pandas as pd
import requests #for http requests
#import xlsxwriter
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

    # Format to YYYYMMDD to input into IEX API url in other functions
    next_exp_string = next_exp.strftime('%Y%m%d')
    # Format with dashes for Robinhood functions
    next_exp_string_dashes = next_exp.strftime('%Y-%m-%d')

    print('Next options expiration date: ', next_exp_string_dashes)
    return next_exp_string, next_exp_string_dashes

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
              expirationDate=expiration,optionType='put',info='strike_price')
    bid_price = rh.find_options_by_expiration([ticker],
              expirationDate=expiration,optionType='put',info='bid_price')
    ask_price = rh.find_options_by_expiration([ticker],
              expirationDate=expiration,optionType='put',info='ask_price')
    volume = rh.find_options_by_expiration([ticker],
              expirationDate=expiration,optionType='put',info='volume')
    open_interest = rh.find_options_by_expiration([ticker],
              expirationDate=expiration,optionType='put',info='open_interest')
    tradability = rh.find_options_by_expiration([ticker],
              expirationDate=expiration,optionType='put',info='tradability')
    #option_data_df = pd.DataFrame(option_data)

    options_data = [strike_price, bid_price, ask_price, volume, open_interest, tradability]
    

    strike_price = [float(i) for i in strike_price] 
    bid_price = [float(i) for i in bid_price] 
    ask_price = [float(i) for i in ask_price] 
    open_interest = [float(i) for i in open_interest] 
    volume = [float(i) for i in volume] 
    mid_price = [(i + j)/2 for i, j in zip(bid_price,ask_price)]
   
    return(strike_price,bid_price,ask_price,mid_price,open_interest,volume)
    

def robinhood_call_option_data(ticker, expiration):
    robinhood_login()
    
    strike_price = rh.find_options_by_expiration([ticker],
              expirationDate=expiration,optionType='call',info='strike_price')
    bid_price = rh.find_options_by_expiration([ticker],
              expirationDate=expiration,optionType='call',info='bid_price')
    ask_price = rh.find_options_by_expiration([ticker],
              expirationDate=expiration,optionType='call',info='ask_price')
    volume = rh.find_options_by_expiration([ticker],
              expirationDate=expiration,optionType='call',info='volume')
    open_interest = rh.find_options_by_expiration([ticker],
              expirationDate=expiration,optionType='call',info='open_interest')
    tradability = rh.find_options_by_expiration([ticker],
              expirationDate=expiration,optionType='call',info='tradability')
    #option_data_df = pd.DataFrame(option_data)

    options_data = [strike_price, bid_price, ask_price, volume, open_interest, tradability]
    

    strike_price = [float(i) for i in strike_price] 
    bid_price = [float(i) for i in bid_price] 
    ask_price = [float(i) for i in ask_price] 
    open_interest = [float(i) for i in open_interest] 
    volume = [float(i) for i in volume]
    mid_price = [(i + j)/2 for i, j in zip(bid_price,ask_price)]
    
    return(strike_price,bid_price,ask_price,mid_price,open_interest,volume)

def iron_condor_profit_checker(ticker):
    "Checks premium profit if buying/selling puts and calls"
    exp_date, exp_date_rh = next_expiration_date()

    # Put Option Table 
    strike_price, bid, ask, mid, open_interest, volume = robinhood_put_option_data(ticker,exp_date_rh)
    
    data_tuples = list(zip(strike_price,mid,mid[1:]))
    #print(data_tuples)
    # Min Risk Profit is the profit received if buying and selling two closest puts
    options_data = pd.DataFrame(data_tuples, columns=['Strike_Price', 'Premium', 'Min Risk Profit'])
    latest_price = robinhood_latest_price(ticker)
    options_data = options_data[options_data.Strike_Price <= latest_price]
    options_data.sort_values(by='Strike_Price', inplace=True, ascending=False)
    # Add sell/buy put fee column
    sell_put_fee = options_data["Premium"]
    # Add column for buy put fee
    buy_put_fee = sell_put_fee[1:]
    buy_put_fee.loc[len(buy_put_fee)] = 0
    zero = pd.Series(0, index = [0])
    #zero_df = pd.Dataframe([ float(0)]) #, columns=['Premium'])
    #buy_put_fee.append(zero)
    #buy_put_fee.append(zero_df, ignore_index=True)
    options_data.insert(3,"Buy Put Fee",buy_put_fee,True)
    print(zero)
    print(sell_put_fee)
    print(buy_put_fee)
    print(strike_price[1:5])
    min_risk_profit = [(i-j) for i, j in zip(sell_put_fee, buy_put_fee)]
    
    print(min_risk_profit)

    print("Iron Condor \n Put Sell-Buy Profit List")
    print(options_data)


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
    # Iterates through the (strike_price,fee) tuples, divides fee/strike_price and returns a fee_percent list
    fee_percent = [(i / j)*100 for i, j in zip(fee,strike_price)]
    return fee_percent

def robinhood_latest_price(ticker):
    "Returns float of latest price"
    price = rh.get_latest_price(ticker)
    price_float = float(price[0])
    return price_float

def sell_put_list(ticker):
    "Outputs a table to calculate % profit from premiums if you sell puts at various strike prices"
    
    exp_date, exp_date_rh = next_expiration_date()
    strike_price, bid, ask, mid, open_interest, volume = robinhood_put_option_data(ticker,exp_date_rh)
    
    fee_percent = sell_option_fee_percent(strike_price,mid)
    data_tuples = list(zip(fee_percent,strike_price,mid))
    options_data = pd.DataFrame(data_tuples, columns=['Profit','Strike_Price', 'Premium'])
    latest_price = robinhood_latest_price(ticker)
    options_data.sort_values(by='Strike_Price', inplace=True, ascending=False)

    options_data = options_data[options_data.Strike_Price <= latest_price]
    options_data = options_data[options_data.Profit > 1]
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    print(robinhood_latest_price(ticker))
    print("Sell Put Premium Profitability List")
    print(options_data)
    

def sell_call_list(ticker):
    "Outputs a table to calculate % profit from premiums if you sell puts at various strike prices"
    exp_date, exp_date_rh = next_expiration_date()
    strike_price, bid, ask, mid, open_interest, volume = robinhood_call_option_data(ticker,exp_date_rh)
    
    fee_percent = sell_option_fee_percent(strike_price,mid)
    data_tuples = list(zip(fee_percent,strike_price,mid))
    options_data = pd.DataFrame(data_tuples, columns=['Profit','Strike_Price', 'Premium'])
    latest_price = robinhood_latest_price(ticker)
    options_data.sort_values(by='Strike_Price', inplace=True, ascending=False)
    # for i in range(len(options_data)):
    #     if options_data['Strike Price'][i] > latest_price:
    #         options_data.drop([i])

    options_data = options_data[options_data.Strike_Price >= latest_price]
    options_data = options_data[options_data.Profit > 1]
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    print("Sell Call Premium Profitability List")
    print(options_data)
    print(robinhood_latest_price(ticker))

#sell_put_list(ticker)
#sell_call_list(ticker)
iron_condor_profit_checker(ticker)