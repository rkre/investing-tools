# Scans options based on a watchlist to output expiration dates in order of expiration
# Calculates % profit for premium fee if selling put
# Robinhood latest price
# 

#import numpy as np
import pandas as pd
import requests #for http requests
#import xlsxwriter
#import math
from secrets import IEX_CLOUD_API_TOKEN, RH_MFA_CODE, RH_PASSWORD, RH_USER_EMAIL
#from scipy import stats
import datetime
import pyotp
import robin_stocks as rh
from stock_scanner import latest_price



# Upload or input watchlist/tickers from CSV file

#stocks = pd.read_csv('genomics_stocks.csv')
#ticker = 'CAT'
ticker = input("Enter ticker: ")



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
    
    return data

def expiring_this_week(ticker):
    "Outputs tickers that are expiring this week"
    # Find the date YYMMDD for upcoming Friday

def robinhood_login():
    totp = pyotp.TOTP(RH_MFA_CODE).now()
    # print("Current OTP:", totp)
    login = rh.login(RH_USER_EMAIL, RH_PASSWORD, mfa_code=totp)

def robinhood_put_option_data(ticker, expiration):
    robinhood_login()

    # Try to optimize for speed here somehow
    # options_data = rh.find_options_by_expiration([ticker],
    #           expirationDate=expiration,optionType='put')
    
    strike_price = rh.find_options_by_expiration([ticker],
              expirationDate=expiration,optionType='put',info='strike_price')
    bid_price = rh.find_options_by_expiration([ticker],
              expirationDate=expiration,optionType='put',info='bid_price')
    ask_price = rh.find_options_by_expiration([ticker],
              expirationDate=expiration,optionType='put',info='ask_price')
    # volume = rh.find_options_by_expiration([ticker],
    #           expirationDate=expiration,optionType='put',info='volume')
    # open_interest = rh.find_options_by_expiration([ticker],
    #           expirationDate=expiration,optionType='put',info='open_interest')
    # tradability = rh.find_options_by_expiration([ticker],
    #           expirationDate=expiration,optionType='put',info='tradability')
    # #option_data_df = pd.DataFrame(option_data)

    # put_list = rh.find_options_by_expiration([ticker],
    #           expirationDate=expiration,optionType='put')
    # strike_price = put_list[:]['strike_price']
    # bid_price = put_list[:]['bid_price']
    # ask_price = put_list[:]['ask_price']
    # volume = put_list[:]['volume']
    # open_interest = put_list[:]['open_interest']
    # tradability = put_list[:]['tradability']
    

    # print("!!!!!!!!!!!!!!", put_list)
    # print(strike_price)
    # print("")

    options_data = [strike_price, bid_price, ask_price]
    

    strike_price = [float(i) for i in strike_price] 
    bid_price = [float(i) for i in bid_price] 
    ask_price = [float(i) for i in ask_price] 
    # open_interest = [float(i) for i in open_interest] 
    # volume = [float(i) for i in volume] 
    mid_price = [(i + j)/2 for i, j in zip(bid_price,ask_price)]
   
    return(strike_price,bid_price,ask_price,mid_price)
    

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

def put_credit_spread(ticker, exp_date_rh):
    "Put Table with sell-buy put profit calculated"
    # Put Option Table 
    print("Loading from Robinhood...")
    strike_price, bid, ask, mid = robinhood_put_option_data(ticker,exp_date_rh)  
    sell_strikes = strike_price
    buy_strikes = strike_price[1:]
    #collat = [sell_strikes[i] - buy_strikes[i] for i in ]
    #print(collat)
    data_tuples = list(zip(strike_price,mid))
    print("Converting data tuples")

    # Min Risk Profit is the profit received if buying and selling two closest puts
    options_data = pd.DataFrame(data_tuples, columns=['Strike_Price', 'Sell_Put'])
    latest_price = robinhood_latest_price(ticker)
    options_data = options_data[options_data.Strike_Price <= latest_price]

    # Cut the .01 priced options
    cutoff = .01
    options_data = options_data[options_data.Sell_Put > cutoff]
    options_data.sort_values(by='Strike_Price', inplace=True, ascending=False)

    
    # Create a Sell Put Table and a Buy Put Table
    sell_put_fee = options_data["Sell_Put"]
    sell_put_strikes = options_data["Strike_Price"]
    buy_put_strikes = sell_put_strikes[1:]
    buy_put_fee = sell_put_fee[1:]
    #print("Sell Put List", sell_put_df)
    #print("Buy Put List", buy_put_df)
    options_data.insert(1,"Buy_Strike", buy_put_strikes)
    options_data["Buy_Strike"] = options_data['Buy_Strike'].shift(-1)

    options_data.insert(3,"Buy_Put",buy_put_fee)
    options_data["Buy_Put"] = options_data['Buy_Put'].shift(-1)
    options_data = options_data[:-1]

    min_risk_profit = [(i-j) for i, j in zip(sell_put_fee, buy_put_fee)]
    put_credit = [i*100.0*100.0 for i in min_risk_profit]
    risk = [((i-j-k)) for i, j, k in zip(sell_put_strikes, buy_put_strikes, min_risk_profit)]
    risk_p = [i*100.0*100.0 for i in risk]
    options_data["Collateral"] = risk_p
    options_data = options_data[options_data.Collateral > 0.0]
    risk_p = options_data["Collateral"]
    options_data.insert(4,"Put_Profit", put_credit)
    options_data = options_data[options_data.Put_Profit > 0.0]
    put_credit = options_data["Put_Profit"]
    print("credit", put_credit)
    print("risk", risk_p)

    profit_percent = [100.0/float(float(i)//float(j)) for i, j in zip(risk_p, put_credit)]
    
    #print(min_risk_profit)
    options_data.insert(5,"Percent_Profit", profit_percent)
    # Cutoff algo function goes here
    #cutoff = 1.0
    #options_data = options_data[options_data.Put_Profit > cutoff]


    # Collateral/Max Loss
    
    options_data["Collateral"] = risk_p #options_data["Strike_Price"] - buy_put_df["Strike_Price"]
    # collateral = float(sell_put_df["Strike_Price"]) - float(buy_put_df["Strike_Price"])
    # print('COLLAT\n', sell_put_df)
    # print(buy_put_df)
    # print(collateral)
    options_data = options_data[options_data.Percent_Profit > 5.0]


    print(" \nPut Sell-Buy Profit List")
    print(options_data)
    return options_data

def call_credit_spread(ticker, exp_date_rh):
    "Call Table with sell-buy call profit calculated"
    # Put Option Table 
    strike_price, bid, ask, mid, open_interest, volume = robinhood_call_option_data(ticker,exp_date_rh)
    
    data_tuples = list(zip(strike_price,mid))

    # Min Risk Profit is the profit received if buying and selling two closest puts
    options_data = pd.DataFrame(data_tuples, columns=['Strike_Price', 'Sell Call'])
    latest_price = robinhood_latest_price(ticker)
    options_data = options_data[options_data.Strike_Price >= latest_price]
    options_data.sort_values(by='Strike_Price', inplace=True, ascending=True)

    # Cut the .01 priced options
    cutoff = .02
    options_data.loc[options_data["Sell Call"] > cutoff]


    # Add sell/buy call fee column
    sell_call_fee = options_data["Sell Call"]
    # Add column for call put fee
    buy_call_fee = sell_call_fee[1:]

    options_data.insert(2,"Buy Call",buy_call_fee)
    options_data["Buy Call"] = options_data['Buy Call'].shift(-1)
    options_data = options_data[:-1]

    min_risk_profit = [(i-j) for i, j in zip(sell_call_fee, buy_call_fee)]
    
    #print(min_risk_profit)
    options_data.insert(3,"Call Profit",min_risk_profit)

    print("Iron Condor \n Call Sell-Buy Profit List")
    print(options_data)
    return options_data

def iron_condor_profit_checker(ticker, exp_date):
    "Checks premium profit if buying/selling puts and calls"
    #exp_date, exp_date_rh = next_expiration_date()

    puts = put_credit_spread(ticker, exp_date)
    calls = call_credit_spread(ticker, exp_date)

    print("***************")
    #print(puts)
    #print(calls)
    print("Latest price: ", latest_price(ticker))
    print(robinhood_latest_price(ticker))


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

def select_exp_date(ticker):
    "User selects exp date from list of available expiration dates"
    exp_dates = options_expiration_dates(ticker)
    # Organize list of expiration dates and format into Robinhood exp format
    print("List of options expirations dates for ", ticker)
    n = 0
    exp_date_formatted = []
    for day in exp_dates:
        # Format with dashes for Robinhood functions
        dt_convert = datetime.datetime.strptime(day, '%Y%m%d')
        exp_date_formatted.append(dt_convert.strftime('%Y-%m-%d'))
        print(n, exp_date_formatted[n])
        n = n+1
    else:
        print("")

    print("Select expiration date [ 0 -", n-1, "]")
    date_selected = input("Selection: ")

    print("Selected ", exp_date_formatted[int(date_selected)])


    return exp_date_formatted[int(date_selected)]

def option_table_profit_cutoff(exp_date):
    "Returns a cutoff factor based on how far the exp date is from today. Used to make quick profit analysis decisions"

    #Find today's date
    today = datetime.date.today()
    print("Today's date:", today)

    # Calculate how many days until expiration
    days_til_expiry = exp_date - today

    # If expiring this week, factor of 1

    
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

def max_loss(strategy, credit, sell_strike_price, buy_strike_price):
    "Calculates the max loss from a strategy"

def collateral_calc(credit, sell_strike_price, buy_strike_price):
    "Calculates the collateral needed for the trade based on the credit received and strike price or prices if strategy is a spread"




#######################
selected_exp_date = select_exp_date(ticker)

print(" 1 - Sell Put ")
print(" 2 - Put Spread ")
print(" 3 - Call Spread ")
print(" 4 - Iron Condor ")
strategy = input("Select strategy: ")

if strategy == '1':
    sell_put_list(ticker)
elif strategy == '2':
    put_credit_spread(ticker, selected_exp_date)
elif strategy == '3':
    call_credit_spread(ticker, selected_exp_date)
elif strategy == '4':
    iron_condor_profit_checker(ticker, selected_exp_date)
else:
    print("Error")
#options_expiration_dates(ticker)
print("Ticker: ", ticker)
print("Latest price: ", robinhood_latest_price(ticker))
print(selected_exp_date)