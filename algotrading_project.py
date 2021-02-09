import numpy as np
import pandas as pd
import requests #for http requests
import xlsxwriter
import math
from secrets import IEX_CLOUD_API_TOKEN
from scipy import stats



# read list of tickers in csv file
stocks = pd.read_csv('genomics_stocks.csv')

# Getting the following from the IEX API
# https://cloud.iexapis.com/ 
# Market cap for each stock
# Price of each stock (quote)

symbol = 'AAPL'
# API URL for testing random sp500 stocks
api_url = f'https://sandbox.iexapis.com/stable/stock/{symbol}/quote/?token={IEX_CLOUD_API_TOKEN}'
#api_url = f'https://cloud.iexapis.com/stable/stock/{symbol}/intraday-prices/?token={IEX_CLOUD_API_TOKEN}'
#api_url = f'https://sandbox.iexapis.com/stable/stock/{symbol}/chart/{range}/{date}?token={IEX_CLOUD_API_TOKEN}'

#api_url = f'https://cloud.iexapis.com/stable/stock/{symbol}/chart/20200415?token={IEX_CLOUD_API_TOKEN}'

# Now we make an HHTP request and store it into a variable
# We use the requests library because it's apparently the best one
# Also .json() will make it json format and you can access it as a Python dictionary (key : value)
data = requests.get(api_url).json()

# type() allows you to see the type of object
type(data)

print(" ")
print(" ")
print(data['symbol'])

#############################################
# Parsing API Call

price = data['latestPrice'] #latestPrice is the one we want from the json
market_cap = data['marketCap']
print(price)
print(market_cap)
print(market_cap/1000000000000) #how many trillions the company is worth 

#############################################

# Adding stock data to a Pandas DataFrame
# Stock symbol, latest price, market cap, number of shares

# Make a python list
my_columns = [ 'Ticker', 'Stock Price', 'Market Cap', 'Number of Shares to Buy']
#final_dataframe = pd.DataFrame([[0,0,0,0]], columns = my_columns) #example

# We need to append the info so we use .append()
# Pandas dataframe is a 2D structure while Pandas series is a 1D (like lists and arrays)
final_dataframe = pd.DataFrame(columns = my_columns)
final_dataframe

# Pandas Series accepts a Python list
# Common error is: TypeError: Can only append a Series if ignore_index=True or if the Series has a name
# Almost always need to add ignore_index=True

print("Working as intended")

final_dataframe = final_dataframe.append(
    pd.Series(
        [
            symbol,
            price,
            market_cap,
            'N/A'
        ],
        index = my_columns # To tell it where the values go 
        ),
        ignore_index=True
)


#print(final_dataframe)
#final_dataframe

###############################################
# Now we loop through tickers
# Note: it is slow because one of the slowest things you can do in python is http request
# Later we will do batch api requests, but we will do single api requests for now

final_dataframe = pd.DataFrame(columns = my_columns)
for stock in stocks['Ticker'][:5]: #limiting this to 5 because it takes so long if not batch api calling
    api_url = f'https://sandbox.iexapis.com/stable/stock/{stock}/quote/?token={IEX_CLOUD_API_TOKEN}'
    data = requests.get(api_url).json()
    final_dataframe = final_dataframe.append(
        pd.Series(
            [
                stock,
                data['latestPrice'],
                data['marketCap'],
                'N/A'
                ],
                index = my_columns),
                ignore_index = True
        )


#print(final_dataframe)


#############################################
# Using Batch API Calls to Improve Performance
# IEX Cloud limits their batch API calls to 100 tickers per request. 
# Still, this reduces the number of API calls we'll make in this section from 500 to 5 - huge improvement!
# In this section, we'll split our list of stocks into groups of 100 and then make a batch API call 
# for each group.

# Split a list into sublists

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

symbol_groups = list(chunks(stocks['Ticker'], 100)) #gives us a list of a pandas series
#print(symbol_groups)

# Now we make a loop to append all the info to the lists
# Make string variable for the symbols
symbol_strings = []
print(" ")
print(" ")

for i in range(0, len(symbol_groups)):
    symbol_strings.append(','.join(symbol_groups[i]))
    #print(symbol_strings[i])
    final_dataframe = pd.DataFrame(columns = my_columns)

# Look up batch in the IEX API btw

for symbol_string in symbol_strings:
    batch_api_call_url = f'https://sandbox.iexapis.com/stable/stock/market/batch?symbols={symbol_string}&types=quote&token={IEX_CLOUD_API_TOKEN}'
    #print(batch_api_call_url)
    # Now to request the http urls we just made
    data = requests.get(batch_api_call_url).json()
    # Opposite of join is split so we can parse the data
    for symbol in symbol_string.split(','):
        final_dataframe = final_dataframe.append(
        pd.Series(
            [
                symbol,
                data[symbol]['quote']['latestPrice'],
                data[symbol]['quote']['marketCap'],
                'N/A'
                ],
                index = my_columns),
                ignore_index = True
        )


print(final_dataframe)


# Calculating the number of shares to buy
# Prompt value of portfolio

portfolio_size = input("Enter the value of your portfolio: ")

# Try-except to catch issues
try:
    portfolio_val = float(portfolio_size)
    print(portfolio_val)
except ValueError: 
    print("Error. Enter a number plz \n")
    portfolio_size = input("Enter the value of your portfolio: ")
    portfolio_val = float(portfolio_size)

# Now let's assume we distribute the portfolio funds evenly...
position_size = portfolio_val/len(final_dataframe.index)
print(position_size)

for i in range(0, len(final_dataframe.index)):
    final_dataframe.loc[i, 'Number of Shares to Buy'] = math.floor(position_size/final_dataframe.loc[i, 'Stock Price'])
    
# Can't buy fractional shares and don't want to over buy, so we use floor in the math library

print(final_dataframe)
#portfolio_leftover = 

###############################################################
# Excel output

# Initilize xlsxwriter object
writer = pd.ExcelWriter('recommended_trades.xlsx', engine = 'xlsxwriter')
final_dataframe.to_excel(writer, 'Recommended Trades', index = False)


#Creating the Formats We'll Need For Our .xlsx File
#Formats include colors, fonts, and also symbols like % and $. We'll need four main formats for our Excel document:

#String format for tickers
#\$XX.XX format for stock prices
#\$XX,XXX format for market capitalization
#Integer format for the number of shares to purchase


background_color = '#0a0a23'
font_color = '#ffffff'
string_format = writer.book.add_format(
    {
        'font_color': font_color,
        'bg_color': background_color,
        'border': 1
    }
)

dollar_format = writer.book.add_format(
    {
        'num_format': '$0.00',
        'font_color': font_color,
        'bg_color': background_color,
        'border': 1
    }
)

integer_format = writer.book.add_format(
    {
        'num_format': '0',
        'font_color': font_color,
        'bg_color': background_color,
        'border': 1
    }
)

#Applying the Formats to the Columns of Our .xlsx File
#We can use the set_column method applied to the writer.sheets['Recommended Trades'] 
#object to apply formats to specific columns of our spreadsheets.

#writer.sheets['Recommended Trades'].set_column('A:A', #This tells the method to apply the format to column B
#                     18, #This tells the method to apply a column width of 18 pixels
#                     string_format #This applies the format 'string_template' to the column
#                    )



column_formats = {
    'A': ['Ticker', string_format],
    'B': ['Stock Price', dollar_format],
    'C': ['Market Capitalization', dollar_format],
    'D': ['Number of Shares to Buy', integer_format]
}

for column in column_formats.keys():
    writer.sheets['Recommended Trades'].set_column(f'{column}:{column}', 18, column_formats[column][1])
    writer.sheets['Recommended Trades'].write(f'{column}1', column_formats[column][0], column_formats[column][1])


writer.save()

print(" ")
print(" ")
#print(stocks)

#print (api_url)