# To test strategies with historical data
import numpy as np
import pandas as pd
import requests #for http requests
import xlsxwriter
import math
import datetime
import os
from secrets import IEX_CLOUD_API_TOKEN
from scipy import stats
import csv



def input_ticker():
    ticker = input("Enter ticker: ")
    return ticker

def save_all_price_data(ticker):
    "Save all the price history"

    # Load list of tickers from csv

    # Save them all

def file_path_creator(ticker, date):
    "Creates file path to save stock data"

    year, month, day = parse_date(date)

    file_path = f'./data/stocks/{ticker}/price_data/{year}/{month}/'
    file_name = f'{day}.csv'

    if not os.path.exists(file_path):
        os.makedirs(file_path)

    full_name = os.path.join(file_path, file_name) 

    return full_name

def save_csv_data_locally(ticker, data, date):
    "Saves a pandas dataframe to csv locally"

    # We're going to save this data in batches for easy indexing and storage and organization. 
    
    file_path = file_path_creator(ticker, date)


    data.to_csv(file_path)

    # with open(file_path, 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerows(data)

    return print("Saved locally!")

def parse_date(date):
    "Takes YYYYMMDD and parses for searching prices"

    year = date[0:4]
    month = date[4:6]
    day = date[6:8]
    print(date)
    print(year)
    print(month)
    print(day)


    return year, month, day


def search_price_at_datetime(ticker, date, time):
    "Searches through the folders for the data requested based on ticker, date, and time"

    file_path = file_path_creator(ticker, date)

    price_info_df = load_stock_data_locally(file_path)

    print("SEARCH RESULTS: \n")
    print(price_info_df.head())

    average_price = price_info_df.iloc[time, 'average']

    return average_price

def load_stock_data_locally(file_path):
    "Loads from local file as dataframe"

    df = pd.read_csv(file_path, index_col=1)
    df = df.transpose()

    return df


def load_as_csv(ticker):
    # initializing the titles and rows list
    fields = []
    rows = []
    # reading csv file
    file_path = f'{ticker}.csv'
    with open(file_path, 'r') as csvfile:
        # creating a csv reader object
        csvreader = csv.reader(csvfile)
        
        # extracting field names through first row
        fields = next(csvreader)
    
        # extracting each data row one by one
        for row in csvreader:
            rows.append(row)
    
        # get total number of rows
        print("Total no. of rows: %d"%(csvreader.line_num))
    
    # printing the field names
    print('Field names are:' + ', '.join(field for field in fields))
    
    #  printing first 5 rows
    print('\nFirst 5 rows are:\n')
    for row in rows[:5]:
        # parsing each column of a row
        for col in row:
            print("%10s"%col),
        print('\n')

def load_stock_data_from_firebase(ticker, data_range):
    "Loads from firebase"

def save_data_to_firebase(ticker):
    return print("Saved to firebase!")


def historical_stock_data_longterm(ticker, specific_date, format_type):
    "Historical price data from IEX for long term yearly, monthly"

    stock_api_url = 'https://cloud.iexapis.com/stable/stock'

    # We want minute prices so we are going to request a specific date from the IEX API like this: /stable/stock/twtr/chart/date/20200220

    # date_range should be 'date' for when you want a specific date with minute-by-minute time info
    # Otherwise, specific_date is empty and date_range can be these: '1m','1yr','2yr', '1w', '1d' (returns latest day)

    # We will set it to get minute-by-minute daily info 

    date_range = 'date'

    historical_prices_url = f'{stock_api_url}/{ticker}/chart/{date_range}/{specific_date}?token={IEX_CLOUD_API_TOKEN}&format={format_type}'
    print(historical_prices_url)
    prices = requests.get(historical_prices_url).json()
    open_prices = []
    close = []
    high = []
    low = []
    volume = []
    date = []
    print(prices[0]['open'])
    print(len(prices))
    for i in range(0,len(prices)):
       # print(i) 
        date.append(prices[i]['date'])
        open_prices.append(prices[i]['open'])
      #  print(open_prices)
        close.append(prices[i]['close'])
        high.append(prices[i]['high'])
        low.append(prices[i]['low'])
        volume.append(prices[i]['volume'])
        

    fields = ['date', 'minute', 'average', 'high', 'low', 'volume']

    #fields = ['date', 'open', 'close', 'high', 'low', 'volume']
    #historical_prices_df = pd.DataFrame(historical_prices, index=[0])
    #pd.set_option("display.max_rows", None, "display.max_columns", None)

    prices_df = pd.DataFrame(columns = fields)
    prices_df
    prices_df = prices_df.append(
    pd.Series(
        [
            date,
            open_prices,
            close,
            high,
            low,
            volume,
            
        ],
        index = fields # To tell it where the values go 
        ),
        ignore_index=True
    )

    print(prices_df)
    return prices_df

def historical_stock_data(ticker, specific_date, format_type):
    "Historical price data from IEX for a specific day"

    stock_api_url = 'https://cloud.iexapis.com/stable/stock'

    # We want minute prices so we are going to request a specific date from the IEX API like this: /stable/stock/twtr/chart/date/20200220

    # date_range should be 'date' for when you want a specific date with minute-by-minute time info
    # Otherwise, specific_date is empty and date_range can be these: '1m','1yr','2yr', '1w', '1d' (returns latest day)

    # We will set it to get minute-by-minute daily info 

    date_range = 'date'

    historical_prices_url = f'{stock_api_url}/{ticker}/chart/{date_range}/{specific_date}?token={IEX_CLOUD_API_TOKEN}&format={format_type}'
    print(historical_prices_url)
    prices = requests.get(historical_prices_url).json()
    
    minute = []
    average = []
    high = []
    low = []
    volume = []
    

    print(prices[0]['minute'])
    print(len(prices))
    for i in range(0,len(prices)):


        minute.append(prices[i]['minute'])
  
        average.append(prices[i]['close'])
        high.append(prices[i]['high'])
        low.append(prices[i]['low'])
        volume.append(prices[i]['volume'])
        

    fields = ['minute', 'average', 'high', 'low', 'volume']

    #fields = ['date', 'open', 'close', 'high', 'low', 'volume']
    #historical_prices_df = pd.DataFrame(historical_prices, index=[0])
    #pd.set_option("display.max_rows", None, "display.max_columns", None)

    prices_df = pd.DataFrame(columns = fields)
    print(prices_df)
    prices_df = prices_df.append(
    pd.Series(
        [
            minute,
            average,
            high,
            low,
            volume,
            
        ],
        index = fields # To tell it where the values go 
        ),
        ignore_index=True
    )

    print(prices_df)
    return prices_df

def buy_at_specific_time(ticker):

    date = input("Enter the day you bought YYYYMMDD: ")
    time = input("Enter the time you bought: ")

    date_range = "date" # set parameter to date to enable specific date search
    
    historical_stock_data(ticker, date_range, date)

    return price_bought

def sell_at_specific_time(ticker):

    date = input("Enter the day you sold YYYYMMDD: ")
    time = input("Enter the time you sold: ")

    date_range = "date" # set parameter to date to enable specific date search
    
    historical_stock_data(ticker, date_range, date)

    return price_sold

def buy_at_limit_price(ticker):

    date = input("Enter the day you bought YYYYMMDD: ")
    time = input("Enter the time you bought: ")

    date_range = "date" # set parameter to date to enable specific date search
    
    historical_stock_data(ticker, date_range, date)

    return price_bought

def sell_at_limit_price(ticker):

    date = input("Enter the day you bought YYYYMMDD: ")
    time = input("Enter the time you bought: ")

    date_range = "date" # set parameter to date to enable specific date search
    
    historical_stock_data(ticker, date_range, date)

    return price_bought

def input_day_and_time():
    "Tales the day and time from user"

    date = input("Enter date YYYYMMDD : ")
    time = input("Enter time HH:mm : ")

    return date, time

ticker = input_ticker()
print(ticker)


specific_date = '20210518'
time = '9:30'



df_data = historical_stock_data(ticker, specific_date, format_type='json')
print(specific_date)
save_csv_data_locally(ticker,df_data, specific_date)
price = search_price_at_datetime(ticker, specific_date, time)
print(price)


# Buy/Sell Strategy Tester
# We need:
# Buy price
# Buy price date
# Sell price
# Sell price date

# Put spread Strategy Tester
# We need:
# Date trade was placed
# Sell/Buy strike prices/Breakeven price
# Expiration date
# Close price of the ticker on the expiration date



