# Checks various price targets:
# Large volume orders
# Upgrades/Downgrades from websites
# Strike price for options
# Technical indicators

from secrets import IEX_CLOUD_API_TOKEN
import requests
import pandas as pd

# Getting the following from the IEX API
# https://cloud.iexapis.com/ 
market_api_url = 'https://cloud.iexapis.com/stable/stock/market/list'
stock_api_url = 'https://cloud.iexapis.com/stable/stock'


def price_target_by_volume(ticker):
    "Get Price targets according to large volume orders"

    # Get average 10 day volume and 30 day volume from IEX
    stats_url = f'{stock_api_url}/{ticker}/advanced-stats?token={IEX_CLOUD_API_TOKEN}'
    stats = requests.get(stats_url).json()
    stats_df = pd.DataFrame(stats, index=[0])

    avg10volume = stats_df[['avg10Volume']]
    avg30volume = stats_df['avg30Volume']
    print("10 Day Average Volume: \n", avg10volume)
    print("30 Day Average Volume: \n", avg30volume)
    print()


    # Get hourly volume data over the period of 6 months
    avgvolume_interval = '1hour'
    timespan = '1m'

    # Pinpoint price target
    # Zoom into 6 month data
    historical_6month_api_url = f'{stock_api_url}/{ticker}/chart/{timespan}/?token={IEX_CLOUD_API_TOKEN}'
    historical_6month = requests.get(historical_6month_api_url).json()
    # historical_6month_df = pd.DataFrame(historical_6month, index=[0])
    # Scan for highest volume
    #max_volume = historical_6month_df['volume']
    volume_list = []
    for i in range(len(historical_6month)):
        volume_list.append(historical_6month[i].get('volume'))
        max_volume = max(volume_list)
    print(volume_list)
    max_volume = max(volume_list)
    print("MAX VOLUME: ", max_volume)
    print(len(historical_6month))
    max_position = volume_list.index(max(volume_list))
    max_vol_high_price = historical_6month[max_position].get('high')
    max_vol_low_price = historical_6month[max_position].get('low')
    print("Date of Max Volume: ", historical_6month[max_position].get('date'))
    print("Close price: ", historical_6month[max_position].get('close'))
    print("High: ", max_vol_high_price)
    print("Low: ",max_vol_low_price)

    # Find date of max volume
    # Find price of max volume date

    # Option to save data so we don't have to call API every time

    #historical_data_api_url = f'https://cloud.iexapis.com/stable/stock/{ticker}/chart/{timespan}/{date}'

    # Largest volume in 6 months

