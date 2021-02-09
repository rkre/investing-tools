# This is the menu to select what to show

# Options are: 
# Most active, gainers, losers
# Stock info feed
# Latest news
# Buy or Sell Stock

#
# Get general market information and news/stats for a ticker from IEX Cloud


import market_news_iex as market_news
import stock_scanner as ss



selection = input("Would you like to see: \n 1 - Market Movers \n 2 - IEX Volume and Percent \n 3 - Information about a particular stock \nSelection: ")

if selection == '1':

    # Most Active
    market_news.most_active()

    # Gainers
    market_news.gainers()

    # Losers
    market_news.losers()

elif selection == '2':

    # IEX Volume
    market_news.iex_volume()

    # IEX Percent
    market_news.iex_percent()
    

elif selection == '3':

    # Enter ticker/symbol of the company to see stats and news
    print("___________________________")
    print("  ")
    ticker = input("Enter ticker of a company to see the stats and news: ")
    print("____________________________")
    print(f"Here are the news and stats for {ticker}: \n")

    # Specific Ticker News
    ss.latest_iex_news(ticker)

    # CEO Compensation
    ss.ceo_compensation(ticker)

    # Fund Ownership
    ss.fund_ownership(ticker)

    # 50 Day Moving Average
    ss.day50MA(ticker)

    # 200 Day Moving Average
    ss.day200MA(ticker)

    # Latest Price
    ss.latest_price(ticker)
    
    # Short Seller Checker
    ss.short_seller_information(ticker)

    # TradingView Recommendations
    ss.recommendation_list(ticker)

    # Largest Trade Price
    ss.largest_trade_information(ticker)

else:
    print("Select from 1-3")
    loop = True




# Watchlist News


# Enter ticker for news


# Parse news
