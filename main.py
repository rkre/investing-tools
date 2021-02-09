# Top level script that calls on functions within the investing_tools folder

import stock_scanner as ss


# Enter ticker/symbol of the company to see stats and news
print("___________________________")
print("  ")
ticker = input("Enter ticker of a company to see the stats and news: ")
print("____________________________")

ss.latest_iex_news(ticker)
ss.recommendation_list(ticker)
ss.day50MA(ticker)  
ss.day200MA(ticker)
ss.latest_price(ticker)
ss.ceo_compensation(ticker)