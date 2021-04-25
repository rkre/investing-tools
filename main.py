# Top level script that calls on functions within the investing_tools folder

import stock_scanner as ss
import price_target_scanner as pts
import options_scanner as ops

# Enter ticker/symbol of the company to see stats and news
print("___________________________")
print("  ")
ticker = input("Enter ticker of a company to see the stats and news: ")
print("____________________________")


ss.latest_iex_news(ticker)
ss.recommendation_list(ticker)
# ss.day50MA(ticker)  
# ss.day200MA(ticker)
#ops.
ss.ceo_compensation(ticker)
ss.company_financials(ticker)
ss.employee_information(ticker)
ss.short_seller_checker(ticker)
# ss.stock_highlow_price(ticker)
# ss.stock_percent_change(ticker)
ss.stock_ratios(ticker)
pts.price_target_by_volume(ticker)
ss.bid_ask_data(ticker)
ss.latest_price(ticker)