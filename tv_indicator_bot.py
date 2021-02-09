from tradingview_data import tradingview_recommendation
from robinhood_trader import make_trade

# Check for changes in recommendations

print("This script checks for changes in recommendation from TradingView indicators")
ticker = input("Enter the ticker to scan for recommendation changes: ")
amount = 5

init_recommendation = tradingview_recommendation(ticker)
recommendation = tradingview_recommendation(ticker)

while init_recommendation == recommendation:
    # Update recommendation
    recommendation = tradingview_recommendation(ticker)
    if init_recommendation != recommendation:
        print("Recommendation changed to: ", recommendation)

        if recommendation == 'SELL':
            make_trade('sell', amount, ticker)
        elif recommendation == 'BUY':
            make_trade('buy', amount, ticker)

    elif recommendation == 'BUY':
        print("Recommend buy at price: ")
    elif recommendation == 'SELL':
        print("Recommend sell at price: ")

    
