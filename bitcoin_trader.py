# Trades bitcoin on Robinhood based on TradingView indicator recommendations

from tradingview_data import btc_recommendation
from robinhood_trader import make_trade

# Check for changes in recommendations

print("This script checks for changes in recommendation from TradingView indicators for Bitcoin and trades .25 BTC")
amount = .25
ticker = 'btc'

init_recommendation = btc_recommendation
recommendation = btc_recommendation()

while init_recommendation == recommendation:
    # Update recommendation
    recommendation = btc_recommendation()
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

    
