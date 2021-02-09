
from tradingview_ta import TA_Handler, Interval, Exchange

def tradingview_recommendation(ticker): 
    "There are 3 types of analysis in TradingView: "
    "oscillators, moving averages, and summary (which is oscillators and moving averages combined)."

# handler = TA_Handler()
# handler.set_symbol_as("TSLA")
# handler.set_exchange_as_crypto_or_stock("NASDAQ")
# handler.set_screener_as_stock("america")
# handler.set_interval_as(Interval.INTERVAL_15_MINUTES)

    stock_ticker = ticker

    stock_ticker = TA_Handler(
        symbol=f"{ticker}",
        screener="america",
        exchange="NASDAQ",
        interval=Interval.INTERVAL_1_DAY
    )

    print(f"Reommendation for {ticker} : \n", stock_ticker.get_analysis().summary)
    summary = stock_ticker.get_analysis().summary
    recommendation = summary["RECOMMENDATION"]
    print("Recommendation: ", recommendation)
    return recommendation
    # Example output: {"RECOMMENDATION": "BUY", "BUY": 8, "NEUTRAL": 6, "SELL": 3}

def btc_recommendation():
    "Bitcoin Recommendation"
    # Bitcoin / USD Tether
    btc = TA_Handler(
        symbol="BTCUSDT",
        screener="crypto",
        exchange="binance",
        interval=Interval.INTERVAL_1_DAY
    )

    print("Bitcoin Recommendation: \n", btc.get_analysis().summary)
    summary = btc.get_analysis().summary
    btc_recommendation = summary["RECOMMENDATION"]
    print("Recommendation: ", btc_recommendation)
    return btc_recommendation
