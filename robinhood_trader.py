# Trading bot for Robinhood

import pyotp
import robin_stocks as rh
from secrets import RH_MFA_CODE, RH_PASSWORD, RH_USER_EMAIL


def trade_stock(action, amount, ticker):
    "Logs into Robinhood with MFA/2FA and buys or sells"
    totp = pyotp.TOTP(RH_MFA_CODE).now()
    # print("Current OTP:", totp)
    login = rh.login(RH_USER_EMAIL, RH_PASSWORD, mfa_code=totp)

    if action == 'buy' or action == '1':
        #rh.order_buy_market(ticker, amount)
        print(f"Bought {amount} shares of {ticker}!\n")
    elif action == 'sell' or action == '2':
        #rh.order_sell_market(ticker, amount)
        print(f"Sold {amount} shares of {ticker}!\n")
    else:
        print("Error?")

def trade_btc(dollar_amount):
    "Trade bitcoin"
    #Sell half a Bitcoin is price reaches 10,000
    rh.order_sell_crypto_limit('BTC',0.5,10000)
    #Buy $500 worth of Bitcoin
    rh.order_buy_crypto_by_price('BTC',500)
    
