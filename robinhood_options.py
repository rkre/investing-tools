# Displays option chain information from Robinhood

import pyotp
import robin_stocks as rh
from secrets import RH_MFA_CODE, RH_PASSWORD, RH_USER_EMAIL


def option_strike_prices(action, amount, ticker):
    "Logs into Robinhood with MFA/2FA and buys or sells"
    totp = pyotp.TOTP(RH_MFA_CODE).now()
    # print("Current OTP:", totp)
    login = rh.login(RH_USER_EMAIL, RH_PASSWORD, mfa_code=totp)

    option_data = robin_stocks.robinhood.options.find_options_by_expiration_and_strike(inputSymbols, expirationDate, strikePrice, optionType=None, info=None)

optionData = robin_stocks.find_options_for_list_of_stocks_by_expiration_date(['fb','aapl','tsla','nflx'],
>>>              expirationDate='2018-11-16',optionType='call')