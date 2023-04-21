import sys
import os
import re
import pandas as pd
import numpy as np
from datetime import datetime
import time
import tickerCfg

#cfg has all the config parameters make them all globals here
import cfg
globals().update(vars(cfg))

    
def timeToString(t,date=False,time=True):
    if date and time:
        return t.strftime("%d-%m %I:%M %p")
    elif date:
        return t.strftime("%d-%m")
    else:
        return t.strftime("%I:%M:%S %p")

def fileNameFromArgs(prefix=''):
    args = sys.argv[1:]
    fname = prefix
    for arg in args:
        key, value = arg.split(':')
        if key in ['zerodha_access_token','dbuser','dbpass','cacheTickData', 'dbname', 'dbhost']:
            continue
        fname = fname + '-' + value
    fname = fname + '.csv'
    return fname
def fileExists(fname):
    return os.path.isfile(fname)
def tickerIsFuture(t):
    return re.match(r'^[a-zA-Z]+\d+[a-zA-Z]{3}FUT$', t)
def getUnderlyingTickerForFuture(t):
    match = re.match(r'^([a-zA-Z]+)\d+[a-zA-Z]{3}FUT$', t)
    if match:
        uderlying_ticker = match.group(1)
        return uderlying_ticker
    else:
        return None
def explodeOptionTicker(t):
    if t.endswith(".NFO"):
        t = t[:-4]
    pattern = r'^([A-Z]+)\d+(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)?\d+((?:PE|CE))$'
    match = re.match(pattern, t)
    if match:
        return (match.group(1), match.group(2))
    else:
        return False
def optionTypeFromTicker(t):
    if explodeOptionTicker(t):
        return explodeOptionTicker(t)[1]
def optionUnderlyingFromTicker(t):
    if explodeOptionTicker(t):
        return explodeOptionTicker(t)[0]
    else:
        return t
def isPutOption(t):
    return optionTypeFromTicker(t) == 'PE'
def isCallOption(t):
    return optionTypeFromTicker(t) == 'CE'
def isOption(t):
    return isPutOption(t) or isCallOption(t)
def isNotAnOption(t):
    return not isOption(t)
def convertPEtoCEAndViceVersa(t):
    if optionTypeFromTicker(t):     
        return t[:-2] + ('CE' if t[-2] == 'P' else 'PE')
    else:
        return t
def getTickerCfg(ticker):
    if explodeOptionTicker(ticker):
        ticker = explodeOptionTicker(ticker)[0]
    else:
        futTicker = getUnderlyingTickerForFuture(ticker)
        if futTicker is not None:
            ticker = futTicker
    if ticker in tickerCfg.tickerCfg:
        print(f"Applying CFG for {ticker}")
        return tickerCfg.tickerCfg[ticker]
    else:
        print(f"No CFG for {ticker}. Reverting to default CFG")
        return cfgDict
    
    
def getNSEHolidays():
    nse_holidays = [
        "January 26, 2022", "March 01, 2022", "March 18, 2022", "April 14, 2022", "April 15, 2022",
        "May 03, 2022", "August 09, 2022", "August 15, 2022", "August 31, 2022", "October 05, 2022", 
        "October 24, 2022", "October 26, 2022", "November 08, 2022",
        "26-Jan-2023","07-Mar-2023","30-Mar-2023","04-Apr-2023","07-Apr-2023","14-Apr-2023",
        "01-May-2023","28-Jun-2023", "15-Aug-2023", "19-Sep-2023", "02-Oct-2023", "24-Oct-2023",
        "14-Nov-2023", "27-Nov-2023", "25-Dec-2023"
    ]
    nse_holidays = pd.to_datetime(nse_holidays)
    nse_holidays = [h.date() for h in nse_holidays]
    return nse_holidays

def cleanDF(df):
    # Kite can sometimes return junk data before 915 or 1530, wich very 
    # low or zero volume.  These set the min/max values for OBV and 
    # affect our analytics and signals for a long time.  So we filter
    # fileter out these junk values

    df = df.between_time('09:15:00+05:30', '15:29:00+05:30')    
    df = df[df.index.weekday<5] # remove weekends
    
    # Remove holiday data from the DataFrame
    df = df[~df.index.isin(getNSEHolidays())]
    return df

def isTradingDay(date):
    is_weekend = date.weekday() >= 5
    is_holiday = date.date() in getNSEHolidays()
    return not (is_weekend or is_holiday)