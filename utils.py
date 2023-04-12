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
        
    if ticker in tickerCfg.tickerCfg:
        print(f"Applying CFG for {ticker}")
        return tickerCfg.tickerCfg[ticker]
    else:
        print(f"No CFG for {ticker}. Reverting to default CFG")
        return cfgDict
