import sys
import os
import re
import pandas as pd
import numpy as np

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
def optionTypeFromTicker(t):
    pattern = r'^[A-Z]+\d+(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)?\d+(?:PE|CE)$'
    match = re.match(pattern, t)
    if match:
        return t[-2:]
    else:
        return False
def convertPEtoCEAndViceVersa(t):
    if optionTypeFromTicker(t):     
        return t[:-2] + ('CE' if t[-2] == 'P' else 'PE')
    else:
        return t
