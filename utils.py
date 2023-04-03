import sys
import os
import re

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