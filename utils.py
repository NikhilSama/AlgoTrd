import sys
import os

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