#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 11:58:12 2023

@author: nikhilsama
"""

import pandas as pd
import warnings
warnings.filterwarnings("ignore")
# Code in separate file DatabaseLogin.py to login to kite connect 
from DatabaseLogin import DBBasic
import kite_init as ki 

db = DBBasic() 
kite = ki.initKite()

def download_instruments(exch):
    lst = []
    if exch == 'NSE':
        lst = kite.instruments(exchange=kite.EXCHANGE_NSE)
    else:
        lst = kite.instruments(exchange=kite.EXCHANGE_NFO)
        
    df = pd.DataFrame(lst)
    if len(df) == 0:
        print('No data returned')
        return
    df.drop('exchange_token', inplace=True, axis=1)
    #df.drop('tick_size', inplace=True, axis=1)
    if (exch == 'NFO'):
        df.rename(columns={'name':'underlying_ticker'}, inplace=True)
    else:
        df.drop('name', inplace=True, axis=1)
        df.drop('strike', inplace=True, axis=1)
        df.drop('expiry', inplace=True, axis=1)

    df.set_index('instrument_token', inplace=True)
    df.drop('last_price', inplace=True, axis=1)
    df.drop('segment', inplace=True, axis=1)
    df.drop('exchange', inplace=True, axis=1)
    return df


db.clearInstruments()
df = download_instruments('NSE')
db.toDB('instruments_zerodha',df)
df = download_instruments('NFO')
db.toDB('instruments_zerodha',df)

