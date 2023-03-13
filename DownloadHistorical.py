#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 13:36:57 2023

@author: nikhilsama
"""



from datetime import date,timedelta

import datetime as dt
import pandas as pd
import mysql.connector as sqlConnector
import warnings
warnings.filterwarnings("ignore")
from kiteconnect import KiteConnect
# Code in separate file DatabaseLogin.py to login to kite connect 
from DatabaseLogin import DBBasic
import kite_init as ki 
import tickerdata as td
import logging


db = DBBasic() 
kite = ki.initKite()

def zget(from_date, to_date, symbol,interval='minute',continuous=False):
    if from_date > to_date:
        return
    #Kite API doesnt like TZ info in dates
    from_date = from_date.replace(tzinfo=None)
    to_date = to_date.replace(tzinfo=None)
    #print(f"exporting {from_date} to {to_date} for {symbol}")
    token = db.get_instrument_token(symbol)
    if token == -1:
        logging.warning(f'Invalid symbol ({symbol}) provided')
        return 'None'
    
    records = kite.historical_data(token, from_date=from_date, to_date=to_date, 
                                   continuous=continuous, interval=interval)
    df = pd.DataFrame(records)
    if len(df) == 0:
        logging.info('No data returned')
        return
    #df.drop('volume', inplace=True, axis=1)   
    df['symbol'] = symbol
    df.set_index('date',inplace=True)
    return df

def zColsToDbCols(df):
    df.rename(columns = {'open' : 'Open', 'close' : 'Adj Close', 'high': 'High', 'low': 'Low', 'volume' : 'Volume'}, inplace=True)
    return df

def dbColsToZCols(df):
    df.rename(columns = {'Open' : 'open', 'Adj Close' : 'close', 'High': 'high', 'Low': 'low', 'Volume' : 'volume'}, inplace=True)
    return df

def zget_w_db_save(from_date, to_date, symbol,interval='minute',continuous=False):
    df = zget(from_date, to_date, symbol,interval,continuous)
    db.toDB('ohlcv1m',df)
    
            
def zsplit_and_get(from_date, symbol, interval = 'minute', continuous = False):
#    token = db.get_instrument_token(symbol)
    to_date = date.today()
    #data = pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'volume'])
    days = 60
    
    while from_date.date() < date.today():
        if from_date.date() >= (date.today() - timedelta(days)):
            zget(from_date, dt.datetime.now(), symbol,interval,continuous)
            #data = data.append(pd.DataFrame(kite.historical_data(instrument,from_date,dt.date.today(),interval)),ignore_index=True)
            break
        else:
            to_date = from_date + timedelta(days)
            zget(from_date, to_date, symbol,interval,continuous)
            #data = data.append(pd.DataFrame(kite.historical_data(instrument,from_date,to_date,interval)),ignore_index=True)
            from_date = to_date
    #data.set_index("date",inplace=True)
    return 

def getAllNiftyHistoricalData(from_date, interval = 'minute', continuous = False):
    tickers = td.get_nifty_tickers()
    tickers.append('NIFTY 50')

    for t in tickers:
        zsplit_and_get(from_date,t) 
        
        
sdate = '2023-01-15'
edate = '2023-03-12'


#getAllNiftyHistoricalData(dt.datetime.strptime(sdate, '%Y-%m-%d'))
#zsplit_and_get(dt.datetime.strptime(sdate, '%Y-%m-%d'), 'NIFTY23MAR17350CE')
#df = zget(sdate, edate, 'INFY','minute')