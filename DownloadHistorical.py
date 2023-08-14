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
import pytz
import os
import pickle
import utils
import cfg
globals().update(vars(cfg))

# set timezone to IST
ist = pytz.timezone('Asia/Kolkata')

db = DBBasic() 
kite = ki.initKite()


def cache_df(df,t,to):
    #create directory if none exists
    path = "Data/td_cache/"+to.strftime('%d-%m-%y')
    if not os.path.exists(path):
        try: 
            os.mkdir(path)
        except OSError as error:
            print(error)

    path = path+'/'+t
    if not os.path.exists(path):
        try: 
            os.mkdir(path)
        except OSError as error:
            print(error)

    #check if file exists
    file_name=path+"/ohlv-"+to.strftime("%I:%M%p")+".csv"
    df.to_csv(file_name)
    
def loadTickerCache(df,symbol,from_date,to_date,interval):
    path = "Data/ticker_cache/"
    fname = path+symbol+ \
        "__"+from_date.strftime('%d-%m-%y-%H-%M')+"__"+ \
            interval+"__"+to_date.strftime('%d-%m-%y-%H-%M')+".pickle"
    if not os.path.exists(path):
        try: 
            os.mkdir(path)
        except OSError as error:
            print(error)
    with open(fname,"wb") as f:
        pickle.dump(df,f)
        
def getCachedTikerData(symbol,from_date,to_date,interval):
    path = "Data/ticker_cache/"
    fname = path+symbol+ \
        "__"+from_date.strftime('%d-%m-%y-%H-%M')+"__"+ \
            interval+"__"+to_date.strftime('%d-%m-%y-%H-%M')+".pickle"
    if os.path.isfile(fname):
        with open(fname, "rb") as f:
            df = pickle.load(f)
    else:
        df = pd.DataFrame()
    return df

def zget_basic(from_date, to_date, symbol,interval='minute',
         continuous=False,token=None):
    if from_date > to_date:
        logging.warning(f'start {from_date} is greater than end {to_date}')
        return pd.DataFrame()
    
    #Kite API doesnt like TZ info in dates
    from_date = from_date.replace(tzinfo=None)
    to_date = to_date.replace(tzinfo=None)
    #print(f"exporting {from_date} to {to_date} for {symbol}")
    if cacheTickData:
        df = getCachedTikerData(symbol,from_date,
                              to_date,interval) 
        if not df.empty:
            return df
        
    if token is None:
        token = db.get_instrument_token(symbol)
    if token == -1:
        logging.warning(f'Invalid symbol ({symbol}) provided')
        return pd.DataFrame()
    try:
        records = kite.historical_data(token, from_date=from_date, to_date=to_date, 
                                       continuous=continuous, interval=interval)
        print(f"exporting {from_date} to {to_date} for {symbol} continuous:{continuous} interval:{interval}")
    except Exception as e:
        logging.error(f'Get Historical Data Failed T: {token} from: {from_date} to: {to_date} continueous:{continuous} interval:{interval} FAILED.')
        logging.error(e.args[0])
        return pd.DataFrame()

    df = pd.DataFrame(records)
    if df.empty:
        logging.info(f'No data returned for {symbol} from: {from_date} to: {to_date} continueous:{continuous} interval:{interval}')
        return df
    df = zColsToDbCols(df)
    # Adding new index column
    df.insert(0, 'i', range(1, 1 + len(df)))

    #Kite Historical timestamp for a candle contails ohlcv data for the 
    #minute that "STARTED" at the timestamp.
    #
    #Last row of the dataframe contains a "live" incomplete candle for the minute
    # that STARTED now
    #We dont want to make any signal decisions based on this live / incomplete
    #view, since as the candle complete, ohlcv data cahnges, that signal may
    #change.  Therefore we remove this last row of data
    # df.drop(df.tail(1).index,inplace=True) # drop last row

    df['symbol'] = symbol
    df.set_index('date',inplace=True)

    df = utils.cleanDF(df,interval)

    # Remove Volume data for options (not relavent)
    if utils.isOption(symbol) and (not cfgUseVolumeDataForOptions):
        df['Volume'] = 0

    if cacheTickData:
        loadTickerCache(df,symbol,from_date,to_date,interval)
    return df

def zAddOptionsData(df,symbol,from_date,to_date,interval='minute',continuous=False):
    # Add options data
    (put_option_ticker,lot_size,tick_size,strike_p) =db.get_option_ticker(symbol,df['Adj Close'].iloc[-1], 'PE',kite,0)
    p_df = zget_basic(from_date,to_date,put_option_ticker,interval,continuous)
    (call_option_ticker,lot_size,tick_size,strike_c) =db.get_option_ticker(symbol,df['Adj Close'].iloc[-1], 'CE',kite,0)
    c_df = zget_basic(from_date,to_date,call_option_ticker,interval,continuous)

    
    # forward fill 0 and NaN volume values
    p_df['Volume'] = p_df['Volume'].replace(0, pd.np.nan).fillna(method='ffill')
    c_df['Volume'] = c_df['Volume'].replace(0, pd.np.nan).fillna(method='ffill')

    df['Open-P'] = p_df['Open']
    df['High-P'] = p_df['High']
    df['Low-P'] = p_df['Low']
    df['Adj Close-P'] = p_df['Adj Close']
    df['Volume-P'] = p_df['Volume']
    df['Strike-P'] = strike_p
    # df[['Open-P', 'High-P', 'Low-P', 'Adj Close-P', 'Volume-P']] = \
    #     df[['Open-P', 'High-P', 'Low-P', 'Adj Close-P', 'Volume-P']].ffill().bfill()

    df['Open-C'] = c_df['Open']
    df['High-C'] = c_df['High']
    df['Low-C'] = c_df['Low']
    df['Adj Close-C'] = c_df['Adj Close']
    df['Volume-C'] = c_df['Volume']
    df['Strike-C'] = strike_c

    # df[['Open-C', 'High-C', 'Low-C', 'Adj Close-C', 'Volume-C']] = \
    #     df[['Open-C', 'High-C', 'Low-C', 'Adj Close-C', 'Volume-C']].ffill().bfill()
    return df

def zGetSurrogateVolumeFromFuture(symbol,from_date,to_date,interval='minute',continuous=False):
    (future_ticker,lot_size,tick_size) =db.get_future_ticker(symbol)
    futDF = zget_basic(from_date,to_date,future_ticker,interval,continuous)
    
    
    
def zget(from_date, to_date, symbol,interval='minute',
         includeOptions=False, continuous=True, instrumentToken=None):
    if utils.tickerIsFuture(symbol) and interval=='day':
        continuous = False # set continuous True for futures//Correction, contiinous only works for day 
        # candles not for minute candles
        
    df = zget_basic(from_date, to_date, symbol,interval,False,instrumentToken)

    if df.empty:
        return df
    # if (df['Volume'].iloc[-1] == 0):
    #     # Likely an Index Ticker, use surrogate volume from its future
    #     df['Volume'] = zGetSurrogateVolumeFromFuture(symbol,from_date,to_date,interval,continuous)
    if (includeOptions):
        df = zAddOptionsData(df,symbol,from_date,to_date,interval,continuous)
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
    to_date = dt.datetime.now(ist)
    #data = pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'volume'])
    days = 60
    
    while from_date.date() < date.today():
        if from_date.date() >= (date.today() - timedelta(days)):
            zget_w_db_save(from_date, dt.datetime.now(ist), symbol,interval,continuous)
            #data = data.append(pd.DataFrame(kite.historical_data(instrument,from_date,dt.date.today(),interval)),ignore_index=True)
            break
        else:
            to_date = from_date + timedelta(days)
            zget_w_db_save(from_date, to_date, symbol,interval,continuous)
            #data = data.append(pd.DataFrame(kite.historical_data(instrument,from_date,to_date,interval)),ignore_index=True)
            from_date = to_date
    #data.set_index("date",inplace=True)
    return 

def getAllNiftyHistoricalData(from_date, interval = 'minute', continuous = False):
    tickers = td.get_nifty_tickers()
    tickers.append('NIFTY 50')

    for t in tickers:
        zsplit_and_get(from_date,t) 

def dbUpdatetoCurrent():
    end = dt.datetime.now(ist)
    start = db.next_tick(end)
    
    getAllNiftyHistoricalData((start))
    
#dbUpdatetoCurrent()
        
        
sdate = '2023-03-13'
edate = '2023-03-14'


#getAllNiftyHistoricalData(dt.datetime.strptime(sdate, '%Y-%m-%d'))
#zsplit_and_get(dt.datetime.strptime(sdate, '%Y-%m-%d'), 'NIFTY23MAR17350CE')
#df = zget(sdate, edate, 'INFY','minute')