#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 14:14:12 2023

@author: nikhilsama
"""

#!python

#cfg has all the config parameters make them all globals here

import log_setup
import logging
import kite_init as ki 
import pytz
import time
import datetime
from time import time, ctime, sleep
import pandas as pd

import time_loop as tl
import DownloadHistorical as downloader
import tickerdata as td
from DatabaseLogin import DBBasic

import cfg
globals().update(vars(cfg))


# set timezone to IST
ist = pytz.timezone('Asia/Kolkata')

db = DBBasic() 

# Initialise
kite, kws = ki.initKiteTicker()
buy_order_id,sell_order_id = 0,0
tickersToTrack = {}
now = datetime.datetime.now(ist)


def getTickersToTrack():
    tickers = td.get_fo_active_nifty_tickers()
    for t in tickers:
        token = \
            db.get_instrument_token(t)
        tickersToTrack[token] = {
                "ticker": t,
                'df': pd.DataFrame(),
                'ticks': pd.DataFrame()
            }
        
        #Create an empty DataFrame with column names and index
        # Initialize the tick DF, so we can assign an index to it
        columns = ['Open', 'High', 'Low', 'Adj Close', 'Volume']
        index = pd.date_range('2023-01-01', periods=0, freq='D')
        tickersToTrack[t]['df'] = pd.DataFrame(columns=columns, index=index)

def trimMinuteDF(t):
    #trim the minute df to last 375 minutes
    tickersToTrack[t]['df'] = tickersToTrack[t]['df'].iloc[-375:]

def getHistoricalTickerData():
    #This code is intended to run before start of trading on day of
    #
    # We want last day of data only; but because we could start on 
    # monday morning, we request 3 days of data, and then truncate
    # the df to last 375 minute rows (i.e. last day 9:15 to 3:30)
    global tickerData, tickersToTrack, now
    start = now - datetime.timedelta(days=5)
    for t in tickersToTrack.keys():
        tickersToTrack[t]['df']= downloader.zget \
        (start,now,tickersToTrack[t]['ticker'],'minute',
         includeOptions=includeOptions,instrumentToken=t)
        trimMinuteDF(t)
    return
def subscribeToTickerData():
    tokenList = list(tickersToTrack.keys())        
    kws.subscribe(tokenList)
    kws.set_mode(kws.MODE_FULL, tokenList)

def addTicksToTickDF(ticks):
    #add the tick to the tick df
    for tick in ticks:
        token = tick['instrument_token']
        
        #Insert this tick into the tick df
        tick_time = tick['timestamp']
        tick_df = tickersToTrack[token]['ticks']
        tick_df_row = {
            'Open': tick['last_price'],
            'High': tick['last_price'],
            'Low': tick['last_price'],
            'Adj Close': tick['last_price'],
            'Volume': tick['last_quantity']
        }
        tick_df = tickersToTrack[token]['ticks']
        tick_df.loc[tick_time] = tick_df_row    
        
def resampleToMinDF():
    resampled_tokens = []
    #process ticks to create minute candles
    for token in tickersToTrack.keys():
        #Create a new minute candle if last  tick was more 
        #than 1 minute since last candle 
        tick_df = tickersToTrack[token]['ticks']
        minute_candle_df = tickersToTrack[token]['df']
        
        timedelta = tick_df[-1] - minute_candle_df.index[-1]
        
        if timedelta.seconds >= 60:
            # Get the last round minute
            this_minute = pd.Timestamp(tick_df[-1].floor('min'))

            # Create a new index that ends at the last round minute
            # Get rows in the DataFrame before the target time
            ticks_upto_this_minute = tick_df.loc[df.index <= this_minute]
            ticks_after_this_minute = tick_df.loc[df.index > this_minute]

            resampled_ticks_upto_this_minute = \
            ticks_upto_this_minute.resample('1min').agg({
              'Open': 'first',
              'Close': 'last',
              'High': 'max',
              'Low': 'min',
              'Adj Close': 'last',
              'Volume': 'sum'  
            })
            #Add in symbol and index to make it at par with the Historical data candles
            resampled_ticks_upto_this_minute['symbol']=tickersToTrack[token]['ticker']
            resampled_ticks_upto_this_minute.insert(0, 'i', 
                    value=pd.Series([minute_candle_df['i'][-1]+1+i for i in range(len(df))]))

            #For now drop all minute_candle df's other than OCHLV and i and symbol
            #Analytics will recreate them, we dont want to confure analytics
            #TODO: Fix this, figure out how to keep that data and just do analytics on the 
            #new minute candle; this will make the tick much faster
            
            keep_columns = ['Open', 'High', 'Low', 'Adj Close', 'Volume','symbol','i']
            drop_columns = list(set(minute_candle_df.columns) - set(keep_columns))
            minute_candle_df.drop(columns=drop_columns, inplace=True)

            # Append the new minute candle rows to the minute candle df
            minute_candle_df = pd.concat([minute_candle_df,
                                          resampled_ticks_upto_this_minute],
                                         axis=0)
            #trip to 375 rows/minutes
            trimMinuteDF(token)

            # Remove the ticks that have been used to create the new minute candle
            tickersToTrack[token]['ticks'] = ticks_after_this_minute
            resampled_tokens.append(token)
    return resampled_tokens 

def tick():
    positions = ki.get_positions(kite)
    for token in tickersToTrack.key():
        tl.tigenerateSignalsAndTradeck(tickersToTrack[token]['df'],positions,True,False)
    
def processTicks(ticks):
    #add the tick to the tick df
    addTicksToTickDF(ticks)
    resampled_tokens = resampleToMinDF()
    tick()
                
####### KITE TICKER CALLBACKS #######
def on_ticks(ws, ticks):
    # Callback to receive ticks.
    print("Ticks: {}".format(ticks))
    logging.debug("Ticks: {}".format(ticks))
    
    processTicks(ticks)
    bid = ticks[0]['depth']['buy'][0]['price']
    ask = ticks[0]['depth']['sell'][0]['price']
    
    print(ctime(time()),">>>Bid=",bid," ASK=",ask, " Last Trade: ", ticks[0]["last_trade_time"])


def on_connect(ws, response):
    print("connect called {}".format(response))
    getTickersToTrack()
    getHistoricalTickerData()
    subscribeToTickerData()

def on_close(ws, code, reason):
    print(f"Close called Code: {code}  Reason:{reason}")

    # On connection close stop the event loop.
    # Reconnection will not happen after executing `ws.stop()`
    ws.stop()
##########END KITE TICKER CALLBACKS ##########

########## KITE TICKER CONFIG AND START ##########
# Assign the callbacks.
kws.on_ticks = on_ticks
kws.on_connect = on_connect
kws.on_close = on_close

# Infinite loop on the main thread. Nothing after this will run.
# You have to use the pre-defined callbacks to manage subscriptions.
#kws.connect()
########## END KITE TICKER CONFIG AND START ##########


## SAMPLE TICK FORMAT
# [{
#     'instrument_token': 53490439,
#     'mode': 'full',
#     'volume': 12510,
#     'last_price': 4084.0,
#     'average_price': 4086.55,
#     'last_quantity': 1,
#     'buy_quantity': 2356
#     'sell_quantity': 2440,
#     'change': 0.46740467404674046,
#     'last_trade_time': datetime.datetime(2018, 1, 15, 13, 16, 54),
#     'timestamp': datetime.datetime(2018, 1, 15, 13, 16, 56),
#     'oi': 21845,
#     'oi_day_low': 0,
#     'oi_day_high': 0,
#     'ohlc': {
#         'high': 4093.0,
#         'close': 4065.0,
#         'open': 4088.0,
#         'low': 4080.0
#     },
#     'tradable': True,
#     'depth': {
#         'sell': [{
#             'price': 4085.0,
#             'orders': 1048576,
#             'quantity': 43
#         }, {
#             'price': 4086.0,
#             'orders': 2752512,
#             'quantity': 134
#         }, {
#             'price': 4087.0,
#             'orders': 1703936,
#             'quantity': 133
#         }, {
#             'price': 4088.0,
#             'orders': 1376256,
#             'quantity': 70
#         }, {
#             'price': 4089.0,
#             'orders': 1048576,
#             'quantity': 46
#         }],
#         'buy': [{
#             'price': 4084.0,
#             'orders': 589824,
#             'quantity': 53
#         }, {
#             'price': 4083.0,
#             'orders': 1245184,
#             'quantity': 145
#         }, {
#             'price': 4082.0,
#             'orders': 1114112,
#             'quantity': 63
#         }, {
#             'price': 4081.0,
#             'orders': 1835008,
#             'quantity': 69
#         }, {
#             'price': 4080.0,
#             'orders': 2752512,
#             'quantity': 89
#         }]
#     }
# },
# ...,
# ...]