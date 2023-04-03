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
import random 
from freezegun import freeze_time   
import math
import threading

import time_loop as tl
import DownloadHistorical as downloader
import tickerdata as td
from DatabaseLogin import DBBasic

import cfg
globals().update(vars(cfg))

global tickThread
tickThread = None
tickThreadBacklog = []

## TEST FREEZESR START

if cfgFreezeGun:
    from freezegun import freeze_time
    freezeTime = "Mar 28nd, 2023 10:00:00+0553"
    freezer = freeze_time(freezeTime, tick=True)
    freezer.start()
    print ("Freeze gun is on. Time is frozen at: ", freezeTime)
    ## END 

# set timezone to IST

# set timezone to IST
ist = pytz.timezone('Asia/Kolkata')

db = DBBasic() 

# Initialise
kws = ki.initKws(tl.get_kite_access_token())
buy_order_id,sell_order_id = 0,0
tickersToTrack = {}
now = datetime.datetime.now(ist)

def tickerlog(s):
    logtime = datetime.datetime.now(ist).strftime("%I:%M:%S %p")
    s = f'{logtime} {s}'

    tlogfile = f"Data/logs/{datetime.datetime.now().strftime('%d-%m-%y')}.tickerlog"
    with open(tlogfile, 'a') as f:
        f.write(s)
        f.write('\n')


def getTickersToTrack():
    tickers = td.get_fo_active_nifty_tickers()
    #tickers = ['RELIANCE']

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
        tickersToTrack[token]['df'] = pd.DataFrame(columns=columns, index=index)
        tickersToTrack[token]['ticks'] = pd.DataFrame(columns=columns, index=index)

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
         includeOptions=False,instrumentToken=t)
        trimMinuteDF(t)
    return
def subscribeToTickerData():
    if cfgFreezeGun: # not live overloaded
        return
    tokenList = list(tickersToTrack.keys())   
    tokenList = [int(x) for x in tokenList]  
    tickerlog(f"Subscribing to {len(tokenList)} tokens: {tokenList}")   
    kws.subscribe(tokenList)
    kws.set_mode(kws.MODE_FULL, tokenList)

def addTicksToTickDF(ticks):
    #add the tick to the tick df
    for tick in ticks:
        token = tick['instrument_token']
        
        #Insert this tick into the tick df
        tick_time = tick['exchange_timestamp']
        tick_time = ist.localize(tick_time)
        tick_df_row = {
            'Open': tick['last_price'],
            'High': tick['last_price'],
            'Low': tick['last_price'],
            'Adj Close': tick['last_price'],
            'Volume': tick['volume_traded']
        }
        tickersToTrack[token]['ticks'].loc[tick_time] = tick_df_row    
        
def resampleToMinDF():
    resampled_tokens = []
    #process ticks to create minute candles
    for token in tickersToTrack.keys():
        #Create a new minute candle if last  tick was more 
        #than 1 minute since last candle 
        tick_df = tickersToTrack[token]['ticks']
        minute_candle_df = tickersToTrack[token]['df']
        
        if tick_df.empty:
            tickerlog(f"Tick DF is empty for token {token}. {tick_df.empty} {minute_candle_df.empty}")
            continue # no ticks yet, so no resample
        
        if minute_candle_df.empty:
            #tickerlog(f"First tick, minute candle is emply for {token}")
             # First tick will populate minute candle w Historic data
            timedelta = datetime.timedelta(seconds=120)
        else:
            timedelta = tick_df.index[-1] - minute_candle_df.index[-1]
        
        if timedelta.seconds >= 120: # last tick of min candle includes data till min+1, we need to resample when another whole min is avail so 120s
            #tickerlog(f"Resampling token {token} to minute candle. Last tick was {timedelta.seconds} seconds ago")
            # Get the last round minute
            this_minute = pd.Timestamp(tick_df.index[-1].floor('min'))

            # Create a new index that ends at the last round minute
            # Get rows in the DataFrame before the target time
            ticks_upto_this_minute = tick_df.loc[tick_df.index < this_minute]
            ticks_after_this_minute = tick_df.loc[tick_df.index >= this_minute]
            #print(f"ticks_upto_this_minute: {ticks_upto_this_minute}")
            #print(f"ticks_after_this_minute: {ticks_after_this_minute}")
            resampled_ticks_upto_this_minute = \
            ticks_upto_this_minute.resample('1min').agg({
              'Open': 'first',
              'High': 'max',
              'Low': 'min',
              'Adj Close': 'last',
              'Volume': lambda x: x[-1] - x[0] #'sum' volume data in ticks is cumulative  
            })

            #For now drop all minute_candle df's other than OCHLV and i and symbol
            #Analytics will recreate them, we dont want to confure analytics
            #TODO: Fix this, figure out how to keep that data and just do analytics on the 
            #new minute candle; this will make the tick much faster
            
            keep_columns = ['Open', 'High', 'Low', 'Adj Close', 'Volume','symbol','i']
            drop_columns = list(set(minute_candle_df.columns) - set(keep_columns))
            minute_candle_df.drop(columns=drop_columns, inplace=True)
            
            if minute_candle_df.empty:
                tickerlog(f"Min candle emply. First call for {token}. Getting historical, and ignoring this first half formed tick candle")
                historicalEnd = tick_df.index[-1] #downloader will remove lst min half formed candle
                historicalStart = historicalEnd - datetime.timedelta(days=5)
                tickersToTrack[token]['df']= downloader.zget \
                    (historicalStart,historicalEnd,tickersToTrack[token]['ticker'],'minute',
                    includeOptions=False,instrumentToken=token)
            else:
                # Append the new minute candle rows to the minute candle df
                            #Add in symbol and index to make it at par with the Historical data candles
                resampled_ticks_upto_this_minute['symbol']=tickersToTrack[token]['ticker']
                resampled_ticks_upto_this_minute.insert(0, 'i', 
                        range(minute_candle_df['i'][-1]+1, minute_candle_df['i'][-1]+1 + len(resampled_ticks_upto_this_minute)))
                tickersToTrack[token]['df'] = pd.concat([minute_candle_df,
                                                resampled_ticks_upto_this_minute],
                                                axis=0)
                tickerlog(f"Adding resampled rows {resampled_ticks_upto_this_minute}  to minute candle {tickersToTrack[token]['df'].tail()}")

            #trip to 375 rows/minutes
            trimMinuteDF(token)
            # Remove the ticks that have been used to create the new minute candle
            tickersToTrack[token]['ticks'] = ticks_after_this_minute
            resampled_tokens.append(token)
    return resampled_tokens 

def tick(tokens):
    positions = tl.get_positions()
    for token in tokens:
        tickerlog(f"tickThread generating signals for: token {token} {tickersToTrack[token]['ticker']}")
        tl.generateSignalsAndTrade(tickersToTrack[token]['df'].copy(),positions,False,True)

def processTicks(ticks):
    #add the tick to the tick df
    global tickThreadBacklog
    addTicksToTickDF(ticks)
    resampled_tokens = resampleToMinDF()
    
    if (len(resampled_tokens) == 0) and (len(tickThreadBacklog) == 0):
        #tickerlog("No tokens to process and no backlog. Skipping this tick")
        return
    
    tickThreadBacklog.extend(resampled_tokens)

    #tick(resampled_tokens)
    #Kite Ticker will close connection if on_ticks takes too
    #long to process. Therefore we need to do analytics
    #and trading in a seperate thread
    global tickThread
    if ((tickThread is not None) \
            and (tickThread.is_alive())):
        tickerlog(f"Tick thread is alive. Skipping this Tick. Backlog: {tickThreadBacklog}")
    else:
        tickerlog(f"Tick thread is done. Joining and restarting Backlog: {tickThreadBacklog}")
        if tickThread is not None:
            tickThread.join()
            tickerlog("tickThread Joined")

        tickThreadBacklog = list(set(tickThreadBacklog)) #remove duplicates due to multi-min runs (should not happen, but just in case)
        tickThread = threading.Thread(target=tick, 
                    args=(tickThreadBacklog.copy(),)) # funky syntax needs (t,) if only one arg to signify its a tuple
        tickThread.start()
        tickThreadBacklog = []

           
####### KITE TICKER CALLBACKS #######
def on_ticks(ws, ticks):
    # Callback to receive ticks.
    #tickerlog("Ticks:")    
    processTicks(ticks)
    # bid = ticks[0]['depth']['buy'][0]['price']
    # ask = ticks[0]['depth']['sell'][0]['price']
    
    # print(ctime(time()),">>>Bid=",bid," ASK=",ask, " Last Trade: ", ticks[0]["last_trade_time"])


def on_connect(ws, response):
    tickerlog("connect called {}".format(response))
    getTickersToTrack()
    #Don't get data now, get it when we start getting ticks
    #That way minutedf is fully updated until ticks take
    #over. Otherwise, there can be a gap or half formed
    #candle in the minute df
   #getHistoricalTickerData()
    subscribeToTickerData()
    #tick(tickersToTrack.keys()) #First tick with historical data

def on_close(ws, code, reason):
    tickerlog(f"Close called Code: {code}  Reason:{reason}")

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

kws.connect()
########## END KITE TICKER CONFIG AND START ##########

# def testTicks():
#     sampleTicks = []
#     volume = 12510
#     startPrice = 2330.0
#     second = 0
#     minute = 0
#     hour = 10
#     for i in range(0,10):
#         if i%2 == 0: 
#             token = 408065#INFY
#         else:
#             token = 738561#RELIANCE
#         if i == 9: 
#             second = 0
#             ms = 1
#             minute = minute +2
#             if (minute == 60): 
#                 minute = 0
#                 hour = hour+1
#         else:
#             second = second+(15*random.random())
#             if second >=60:
#                 second = second - 60
#                 minute= minute+1
#             ms, second = math.modf(second)
#             second = int(second)
#             ms = int(round(ms*1000))
#         volume = volume + random.randint(0,100)

#         sampleTick = {
#             'instrument_token': token,#RELIANCE
#             'volume':volume,
#             'last_price': startPrice + random.randint(-10,10), 
#             'timestamp': datetime.datetime(2023, 3, 28, hour, minute, second, microsecond=ms,tzinfo=ist),
#         }
#         sampleTicks.append(sampleTick)
        
#     on_ticks('trash',sampleTicks)

# on_connect('trash','trash')

# print(tickersToTrack[738561]['df'].tail())
# print(tickersToTrack[408065]['df'].tail())

# testTicks()

# print(tickersToTrack[738561]['ticks'])
# print(tickersToTrack[738561]['df'].tail())

# print(tickersToTrack[408065]['ticks'])
# print(tickersToTrack[408065]['df'].tail())