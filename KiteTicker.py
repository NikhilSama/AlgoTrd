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
import numpy as np
import random 
from freezegun import freeze_time   
import math
import threading
import signals 
import utils
import subprocess

import time_loop as tl
import DownloadHistorical as downloader
import tickerdata as td
from DatabaseLogin import DBBasic
import sendemail as email

import cfg
globals().update(vars(cfg))

global tickThread
tickThread = None
tickThreadBacklog = []
nifty_ltp = 0

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
tradingStartTime = now.replace(hour=cfgStartTimeOfDay.hour, minute=cfgStartTimeOfDay.minute, 
                               second=0, microsecond=0)

def getNiftyLTP():
    return nifty_ltp

def ltp(t=None,ticker=None):
    if t is None:
        t = db.get_instrument_token(ticker)
    if (t in tickersToTrack.keys()) and (len(tickersToTrack[t]['ticks'])>0):
        return tickersToTrack[t]['ticks']['Adj Close'].iloc[-1]
    else:
        return False 
    
def tickerlog(s):
    logtime = datetime.datetime.now(ist).strftime("%I:%M:%S %p")
    s = f'{logtime} {s}'

    tlogfile = f"Data/logs/{datetime.datetime.now().strftime('%d-%m-%y')}.tickerlog"
    with open(tlogfile, 'a') as f:
        f.write(s)
        f.write('\n')


def getTickersToTrack():
    tickers = td.get_fo_active_nifty_tickers(offset=100)
    #tickers = ['RELIANCE']

    for t in tickers:
        token = \
            db.get_instrument_token(t)
        tickersToTrack[token] = {
                "ticker": t,
                'df': pd.DataFrame(),
                'ticks': pd.DataFrame(),
                'status': {},
                'orders': {
                    'limit1': {'order_id': None, 'price': None, 'status': None},
                    'limit2': {'order_id': None, 'price': None, 'status': None},
                    'sl1': {'order_id': None, 'price': None, 'status': None},
                    'sl2': {'order_id': None, 'price': None, 'status': None}
                }
            }
        
        #Create an empty DataFrame with column names and index
        # Initialize the tick DF, so we can assign an index to it
        columns = ['Open', 'High', 'Low', 'Adj Close', 'Volume']
        index = pd.date_range('2023-01-01', periods=0, freq='D')
        tickersToTrack[token]['df'] = pd.DataFrame(columns=columns, index=index)
        tickersToTrack[token]['ticks'] = pd.DataFrame(columns=columns, index=index)
        tickersToTrack[token]['targetExitAchieved'] = []
# def trimMinuteDF(t):
#     #trim the minute df to last cfgMaxLookbackCandles minutes
#     tickersToTrack[t]['df'] = tickersToTrack[t]['df'].iloc[-cfgMaxLookbackCandles:]

def getHistoricalTickerData():
    #This code is intended to run before start of trading on day of
    #
    # We want last day of data only; but because we could start on 
    # monday morning, we request 3 days of data, and then truncate
    # the df to last cfgMaxLookbackCandles minute rows (i.e. last day 9:15 to 3:30)
    global tickerData, tickersToTrack, now
    start = now - datetime.timedelta(days=5)
    for t in tickersToTrack.keys():
        tickersToTrack[t]['df']= downloader.zget \
        (start,now,tickersToTrack[t]['ticker'],'minute',
         includeOptions=False,instrumentToken=t)
        if tickersToTrack[t]['df'].empty:
            tickerlog(f"Error:  getHistoricalTickerData: returned empty df for {t} from {start} to {now}")
        # trimMinuteDF(t)
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
    global nifty_ltp
    #tickerlog(f"{ticks}")
    #add the tick to the tick df
    for tick in ticks:
        token = tick['instrument_token']
        if tickersToTrack[token]['ticker'] == 'NIFTY 50':
            # Dont store/resample nifty ticks as it is not tradable
            # we subscribe just to get the ltp
            nifty_ltp = tick['last_price']
            continue
        #Insert this tick into the tick df
        tick_time = tick['exchange_timestamp']
        tick_time = ist.localize(tick_time)
        #Filter out volume for options (irrelevant) and for
        #indexes (vol not available)
        tick_volume = tick['volume_traded'] \
            if (('volume_traded' in tick) and \
                (utils.isNotAnOption(tickersToTrack[token]['ticker']) or \
                    cfgUseVolumeDataForOptions)) \
                    else 0
        tick_df_row = {
            'Open': tick['last_price'],
            'High': tick['last_price'],
            'Low': tick['last_price'],
            'Adj Close': tick['last_price'],
            'Volume': tick_volume
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
            if tickersToTrack[token]['ticker'] != 'NIFTY 50':
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
                historicalStart = datetime.datetime.combine(historicalEnd.date(),datetime.time(9, 15))#historicalEnd - datetime.timedelta(days=cfgHistoricalDaysToGet)
                historicalStart = ist.localize(historicalStart)
                tickersToTrack[token]['df']= downloader.zget \
                    (historicalStart,historicalEnd,tickersToTrack[token]['ticker'],'minute',
                    includeOptions=False,instrumentToken=token)
                if tickersToTrack[token]['df'].empty:
                    tickerlog(f"Error: zget returned empty df for {token} from {historicalStart} to {historicalEnd}")
                #trim to cfgMaxLookbackCandles rows/minutes
                # trimMinuteDF(token)
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

            # Remove the ticks that have been used to create the new minute candle
            tickersToTrack[token]['ticks'] = ticks_after_this_minute
            resampled_tokens.append(token)
    return resampled_tokens 

def tick(tokens):
    positions = tl.get_positions()
    tickerlog("Tick thread started")
    for token in tokens:
        tokenData = tickersToTrack[token]
        targetExitAchieved = tokenData['targetExitAchieved']
        tickerlog(f"tickThread generating signals for: token {token} {tokenData['ticker']}")
        tickerlog(f"positions: {positions} df: {tokenData['df']}")
        df = tl.generateSignalsAndTrade(tokenData['df'].copy(),positions,
                                   False,True,tradeStartTime=tradingStartTime,
                                   targetClosedPositions=targetExitAchieved)
        tickerlog(f"tickThread done generating signals for {token}.  Placing limit orders now")
        placeLimitOrders(df,tokenData['orders'])
    tickerlog("Tick thread done")
    email.ping()
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

def placeStopLossOrder(ticker,exchange,q,p): 
    print("Placing stop loss")
def placeTargetOrder(ticker,exchange,q,p):
    print("Placing target")
    # tl.placeOrder(ticker,exchange,'BUY',q,p,targetPrice,)
    
def placeStopLossAndTargetOrders (ticker,exchange,q,p):
    # placeStopLossOrder(ticker,exchange,q,p)
    # placeTargetOrder(ticker,exchange,q,p)
    x = None
def placeLimitOrders(df,orders):
    (orders['limit1']['price'],orders['limit2']['price'],orders['sl1']['price'],orders['sl2']['price']) = \
    (df['limit1'][-1],df['limit2'][-1],df['sl1'][-1],df['sl2'][-1])
    positions = tl.get_positions()
    (orders['limit1']['order_id'],orders['sl1']['order_id']) = tl.placeExitOrder(df,positions)
    # print(f"lim orderID: {orders['limit1']['order_id']} SL orderID: {orders['sl1']['order_id']}")
####### KITE TICKER CALLBACKS #######
def on_ticks(ws, ticks):
    # Callback to receive ticks.
    #tickerlog("Ticks:")    
    processTicks(ticks)
    # bid = ticks[0]['depth']['buy'][0]['price']
    # ask = ticks[0]['depth']['sell'][0]['price']
    
    # print(ctime(time()),">>>Bid=",bid," ASK=",ask, " Last Trade: ", ticks[0]["last_trade_time"])

def on_message(ws, payload, is_binary):
    # Callback to receive all messages.
    if is_binary:
        # VERY Chatty ? what are we getting ? 
        #tickerlog("Binary message received: {}".format(len(payload)))
        x = 1
    else:
        tickerlog("Message: {}".format(payload))
    
def on_order_update(ws, data):
    # Callback to receive order updates.
    timestamp_str = data['exchange_update_timestamp']
    if timestamp_str is None:
        tickerlog(f"Got order with None stimestamp. Ignoring. {data}")
        return
    format_str = '%Y-%m-%d %H:%M:%S'
    timestamp = datetime.datetime.strptime(timestamp_str, format_str)
    rounded_timestamp = datetime.datetime(timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute)
    localized_timestamp = ist.localize(rounded_timestamp)
    oid = data['order_id']
    ticker = data['tradingsymbol']
    exchange = data['exchange']
    token = data['instrument_token']
    status = data['status']
    tx = data['transaction_type']
    q = data['quantity']
    p = data['price']
    tag = data['tag']
    filled = data['filled_quantity']
    
#    tickerlog("Order update: {}".format(data))  
    #tickerlog(f"Order update: {tx} {q} {ticker}@{q} {status} filledQ:{filled}")
    if status == 'COMPLETE':
        if tag == 'main-long':
            placeStopLossAndTargetOrders(ticker,exchange,q,p)
    if tag == 'TargetExit' and status == 'COMPLETE':
        tickerlog(f"TargetExit order complete. Adding {localized_timestamp} to targetExitAchieved")
        tickersToTrack[token]['targetExitAchieved'].append(localized_timestamp)

    # { #NOTE:SAMPLE DATA
    # 'account_id': 'ZT1533', 'unfilled_quantity': 0, 'checksum': '', 'placed_by': 'ZT1533', 'order_id': '230411401989627', 'exchange_order_id': '2500000086137901', 'parent_order_id': None, 'status': 'OPEN', 'status_message': None, 'status_message_raw': None, 'order_timestamp': '2023-04-11 14:21:05', 'exchange_update_timestamp': '2023-04-11 14:21:05', 'exchange_timestamp': '2023-04-11 14:21:05', 'variety': 'regular', 'exchange': 'NFO', 'tradingsymbol': 'RELIANCE23APR2340PE', 'instrument_token': 36528898, 'order_type': 'LIMIT', 'transaction_type': 'BUY', 'validity': 'TTL', 'product': 'MIS', 'quantity': 250, 'disclosed_quantity': 0, 'price': 38, 'trigger_price': 0, 'average_price': 0, 'filled_quantity': 0, 'pending_quantity': 250, 'cancelled_quantity': 0, 'market_protection': 0, 'meta': {}, 'tag': None, 'guid': '71318X4ytREEwqnRDj'
    # }


    # {
    # 'account_id': 'ZT1533', 'unfilled_quantity': 0, 'checksum': '', 'placed_by': 'ZT1533', 'order_id': '230411401989627', 'exchange_order_id': '2500000086137901', 'parent_order_id': None, 'status': 'COMPLETE', 'status_message': None, 'status_message_raw': None, 'order_timestamp': '2023-04-11 14:21:05', 'exchange_update_timestamp': '2023-04-11 14:21:05', 'exchange_timestamp': '2023-04-11 14:21:05', 'variety': 'regular', 'exchange': 'NFO', 'tradingsymbol': 'RELIANCE23APR2340PE', 'instrument_token': 36528898, 'order_type': 'LIMIT', 'transaction_type': 'BUY', 'validity': 'TTL', 'product': 'MIS', 'quantity': 250, 'disclosed_quantity': 0, 'price': 38, 'trigger_price': 0, 'average_price': 37.25, 'filled_quantity': 250, 'pending_quantity': 0, 'cancelled_quantity': 0, 'market_protection': 0, 'meta': {}, 'tag': None, 'guid': '71318X4ytREEwqnRDj'
    # }

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
    tickerlog(f"Close called Code: {code}  Reason:{reason} Reconnecting")

    # On connection close stop the event loop.
    # Reconnection will not happen after executing `ws.stop()`
    # ws.stop()
    
# Callback when connection closed with error.
def on_error(ws, code, reason):
    tickerlog("Connection error: {code} - {reason}".format(code=code, reason=reason))

##########END KITE TICKER CALLBACKS ##########

########## KITE TICKER CONFIG AND START ##########
# Assign the callbacks.
kws.on_ticks = on_ticks
kws.on_connect = on_connect
kws.on_close = on_close
kws.on_order_update = on_order_update
kws.on_message = on_message
kws.on_error = on_error
# Infinite loop on the main thread. Nothing after this will run.
# You have to use the pre-defined callbacks to manage subscriptions.

while datetime.datetime.now(ist).time() < cfgStartTimeOfDay:
    email.ping()
    sleep(60)
    tickerlog("Waiting for 9:20")

tickerlog("Its 9:20 ! Starting algo")

kws.connect()
subprocess.call(["afplay", '/System/Library/Sounds/Glass.aiff'])
email.send_email("AlgoTrading Restarted", "AlgoTrading Restarted") if datetime.datetime.now(ist).hour > 9 else None

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