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
import os
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
niftyFutureTicker = utils.getImmidiateFutureTicker()

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
        
        if token in tickersToTrack.keys():
             # This will happen if we lost connection and are re-connection. 
             # No need to reset the df in that case, else we lose VolDelta
             # info in the old df (because we rebuild the df from historical,
             # and historical does not have VolDelta and OrderBook Imbalance info
             # ), and therefore positions will be inconsistent with live
            continue
        
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
        columns = ['Open', 'High', 'Low', 'Adj Close', 'Volume','buyVol','sellVol',\
            'volDelta', 'bid','ask','oi','buyQt', 'sellQt', 'obImabalance', 'obImabalanceRatio', \
                'buyQtLvl2', 'sellQtLvl2', 'obSTImabalance', 'obSTImabalanceRatio']
        index = pd.date_range('2023-01-01', periods=0, freq='D')
        tickersToTrack[token]['df'] = pd.DataFrame(columns=columns, index=index)
        tickersToTrack[token]['ticks'] = pd.DataFrame(columns=columns, index=index)
        tickersToTrack[token]['targetExitAchieved'] = []
        tickersToTrack[token]['minHigh'] = float('nan')
        tickersToTrack[token]['minLow'] = float('nan')
# def trimMinuteDF(t):
#     #trim the minute df to last cfgMaxLookbackCandles minutes
#     tickersToTrack[t]['df'] = tickersToTrack[t]['df'].iloc[-cfgMaxLookbackCandles:]
def getFutOrderBookBuySellQt(t):
    quote = tl.getFullQuote(t,'NFO')
    if quote is None:
        tickerlog(f"Error:  addOrderBookInfo: quote is None for {t}")
        return (0,0,0,0)
    oi = quote['oi']
    ltp = quote['last_price']
    lastQt = quote['last_quantity']
    orderBook = quote['depth']
    buyQt = quote['buy_quantity']
    sellQt = quote['sell_quantity']
    buyOrderBook = orderBook['buy']
    sellOrderBook = orderBook['sell']
    obImabalance = buyQt - sellQt
    

    buyQt2 = 0
    sellQt2 = 0
    for i in buyOrderBook:
        buyQt2 += i['quantity']
    for i in sellOrderBook:
        sellQt2 += i['quantity']    
    obSTImabalance = buyQt2 - sellQt2
    
    tickerlog(f"ltp:{ltp} lastQt:{lastQt} buyQt:{buyQt} sellQt:{sellQt} obImabalance:{obImabalance} / {round(buyQt/sellQt,2)} | buyQt2:{buyQt2} sellQt2:{sellQt2} obSTImabalance:{obSTImabalance} / {round(buyQt2/sellQt2,2)}")   
    # print(f"ltp:{ltp} lastQt:{lastQt} buyQt:{buyQt} sellQt:{sellQt} obImabalance:{obImabalance} / {round(buyQt/sellQt,2)} | buyQt2:{buyQt2} sellQt2:{sellQt2} obSTImabalance:{obSTImabalance} / {round(buyQt2/sellQt2,2)}")   
    # (t['df']['buyQt'],t['df']['sellQt'],t['df']['obImabalance'],t['df']['buyQt2'],t['df']['sellQt2'],t['df']['obSTImabalance']) = \
    #     (buyQt,sellQt,obImabalance,buyQt2,sellQt2,obSTImabalance)
    return (buyQt,sellQt,buyQt2,sellQt2)

#takes a one second tick df adds Vol Delta to it and returns it
def addVolDelta(df):
    # input df is in second tick format
    buyPrice = df['BuyPrice'].iloc[-1]
    df['midPrice'] = (df['BuyPrice'] + df['SellPrice'])/2
    df['buyVol'] = np.where(df['Adj Close'] >= df['midPrice'].shift(), df['Volume'], 0)
    df['sellVol'] = np.where(df['Adj Close'] <= df['midPrice'].shift(), df['Volume'], 0)
    df['VolDelta'] = df['buyVol'] - df['sellVol']
    df['VolDeltaRatio'] = df['buyVol'] / df['sellVol']
    df.drop(columns=['midPrice'], inplace=True)
    df.drop(columns=['BuyPrice', 'BuyQty', 'SellPrice', 'SellQty'], inplace=True)

    return df

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

def getLazyData():
    
    # getLazyData is a seperate pythong script that runs from cron 
    # every minute at the 30th second, and fetches VolDelta and PCR
    # stuff we dont want to do in the main loop
    # getLazyData dumps its updated output into a file, that we read here
    # to fill in the df
    
    # VolDeta: is one of the parameters that getLazyData gets
    # It Measures vol delta of intra-second up or down candles   
    # HFT Marker
    # I think its When derivative price has deviated too far from underlying; typically marks end of trend .. at least short term
    # but many times long term too
        
    file_path = f'{cfgLazyDataFilePath}'
    file_modified_time = os.path.getmtime(file_path)
    file_modified_time = datetime.datetime.fromtimestamp(file_modified_time)
    time_difference = datetime.datetime.now() - file_modified_time
    time_difference = time_difference.total_seconds()
    if time_difference > 60:
        tickerlog(f"getLazyData: File '{file_path}' is stale. Last Modified: {file_modified_time}.")
        return (0,0,0,0,0,0,0)
    
    try:
        with open(file_path, 'r') as file:
            # Read the contents of the file
            file_content = file.read().strip()

            # Split the text by comma separator
            split_values = file_content.split(',')
            # tickerlog(f"getSpaceMVolDelta: {split_values}")
            # Assign the values to separate variables
            pcr = split_values[0].strip()
            upVolNifty = split_values[1].strip()
            dnVolNifty = split_values[2].strip()
            upVolFut = split_values[3].strip()
            dnVolFut = split_values[4].strip()
            maxpain = split_values[5].strip()
            weightedAvMaxPain = split_values[6].strip()
            dnVolNifty = float(dnVolNifty)
            upVolNifty = float(upVolNifty)
            pcr = float(pcr)
            dnVolFut = float(dnVolFut)
            upVolFut = float(upVolFut)
            maxpain = float(maxpain)
            weightedAvMaxPain = float(weightedAvMaxPain)
            # tickerlog(f"getSpaceMVolDelta: {upVol} {dnVol}")
    except FileNotFoundError:
        tickerlog(f"File '{file_path}' not found.")
        return (0,0,0,0,0,0,0)
    except Exception as e:
        tickerlog(f"Error occurred while reading the file: {str(e)}")
        return (0,0,0,0,0,0,0)

    return (pcr,upVolNifty,dnVolNifty,upVolFut,dnVolFut,maxpain,weightedAvMaxPain)

def getVolDelta(token,tick):
    tickTimeStamp = tick['exchange_timestamp']
    if tickersToTrack[token]['ticks'].empty:
        buyVol = sellVol = volDelta = lastOrderBookMidPrice= bid = ask = this_tick_volume = 0
    else:
        (o,h,l,c,v) = getTickOHLCV(tick,token)
        o = tickersToTrack[token]['ticks']['Open'][-1]
        
        #voume_traded in tick is cumulative volume, so subtract last tick volume to get this tick volume
        lastVol = tickersToTrack[token]['ticks']['Volume'][-1] if not tickersToTrack[token]['ticks'].empty else 0
        this_tick_volume = v - lastVol

        buyVol = this_tick_volume if c > o else 0
        sellVol = this_tick_volume if c < o else 0
        volDelta = buyVol - sellVol

        # this_tick_price = tick['last_price']
        
        # bid = tickersToTrack[token]['ticks']['bid'][-1]
        # ask = tickersToTrack[token]['ticks']['ask'][-1]
        # lastOrderBookMidPrice = (bid+ask)/2
        # buyVol = this_tick_volume if this_tick_price >= lastOrderBookMidPrice else 0
        # sellVol = this_tick_volume if this_tick_price <= lastOrderBookMidPrice else 0
        # volDelta = buyVol - sellVol
    return (buyVol,sellVol,volDelta,this_tick_volume)

def getTickOHLCV(tick,token):
    # OHLC is OHLC for the day, not for the tick
    # ohlc = tick['ohlc'] 
    # (o,h,l,c) = (ohlc['open'],ohlc['high'],ohlc['low'],ohlc['close'])
    
    (o,h,l,c) = (tick['last_price'],tick['last_price'],tick['last_price'],tick['last_price'])
    
    #Filter out volume for options (irrelevant) and for
    #indexes (vol not available)

    v = tick['volume_traded'] \
    if (('volume_traded' in tick) and \
        (utils.isNotAnOption(tickersToTrack[token]['ticker']) or \
            cfgUseVolumeDataForOptions)) \
            else 0
    return (o,h,l,c,v)

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
        buyOrders = tick['depth']['buy']
        sellOrders = tick['depth']['sell'] 
        buyQt2 = sellQt2 = 0
        for od in buyOrders:
            buyQt2 += od['quantity']
        for od in sellOrders:
            sellQt2 += od['quantity']

        (o,h,l,c,v) = getTickOHLCV(tick,token)
        
        (buyVol,sellVol,volDelta,this_tick_volume) = getVolDelta(token,tick)
        #Sample tick Data
        # 09:33:01 AM Ticks:[{'tradable': True, 'mode': 'full', 'instrument_token': 14762754, 'last_price': 272.2, 'last_traded_quantity': 50, 'average_traded_price': 255.83, 'volume_traded': 528650, 'total_buy_quantity': 149100, 'total_sell_quantity': 77150, 'ohlc': {'open': 250.1, 'high': 278.7, 'low': 227.0, 'close': 263.05}, 'change': 3.4784261547234276, 'last_trade_time': datetime.datetime(2023, 6, 22, 9, 33), 'oi': 611400, 'oi_day_high': 613900, 'oi_day_low': 574800, 'exchange_timestamp': datetime.datetime(2023, 6, 22, 9, 33, 1), 'depth': {'buy': [{'quantity': 350, 'price': 272.25, 'orders': 2}, {'quantity': 600, 'price': 272.2, 'orders': 2}, {'quantity': 500, 'price': 272.15, 'orders': 1}, {'quantity': 1250, 'price': 272.1, 'orders': 5}, {'quantity': 200, 'price': 272.05, 'orders': 2}], 'sell': [{'quantity': 50, 'price': 272.85, 'orders': 1}, {'quantity': 1150, 'price': 272.9, 'orders': 4}, {'quantity': 1250, 'price': 272.95, 'orders': 3}, {'quantity': 2650, 'price': 273.0, 'orders': 6}, {'quantity': 800, 'price': 273.05, 'orders': 3}]}}]


        tick_df_row = {
            'Open': o,
            'High': h,
            'Low': l,
            'Adj Close': c,
            'Volume': v,
            'buyVol': buyVol,
            'sellVol': sellVol,
            'volDelta': volDelta,
            'bid': buyOrders[0]['price'] if len(buyOrders) > 0 else 0,
            'ask': sellOrders[0]['price'] if len(sellOrders) > 0 else 0,
            'oi': tick['oi'] if 'oi' in tick else 0,
            'buyQt': tick['total_buy_quantity'],
            'sellQt': tick['total_sell_quantity'],
            'obImabalance': tick['total_buy_quantity'] - tick['total_sell_quantity'],
            'obImabalanceRatio': tick['total_buy_quantity']/tick['total_sell_quantity'],
            'buyQtLvl2': buyQt2,
            'sellQtLvl2': sellQt2,
            'obSTImabalance': buyQt2 - sellQt2,
            'obSTImabalanceRatio': buyQt2/sellQt2,
        }
        tickerlog(f"{tick_time.minute}:{tick_time.second}:{tick_time.microsecond} V: {this_tick_volume} ltp: {tick['last_price']} buyVol:{buyVol} sellVol:{sellVol} VolDelta: {volDelta} OrderQtImbalance1: {tick['total_buy_quantity'] - tick['total_sell_quantity']} OrderQtImbalance2: {buyQt2 - sellQt2}")
        tickersToTrack[token]['ticks'].loc[tick_time] = tick_df_row  

def resampleToMinDF():
    global nifty_ltp
    
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
            # print(f"ticks_upto_this_minute: {ticks_upto_this_minute}")
            #print(f"ticks_after_this_minute: {ticks_after_this_minute}")
            resampled_ticks_upto_this_minute = \
            ticks_upto_this_minute.resample('1min').agg({
              'Open': 'first',
              'High': 'max',
              'Low': 'min',
              'Adj Close': 'last',
              'Volume': lambda x: x[-1] - x[0], #'sum' volume data in ticks is cumulative  ,
              'buyVol': 'sum',
              'sellVol': 'sum',
              'volDelta': 'sum',
              'bid': 'last',
              'ask': 'last',
              'oi': 'last',
              'buyQt': 'mean',
              'sellQt': 'mean',
              'obImabalance': 'mean',
              'obImabalanceRatio': 'mean',
              'buyQtLvl2': 'mean',
              'sellQtLvl2': 'mean',
              'obSTImabalance': 'mean',
              'obSTImabalanceRatio': 'mean'
            })

            #minHigh/minLow are used to note the actual execution prices of limit and SL orders.  Sometimes these orders get hit
            #intra-second, and we never get these extreme prices in our second tick snapshot.  So, these High/Low prices
            #never make it to our minute df.  Here we synthetically add them to High/Low values so that our signal's
            #df is aware that these were actual High/Low values and our limit/SL orders have indeed fired. This is important
            resampled_ticks_upto_this_minute['High'] = resampled_ticks_upto_this_minute['High'].combine(tickersToTrack[token]['minHigh'], max)
            resampled_ticks_upto_this_minute['Low'] = resampled_ticks_upto_this_minute['Low'].combine(tickersToTrack[token]['minLow'], min)
            tickerlog(f"minHigh: {tickersToTrack[token]['minHigh']} minLow: {tickersToTrack[token]['minLow']} High: {resampled_ticks_upto_this_minute['High']} Low: {resampled_ticks_upto_this_minute['Low']}")
            tickersToTrack[token]['minHigh'] = tickersToTrack[token]['minLow'] = float('nan')
            
            resampled_ticks_upto_this_minute['maxVolDelta'] = ticks_upto_this_minute['volDelta'].max()
            resampled_ticks_upto_this_minute['minVolDelta'] = ticks_upto_this_minute['volDelta'].min()
            # ratio = resampled_ticks_upto_this_minute['buyQtLvl2']/resampled_ticks_upto_this_minute['sellQtLvl2']
            # tickerlog(f"Resampled VolDelta: {ratio}")
            # tickerlog(f"Resampled {resampled_ticks_upto_this_minute['buyQtLvl2']} s:{resampled_ticks_upto_this_minute['sellQtLvl2']} ratio:{resampled_ticks_upto_this_minute['buyQtLvl2']/resampled_ticks_upto_this_minute['sellQtLvl2']}")
            #For now drop all minute_candle df's other than OCHLV and i and symbol
            #Analytics will recreate them, we dont want to confure analytics
            #TODO: Fix this, figure out how to keep that data and just do analytics on the 
            #new minute candle; this will make the tick much faster
            
            keep_columns = ['Open', 'High', 'Low', 'Adj Close', 'Volume','symbol','i','buyVol','sellVol','volDelta',\
                'maxVolDelta','minVolDelta','bid','ask','oi','buyQt', 'sellQt', 'obImabalance', 'obImabalanceRatio', 'buyQtLvl2', \
                    'sellQtLvl2', 'obSTImabalance', 'obSTImabalanceRatio', 'futOrderBookBuyQt', 'futOrderBookSellQt', \
                        'futOrderBookBuyQtLevel1', 'futOrderBookSellQtLevel1', 'niftyUpVol', \
                        'niftyDnVol', 'niftyFutureUpVol', 'niftyFutureDnVol', 'nifty', 'niftyPCR', 'niftyMaxPain', 'niftyWMaxPain']
            drop_columns = list(set(minute_candle_df.columns) - set(keep_columns))
            minute_candle_df.drop(columns=drop_columns, inplace=True)
            
            resampled_ticks_upto_this_minute['nifty'] = nifty_ltp
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
                # tickerlog(f"Adding resampled rows {resampled_ticks_upto_this_minute}  to minute candle {tickersToTrack[token]['df'].tail()}")

            # Remove the ticks that have been used to create the new minute candle
            tickersToTrack[token]['ticks'] = ticks_after_this_minute
            resampled_tokens.append(token)
    return resampled_tokens 

def addNiftyVolDelta(token):
    df = tickersToTrack[token]['df']
    if not 'futOrderBookBuyQt' in df.columns:
        #initialize if it doesnt already exist
        df['futOrderBookBuyQt'] = df['futOrderBookSellQt'] = df['futOrderBookBuyQtLevel1'] = df['futOrderBookSellQtLevel1'] \
            = df['niftyUpVol'] = df['niftyDnVol'] = df['niftyFutureUpVol'] = df['niftyFutureDnVol'] = df['niftyPCR'] = \
                df['niftyMaxPain'] = df['niftyWMaxPain'] = 0
        
    (df['futOrderBookBuyQt'][-1],df['futOrderBookSellQt'][-1],df['futOrderBookBuyQtLevel1'][-1],df['futOrderBookSellQtLevel1'][-1]) =\
        getFutOrderBookBuySellQt(niftyFutureTicker)
    (df['niftyPCR'][-1],df['niftyUpVol'][-1],df['niftyDnVol'][-1],df['niftyFutureUpVol'][-1],df['niftyFutureDnVol'][-1],df['niftyMaxPain'],df['niftyWMaxPain']) = \
        getLazyData()


def tick(tokens):
    positions = tl.get_positions()
    # tickerlog("Tick thread started")
    for token in tokens:
        tokenData = tickersToTrack[token]
        targetExitAchieved = tokenData['targetExitAchieved']
        addNiftyVolDelta(token)
        # tickerlog(f"tickThread generating signals for: token {token} {tokenData['ticker']}")
        # tickerlog(f"positions: {positions} df: {tokenData['df']}")
        # tickerlog(f"{tokenData['df']['futOrderBookBuyQt']}")
        # tickerlog(f"{tokenData['df']['niftyFutureDnVol']}")
        df = tl.generateSignalsAndTrade(tokenData['df'].copy(),positions,
                                   False,True,tradeStartTime=tradingStartTime,
                                   targetClosedPositions=targetExitAchieved)
        # tickerlog(f"tickThread done generating signals for {token}.  Placing limit orders now")
        placeLimitOrders(df,tokenData['orders']) if df['symbol'][0] != 'NIFTY23JUNFUT' else None
    # tickerlog("Tick thread done")
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
    # tickerlog(f"Ticks:{ticks}")    
    processTicks(ticks)
    # bid = ticks[0]['depth']['buy'][0]['price']
    # ask = ticks[0]['depth']['sell'][0]['price']
    
    # print(ctime(time()),">>>Bid=",bid," ASK=",ask, " Last Trade: ", ticks[0]["last_trade_time"])

def on_message(ws, payload, is_binary):
    None
    # Callback to receive all messages.
    # if is_binary:
    #     # VERY Chatty ? what are we getting ? 
    #     #tickerlog("Binary message received: {}".format(len(payload)))
    #     x = 1
    # else:
    #     tickerlog("Message: {}".format(payload))
    
def on_order_update(ws, data):
    # Callback to receive order updates.
    timestamp_str = data['exchange_update_timestamp']
    if timestamp_str is None:
        tickerlog(f"Got order with None stimestamp. Ignoring. {data}")
        return
    token = data['instrument_token']
    ticker = data['tradingsymbol']

    if not token in tickersToTrack.keys():
        # could be a manual trade
        tickerlog(f"Got order for token {token} : {ticker} that we are not tracking. Ignoring. {data}")
        return
    
    format_str = '%Y-%m-%d %H:%M:%S'
    timestamp = datetime.datetime.strptime(timestamp_str, format_str)
    rounded_timestamp = datetime.datetime(timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute)
    localized_timestamp = ist.localize(rounded_timestamp)
    oid = data['order_id']
    exchange = data['exchange']
    status = data['status']
    tx = data['transaction_type']
    q = data['quantity']
    p = data['price']
    tag = data['tag']
    filled = data['filled_quantity']
    order_type = data['order_type']
    trigger_price = data['trigger_price']   
    
#    tickerlog("Order update: {}".format(data))  
    #tickerlog(f"Order update: {tx} {q} {ticker}@{q} {status} filledQ:{filled}")
    if status == 'COMPLETE':
        if tag == 'main-long':
            placeStopLossAndTargetOrders(ticker,exchange,q,p)
    if tag == 'Exit1' and status == 'COMPLETE':
        tickerlog(f"TargetExit order complete. Adding {localized_timestamp} to targetExitAchieved")
        tickersToTrack[token]['targetExitAchieved'].append(localized_timestamp)

    minHigh = minLow = float('nan')
    if order_type == 'LIMIT' and status == 'COMPLETE':
        if tx == 'BUY':
            minHigh = p+.1
            tickerlog(f"Limit Order update: {tx} {q} {ticker}@{q} {status} filledQ:{filled} minHigh:{minHigh}")
        else:
            minLow = p-.1
            tickerlog(f"Limit Order update: {tx} {q} {ticker}@{q} {status} filledQ:{filled} minLow:{minLow}")
    elif order_type == 'SL' and status == 'COMPLETE':
        if tx == 'BUY':
            minLow = trigger_price
            tickerlog(f"SL Order update: {tx} {q} {ticker}@{q} {status} filledQ:{filled} minLow:{minLow}")
        else:
            minHigh = trigger_price
            tickerlog(f"SL Order update: {tx} {q} {ticker}@{q} {status} filledQ:{filled} minHigh:{minHigh}")
    
    # Set minHigh/minLow for the minute, we want to ensure that the minute candles
    # we build have at least these as high/Low value.  Sometimes these orders get executed intra-second, 
    # but the ltp second snapshot we get in KitTicker does not see/capture these extreme values, 
    # so our signal's df is not aware that these orders have been
    # executed, so we synthetically insert these as minHigh/minLow for the minute
    tickersToTrack[token]['minHigh'] = max(minHigh,tickersToTrack[token]['minHigh']) if not math.isnan(minHigh) else tickersToTrack[token]['minHigh']
    tickersToTrack[token]['minLow'] = min(minLow,tickersToTrack[token]['minLow']) if not math.isnan(minLow) else tickersToTrack[token]['minLow']
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
print(f"time is {datetime.datetime.now(ist).time()}")
print(f"start time is {cfgStartTimeOfDay}")
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