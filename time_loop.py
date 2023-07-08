#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed Mar  1 19:42:07 2023

@author: nikhilsama
"""

isTimeLoop = __name__ == '__main__'

import os
import subprocess
import log_setup
import time
import datetime
from datetime import date,timedelta,timezone
import tickerdata as td
import numpy as np
import signals as signals
import pandas as pd
import DownloadHistorical as downloader
import kite_init as ki 
from DatabaseLogin import DBBasic
import performance as perf

from freezegun import freeze_time   
import pytz
import logging
import signal
import sys
import math
import threading
import tickerCfg
import utils

#cfg has all the config parameters make them all globals here
import cfg
globals().update(vars(cfg))


if showTradingViewLive:
    import live_charting as liveCharts
    liveCharts.init()
    global liveTVThread 
    liveTVThread = None

def sigterm_handler(_signo, _stack_frame):
    # Raises SystemExit(0):
    print("SIGTERM -- EXITING POSITIONS")
    logging.info("SIGTERM -- EXITING POSITIONS, and EXITINT ALGO")
    ki.exit_positions(kite)
    sys.exit(0) 
# signal.signal(signal.SIGTERM, sigterm_handler)

## TEST FREEZESR START

if cfgFreezeGun:
    from freezegun import freeze_time
    freezeTime = "Apr 11th, 2023 10:58:00+0530"
    freezer = freeze_time(freezeTime, tick=True)
    freezer.start()
    print ("Freeze gun is on. Time is frozen at: ", freezeTime)
    ## END 

# set timezone to IST
ist = pytz.timezone('Asia/Kolkata')

def applyTickerSpecificCfg(ticker):
        
    tCfg = utils.getTickerCfg(ticker)
    
    for key, value in tCfg.items():
        globals()[key] = value
        #print(f"setting {key} to {value}")

def sleep_till_10am():
    global now
    
    tenAMToday = signals.tenAMToday(now)
    
    # If the target time has already passed today, set it for tomorrow
    if now < tenAMToday:    
        # Convert the target datetime to a timestamp
        tenAM_timestamp = time.mktime(tenAMToday.timetuple())
        logging.info(f'It is {now.strftime("%I:%M %p")} <  (10AM) = {tenAMToday.strftime("%I:%M %p")}  Sleeping: {tenAM_timestamp - time.time()} seconds')
        # Sleep until the target time
        time.sleep(tenAM_timestamp - time.time())
    return
    
# get current datetime in IST
global tradingStartTime
tradingStartTime = datetime.datetime.now(ist)
now = tradingStartTime
#sleep_till_10am()

#sleep a few seconds to ensure we kick off on a rounded minute
#time.sleep(60-now.second)
now = datetime.datetime.now(ist)

print(f'Tick @ {now.strftime("%I:%M %p")}')
logging.info(f'>>>>>>>>NEW RUN @ {now.strftime("%I:%M:%S %p")}>>>>>>>')
logging.debug(f">>>>>>>>DEBUG IS ON>>>>>>>")

db = DBBasic() 

kite = ki.initKite()

def get_positions():
    positions = ki.get_positions(kite)
    for t in positions:
        for pos in positions[t]['positions']:
            if utils.isOption(pos['tradingsymbol']):
                optionTicker,lot_size,tick_size,strike = db.get_option_ticker(pos['tradingsymbol'], None, None,None) #if t is an option other arguments are not looked at
                pos['lot_size'] = lot_size
                pos['tick_size'] = tick_size
            else:
                pos['lot_size'] = 1
                pos['tick_size'] = 0.05
    return positions

def get_kite_access_token():
    return ki.getAccessToken(kite)
def get_ltp(t,exch):
    return ki.get_ltp(kite,t,exch)
def placeOrder(t,exchange,tx_type,q,ltp,p,tag,order_type):
    ki.exec(kite,t,exchange,tx_type,q=q,ltp=ltp,p=p,tag=tag,
            order_type=order_type)
def getFullQuote(t,exch):
    return ki.get_quote(kite,t,'NFO')

# BUY CONDITION
# Buy if current signal is 1 and position is not 1, 
# or if position is 1 and net position is not 1 as long as signal is not -1
# Logic for the second one is a missed candle or a missed trade(order not executed) 
# can lead to position being 1, and net_postion not 1; we shoudl fix this
# UNLESS the signal is -1 or 0 (nan or 1 is ok), in which case we should not, since we are going short this tick anyway

def buyCondition(df,net_position):
    
    return net_position != 1 and \
        ((df['signal'][-1] == 1 and df['position'][-1] != 1) or \
        (df['position'][-1] == 1 and df['signal'][-1] != -1 and df['signal'][-1] != 0))

# SELL CONDITION
# Sell if current signal is -1 and position is not -1,
# or if position is -1 and net position is not -1 as long as signal is not 1
# Logic for the second one is a missed candle or a missed trade(order not executed)
# can lead to position being -1, and net_postion not -1; we shoudl fix this
# UNLESS the signal is 1 or 0 (nan or -1 is ok), in which case we should not, since we are going long this tick anyway

def sellCondition(df,net_position):
    return net_position != -1 and \
        ((df['signal'][-1] == -1 and df['position'][-1] != -1) or \
            (df['position'][-1] == -1 and df['signal'][-1] != 1 and df['signal'][-1] != 0))

# EXIT CONDITION
# Exit if there is a current position (net_position != 0) 
# and current signal is 0 or current signal is nan and position is 0 (missed candle or missed trade)
def exitCondition(df,net_position):
    #We have a position AND
    #   Either current signal is to exit (0) OR
    #   current signal is nan and position is 0 (missed candle or missed trade) 
    return (net_position != 0) and \
        ((df['signal'][-1] == 0) or \
        (df['position'][-1] == 0 and math.isnan(df['signal'][-1])))

def exitCurrentPosition(t,positions,net_position,nextPosition):
    qToExit = 0
    optType = None
    if utils.isOption(t):
        (t,optType) = utils.explodeOptionTicker(t)
        if optType == 'PE':
            nextPosition = -1 * nextPosition
                
    if t in positions: 
        #if multiple position exists, exit all
        if (len(positions[t]['positions']) > 1):
            logging.error("More than one position exists for {t}. Exiting all")
            ki.exit_given_positions(kite,positions[t]['positions'],nextPosition)
        else:
            if (net_position == -1 and nextPosition == 1) or \
                (net_position == 1 and nextPosition == -1):
                #Exit Options positions, because we always engage in options by selling
                #ie: if we are short a stock, we sold a call option; we dont reverse
                #positions by buying 2 calls; instead we sell the call and then sell
                #a put;  So we need to exit any options positions here if the 
                #net position is not consistant with nextPosition
                if (optType is None):
                    #ticker is not an option, exit any option positions we took
                    #for this ticker; cant doubleQ to exit
                    ki.exitNFOPositionsONLY(kite,positions[t]['positions'])
                    #For Equity positions we can reverse simply by doubling the buy when 
                    # we change direction; so we dont need to exit equity positions here
                    qToExit = positions[t]['positions'][0]['quantity'] if \
                        positions[t]['positions'][0]['exchange'] != 'NFO' else 0
                else:
                    #ticker is an  option, we can exit by just doubling the quantity
                    qToExit = positions[t]['positions'][0]['quantity'] if \
                        positions[t]['positions'][0]['exchange'] == 'NFO' else 0# We have a short position, so we need to exit both

    return abs(qToExit)


def tradeNotification(type, t,ltp,signal,position,net_position):
    subprocess.call(["afplay", '/System/Library/Sounds/Glass.aiff'])
    logging.info(f"{type} {t} LastCandleClose:{ltp} Signal:{signal} Position:{position} NetPosition:{net_position}" )
    if showTradingViewLive and type != 'EXIT':
        global liveTVThread 
        if liveTVThread is not None: 
            liveTVThread.join()
        liveTVThread = threading.Thread(target=liveCharts.loadChart, 
                                                args=(t,)) # funky syntax needs (t,) if only one arg to signify its a tuple
        liveTVThread.start()
            #liveCharts.loadChart(t)

def checkPositionsForConsistency(positions,df):    
    ## DF Position can mismatch with kit positions for several reasons
    # 1) We missed the previous tick -- Either softare was slow, or kite 
    #  historical data api failed ephemerally, or in some cases it is is 
    # even possible that kite api returns an incomplete final tick, with
    # best OHLV data, and that tick data itself changes when fetched again 
    # in future, as complete data is complied with acurate data
    #
    # Either way if df position is inconsistant w Kite, then change Kite
    # position to become consistant
    net_position = 0

    t = df['symbol'][0]
    signal = df['signal'][-1]
    position = df['position'][-1]
    optType = None
    
    if utils.isOption(t):
        (t,optType) = utils.explodeOptionTicker(t)
    
    # print(f"Checking {t} pos: {position} signal: {signal} for consistency")
    # print(f"Kite Positions: {positions}")
    if t in positions:
        net_position = positions[t]['net_position']
        if optType == 'PE':
            net_position = -1 * net_position#HACK: For Put Options, we need to reverse the net_position, since positions are based 
        #HACK long or short of the underlying ticker.  So if we go long a put option, we are short the underlying ticker
            
        # net_position can be 1,-1 or 'inconsistent'
        # if inconsistent, then we need to exit all positions
        if (np.isnan(signal) and position != net_position):
#                if (df['signal'][-1] != net_position):
            logging.info(f"{t}: Exiting inconsistant Positions.  Live Kite positions({positions[t]['net_position']} inconsistant with DF pos:{position} signal: {signal} ")
            ki.exit_given_positions(kite,positions[t]['positions'],signal)
            net_position =0
            del positions[t]
#        elif (not math.isnan(df['signal'][-1])) and df['position'][-1] != 0:
    elif position != 0 or signal != 0:
            logging.info(f"{t}: No Live Kite positions inconsistant with DF ({position})")
    
    return net_position
def placeEntrySL1Order(df):
    slTrigger = df['sl1'][-1] 
    
    if slTrigger > 1: 
        oType = 'BUY'
    elif slTrigger < -1:
        oType = 'SELL'
        slTrigger = abs(slTrigger)
    else: # shoulud never happen
        print("ERROR: slTrigger is 0")
        logging.error("ERROR: slTrigger is 0.  Exiting")
        exit(-1)
        
    t = df['symbol'][0]
    ltp = df['Adj Close'][-1]
    
    if utils.tickerIsFutOrOption(t):
        tput,lot_size,tick_size,strike = db.get_option_ticker(t, ltp, 'XX')
        exch = 'NFO'
    else:
        lot_size = 1
        tick_size = 0.05
        exch = 'NSE'
    logging.info(f"EntrySLOrder => {t}:{oType} @ {slTrigger} betsize:{bet_size}")
    return ki.exec_sl(kite,t,exch,oType,slTrigger,lot_size,tick_size, ltp=ltp, betsize=bet_size,tag='EntrySL1')
def placeEntryLimit1Order(df):
    p = df['limit1'][-1] 
    
    if p > 1: 
        oType = 'BUY'
    elif p < -1:
        oType = 'SELL'
    else: # shoulud never happen
        print("ERROR: limit is 0")
        logging.error("ERROR: limit is 0.  Exiting")
        exit(-1)
        
    t = df['symbol'][0]
    ltp = df['Adj Close'][-1]
    
    if utils.tickerIsFutOrOption(t):
        tput,lot_size,tick_size,strike = db.get_option_ticker(t, ltp, 'XX')
        exch = 'NFO'
    else:
        lot_size = 1
        tick_size = 0.05
        exch = 'NSE'
    logging.info(f"EntryLimitOrder => {t}:{oType} @ {p} betsize:{bet_size}")
    (lim1OrderId,SL1OrderId) = ki.exec(kite,t,exch,oType,lot_size=lot_size,tick_size=tick_size,p=abs(p),ltp=ltp,tag='Entry1')

    return lim1OrderId

def placeEntryOrder(df):
    if df.empty or \
        ((not 'sl1' in df.columns or np.isnan(df['sl1'][-1])) and \
            (not 'limit1' in df.columns or np.isnan(df['limit1'][-1]))):
        return
    
    if 'sl1' in df.columns and not np.isnan(df['sl1'][-1]):
        entrySLOrderId = placeEntrySL1Order(df)
    if 'limit1' in df.columns and not np.isnan(df['limit1'][-1]):
        placeEntryLimit1Order(df)
     

def placeExitOrder(df,positions):
    optType = None
    t = df['symbol'][0]
    (lim1OrderId,SL1OrderId) = (None,None)
    if utils.isOption(t):
        (t,optType) = utils.explodeOptionTicker(t)

    if df['position'][-1] == 0 \
        or t not in positions:
        lim1OrderId = placeEntryOrder(df)
        return (lim1OrderId,SL1OrderId)# no target order if there is no position, or if ticker is trending
        # or if we dont have a position for this ticker(we should exit later anyway)


    if len(positions[t]['positions']) > 1:
        logging.error("More than one position exists for {t}. No Exit order")
        return (lim1OrderId,SL1OrderId)
    
    if df['renko_brick_diff'][-1] != 0:
        logging.info(f"Skipping Target Exit order for {t} as brick diff is {df['renko_brick_diff'][-1]}")
        return (lim1OrderId,SL1OrderId)
    
    pos = positions[t]['positions'][0]
    
    ticker = pos['tradingsymbol']
    qt = abs(pos['quantity'])   
    exch = pos['exchange']
    lot_size = pos['lot_size']
    tick_size = pos['tick_size']

    oType = 'SELL' if pos['quantity'] > 0 else 'BUY'
    if (optType is not None) or (exch != 'NFO'):
        #If ticker is an option, or if ticker is an option then df has the exit price            
        # price = df['upper_band'][-1] if pos['quantity'] > 0 \
        #     else df['lower_band'][-1]
        (longTarget,longSL,shortTarget,shortSL) = signals.getTickerRenkoTargetPrices(df.iloc[-1])
        if pos['quantity'] > 0:
            # logging.info("Long Target Exit Order target is {longTarget} close: {df['Adj Close'][-1]} sl: {longSL}")
            target = max(longTarget,df['Adj Close'][-1]) if np.isnan(df['limit1'][-1]) else abs(df['limit1'][-1])
            sl = longSL if np.isnan(df['sl1'][-1]) else abs(df['sl1'][-1])
            if not np.isnan(df['sl1'][-1]) and df['sl1'][-1] > 0:
                print("ERROR: SL is positive(buy) when we have a long position")
            if not np.isnan(df['limit1'][-1]) and df['limit1'][-1] > 0:
                print("ERROR: Limit1 is positive(buy) when we have a long position")

        else:
            # logging.info(f"Short Target Exit Order. target is {shortTarget} close: {df['Adj Close'][-1]} sl: {shortSL}")
            target= min(shortTarget,df['Adj Close'][-1]) if np.isnan(df['limit1'][-1]) else abs(df['limit1'][-1])
            sl = shortSL if np.isnan(df['sl1'][-1]) else abs(df['sl1'][-1])
            if not np.isnan(df['sl1'][-1]) and df['sl1'][-1] < 0:
                print("ERROR: SL is negative(sell) when we have a short position")
            if not np.isnan(df['limit1'][-1]) and df['limit1'][-1] < 0:
                print("ERROR: Limit1 is negative(sell) when we have a short position")

        slqt = qt
        targetqt = max(round(qt*cfgPartialExitPercent/lot_size)*lot_size,lot_size)
        if targetqt == 0:
            logging.info(f"Skipping 0 Qt ExitOrders => {t}:{oType}  Target:{targetqt} @ {target} SL:{slqt} @ {sl} brick diff:{df['renko_brick_diff'][-1]}")
        else:
            logging.info(f"ExitOrders => {t}:{oType}  Target:{targetqt} @ {target} SL:{slqt} @ {sl} brick diff:{df['renko_brick_diff'][-1]}")
            (lim1OrderId,SL1OrderId) = ki.exec(kite,ticker,exch,oType,lot_size=lot_size,tick_size=tick_size,
                    q=targetqt,p=target,tag="Exit1",sl=1,sltrigger=sl,slTxType=oType,slqt=slqt)
    else: #ticker is equity underlying, and we are trading its option
        # df does not have the bb exit price, so we can skip for now
        #TODO: Calculate exit price
        return (lim1OrderId,SL1OrderId)
    return (lim1OrderId,SL1OrderId)

def checkAndUpdateTargetExits(df, targetClosedPositions):
    #return df
    #Check if ticker has already exited and hit target within candle,
    #before Adj Close hit the exit band
    #Mark it as such so that our kite positions are
    #consistent with df
    #df.loc[df.index.isin(targetClosedPositions), 'signal'] = 0
    # return df
    if targetClosedPositions is None:
        return df
    for closedPositionTime in targetClosedPositions:
        # if closedPositionTime == df.index[-1]:
        #     logging.info(f"Target Exit for {df['symbol'][0]} hit within candle")
        #logging.info(f"Adding exit for {df['symbol'][0]} at {closedPositionTime}")
        if closedPositionTime in df.index and pd.isna(df.loc[closedPositionTime, 'signal']):
            df.at[closedPositionTime, 'signal'] = 0
            logging.info(f"{df['symbol'][0]}: Target Exit hit at {closedPositionTime}")
        # else:
        #     logging.info(f"Did not add Target Exit order. signal is {df.loc[closedPositionTime, 'signal']} => not nan")
    return df

def generateSignalsAndTrade(df,positions,stock,options,tradeStartTime=None, dataPopulators=None, signalGenerators=None,
                            targetClosedPositions=None):
    global tradingStartTime
    
    if df.empty:
        return

    # Stuff to hack if this function is called from KitTicker instaed of 
    # time_loop
    global now
    
    applyTickerSpecificCfg(df['symbol'][0]) 

    now = df.index[-1]
    if tradeStartTime is not None:
        tradingStartTime = tradeStartTime
    else: 
        tradingStartTime = df.index[0]
        
    t = df['symbol'][0]
    
    #update moving averages and get signals
    dataPopulators = {
        'daily': [
            signals.populateATR,
            signals.populateRenko,
            signals.populateBB,     
            # signals.populateADX, 
            # signals.populateSuperTrend,
            # signals.populateOBV,
            # signals.vwap,
            signals.populateSVP,
            signals.populateVolDelta,

            # signals.populateCandleStickPatterns
        ], 
        'hourly': [
        ]
    } if dataPopulators is None else dataPopulators
    
    signalGenerators = [    
                        signals.followSVP
                        #signals.followObvAdxMA
                        #signals.followObvMA
                        #,signals.exitOBV
                        # signals.getSig_BB_CX
                        # ,signals.getSig_ADX_FILTER
                        # ,signals.getSig_MASLOPE_FILTER
                        # ,signals.getSig_OBV_FILTER
                        # ,signals.getSig_exitAnyExtremeADX_OBV_MA20_OVERRIDE
                        # ,signals.getSig_followAllExtremeADX_OBV_MA20_OVERRIDE
                        # ,signals.exitTrendFollowing
                        #,signals.exitTargetOrSL

                        ] if signalGenerators is None else signalGenerators
    overrideSignalGenerators = []   
    df = \
        signals.applyIntraDayStrategy(df,dataPopulators,signalGenerators, \
                                overrideSignalGenerators, tradeStartTime=tradingStartTime)
    # update df for any target exits that happened within candle
    df = checkAndUpdateTargetExits(df,targetClosedPositions)
    df = perf.calculate_positions(df,close_at_end=False)
    # place target exit order if needed
    #placeExitOrder(df,positions)
    
    downloader.cache_df(df, t, now)

    if df['symbol'][0] == 'NIFTY23JUNFUT':
        return df
            
    ltp = df['Adj Close'][-1]

    net_position = checkPositionsForConsistency(positions,df)
                
    if ('signal' not in df.columns or 'position' not in df.columns):
        logging.error('signal or position does not exist in dataframe')
        logging.error(df)
        return
            
    if (buyCondition(df,net_position)): 
        #Go Long --  Sell Put AND buy back any pre-sold Calls
        qToExit = \
            exitCurrentPosition(t,positions,net_position,1)           
        if stock:
            ki.nse_buy(kite,t,qToExit=qToExit,tag='main') 
        if options:
            tput,tput_lot_size,tput_tick_size,strike = db.get_option_ticker(t, ltp, 'PE',kite)
            if utils.isOption(t):
                #ticker is itself an option; just buy it
                print(f"Buying {t} with {qToExit} additional to Exit")
                ki.nfo_buy(kite,tput,lot_size=tput_lot_size,
                           tick_size=tput_tick_size,qToExit=qToExit,
                           betsize=bet_size,tag='main-long')                 
            else:
                #ticker is a stock or future, passed with options = True
                #parameters; so we need to sell put option with this underlyting
                #ticker
                ki.nfo_sell(kite,tput,lot_size=tput_lot_size,
                            tick_size=tput_tick_size,
                            qToExit=qToExit,betsize=bet_size, tag='main-long') 
        tradeNotification("GO LONG", t,ltp,df['signal'][-1],df['position'][-1],net_position)

    
    if (sellCondition(df,net_position)): 
        #Go Short --  Sell Call AND buy back any pre-sold Puts
        qToExit = \
            exitCurrentPosition(t,positions,net_position,-1)           
        if stock:
            ki.nse_sell(kite,t,qToExit=qToExit, tag='main-short')
        if options:
            # No need to check utils.isOption() here.  If it is an option, then 
            #get_option_ticker() will return the same ticker and we need to sell it to 
            #go short anyway
            tcall,tcall_lot_size,tcall_tick_size,strike = db.get_option_ticker(t, ltp, 'CE',kite)
            print(f"Selling {t} with {qToExit} additional to Exit")
            ki.nfo_sell(kite,tcall,lot_size=tcall_lot_size,
                        tick_size=tcall_tick_size,qToExit=qToExit,betsize=bet_size, tag='main-short')
        
        tradeNotification("GO SHORT", t,ltp,df['signal'][-1],df['position'][-1],net_position)
        

    if(exitCondition(df,net_position)):
        posTicker = utils.optionUnderlyingFromTicker(t)
        tradeNotification("EXIT", f"{t}({posTicker})",ltp,df['signal'][-1],df['position'][-1],net_position)
        ki.exit_given_positions(kite,positions[posTicker]['positions'])
    return df

def getTargetEntryExitPoints(df,positions,tradeStartTime=tradingStartTime):
    global tradingStartTime
    
    data = {
        'isTrending': False,
        'isMeanRev': False,
        'trades': {
            'trend': {
                'entry': {
                    'target': None,
                    'stoploss': None,
                    'status': None,
                    'targetOrderId': None,
                    'stoplossOrderId': None
                },
                'exit': {
                    'target': None,
                    'stoploss': None,
                    'status': None,
                    'targetOrderId': None,
                    'stoplossOrderId': None
                }
            },
            'meanRev': {
                'low': {
                    'entry': None,
                    'stoploss': None,
                    'exit': None,
                    'status': None,
                    'targetOrderId': None,
                    'stoplossOrderId': None,
                    'exitOrderId': None
                },
                'high': {
                    'entry': None,
                    'stoploss': None,
                    'exit': None,
                    'status': None,
                    'targetOrderId': None,
                    'stoplossOrderId': None,
                    'exitOrderId': None
                }
            }
        }
    }
    
    if df.empty:
        return 

    # Stuff to hack if this function is called from KitTicker instaed of 
    # time_loop
    global now
    
    applyTickerSpecificCfg(df['symbol'][0]) 

    now = df.index[-1]
    if tradeStartTime is not None:
        tradingStartTime = tradeStartTime
    else: 
        tradingStartTime = df.index[0]
        
    t = df['symbol'][0]
    
    #update moving averages and get signals
    dataPopulators = {
        'daily': [
            signals.populateATR,
            signals.populateRenko,
            signals.populateBB,     
            # signals.populateADX, 
            # signals.populateSuperTrend,
            # signals.populateOBV,
            signals.vwap,
            signals.populateSVP

            # signals.populateCandleStickPatterns
        ], 
        'hourly': [
        ]
    }
    signalGenerators = []
    overrideSignalGenerators = []   
    data = \
        signals.applyIntraDayStrategy(df,dataPopulators,signalGenerators, \
                                overrideSignalGenerators, tradeStartTime=tradingStartTime)
    return data
def Tick(stock,options):

    logging.info("\n\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n")
    logging.info(f'Tick @ {now.strftime("%I:%M:%S %p")}')
    logging.info("\n\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n")
    positions = get_positions()

    frm = now - timedelta(days=1)
    #get the last date we have in the DB for Adani; assume other stocks are 
    #also updated to this date; we need to get data from here to now
    #frm_db = db.next_tick(now)
    
    #if (frm_db >= now):
     #   return

    for t in nifty:

        logging.debug(f"Looping for {t}. frm:{frm} to:{now} ")

        #Get latest minute tick from zerodha
        df = downloader.zget(frm,now,t) 
            
        if (df.empty):
            continue
        
        generateSignalsAndTrade(df,positions,stock,options,frm)
                  
    return positions
    
### MAIN LOOP RUNS 9:15 AM to 3:00 @ 3 PM MIS orders will be auto closed anyway (bad pricing)###
if isTimeLoop:
    while (now.time() >= cfgStartTimeOfDay and now.time() < cfgEndExitTradesOnlyTimeOfDay):
        nxt_tick = now + timedelta(minutes=1) - timedelta(seconds=now.second)

        #Tick during market hours only
        positions = Tick(stock=False, options=True)
            
        now = datetime.datetime.now(ist)
        #Sleep for seconds until the next minute
        slp_time = (nxt_tick - now).total_seconds()
        
        if (now.time() > cfgEndNewTradesTimeOfDay):
            if len(positions) == 0:
                logging.info('In exit window and no more positions to exit. Quitting')
                break
            logging.info(f'Done >>> It is {now.strftime("%I:%M:%S %p")} -- NO MORE NEW POSITION ENTRY; EXITS ONLY -- Sleeping for {slp_time}')
        else:
            logging.info(f'Done >>> It is {now.strftime("%I:%M:%S %p")} Sleeping for {slp_time}')
            
        ki.endOfTick()

        time.sleep(max(slp_time , 0))
        ki.startOfTick()

        #update now
        now = datetime.datetime.now(ist)

    logging.info(f"*** TRADING HOURS OVER ***" )

    #Exit all positions at end of trading
    ki.exit_positions(kite)            

