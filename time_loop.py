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
    ki.exit_positions(kite)            
    sys.exit(0)
signal.signal(signal.SIGTERM, sigterm_handler)

## TEST FREEZESR START

if cfgFreezeGun:
    from freezegun import freeze_time
    freezeTime = "Mar 28nd, 2023 10:05:00+0530"
    freezer = freeze_time(freezeTime, tick=True)
    freezer.start()
    print ("Freeze gun is on. Time is frozen at: ", freezeTime)
    ## END 

# set timezone to IST
ist = pytz.timezone('Asia/Kolkata')


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
startTime = datetime.datetime.now(ist)
# startTime = datetime.datetime(2000,1,1,10,0,0) #Long ago :-)
# startTime = ist.localize(startTime)
now = startTime
sleep_till_10am()

#sleep a few seconds to ensure we kick off on a rounded minute
#time.sleep(60-now.second)
now = datetime.datetime.now(ist)

print(f'Tick @ {now.strftime("%I:%M %p")}')
logging.info(f'>>>>>>>>NEW RUN @ {now.strftime("%I:%M:%S %p")}>>>>>>>')
logging.debug(f">>>>>>>>DEBUG IS ON>>>>>>>")

db = DBBasic() 

nifty = td.get_fo_active_nifty_tickers()

kite = ki.initKite()

def get_positions():
    return ki.get_positions(kite)

# BUY CONDITION
# Buy if current signal is 1 and position is not 1, 
# or if position is 1 and net position is not 1 as long as signal is not -1
# Logic for the second one is a missed candle or a missed trade(order not executed) 
# can lead to position being 1, and net_postion not 1; we shoudl fix this
# UNLESS the signal is -1 or 0 (nan or 1 is ok), in which case we should not, since we are going short this tick anyway

def buyCondition(df,net_position):
    
    return (df['signal'][-1] == 1 and df['position'][-1] != 1) or \
            (df['position'][-1] == 1 and net_position != 1 and df['signal'][-1] != -1 and df['signal'][-1] != 0)

# SELL CONDITION
# Sell if current signal is -1 and position is not -1,
# or if position is -1 and net position is not -1 as long as signal is not 1
# Logic for the second one is a missed candle or a missed trade(order not executed)
# can lead to position being -1, and net_postion not -1; we shoudl fix this
# UNLESS the signal is 1 or 0 (nan or -1 is ok), in which case we should not, since we are going long this tick anyway

def sellCondition(df,net_position):
    return (df['signal'][-1] == -1 and df['position'][-1] != -1) or \
            (df['position'][-1] == -1 and net_position != -1 and df['signal'][-1] != 1 and df['signal'][-1] != 0)

# EXIT CONDITION
# Exit if there is a current position (net_position != 0) 
# and current signal is 0 or current signal is nan and position is 0 (missed candle or missed trade)
def exitCondition(df,net_position):
    return (net_position != 0) and \
        ((df['signal'][-1] == 0 and df['position'][-1] != 0) or \
        (df['position'][-1] == 0 and math.isnan(df['signal'][-1])))

def exitCurrentPosition(t,positions,net_position,nextPosition):
    qToExit = 0
    if t in positions: 
        #if multiple position exists, exit all
        if (len(positions[t]['positions']) > 1):
            logging.error("More than one position exists for {t}. Exiting all")
            ki.exit_given_positions(kite,positions[t]['positions'])
        else:

            if (net_position == -1 and nextPosition == 1) or \
                (net_position == 1 and nextPosition == -1):
                #Exit Options positions, because we always engage in options by selling
                #ie: if we are short a stock, we sold a call option; we dont reverse
                #positions by buying 2 calls; instead we sell the call and then sell
                #a put;  So we need to exit any options positions here if the 
                #net position is not consistant with nextPosition
                ki.exitNFOPositionsONLY(kite,positions[t]['positions'])
                #For Equity positions we can reverse simply by doubling the buy when 
                # we change direction; so we dont need to exit equity positions here
                qToExit = positions[t]['positions'][0]['quantity']# We have a short position, so we need to exit both
    return qToExit


def tradeNotification(type, t,ltp,signal,position,net_position):
    subprocess.call(["afplay", '/System/Library/Sounds/Glass.aiff'])
    logging.info(f"GO {type} {t} LastCandleClose:{ltp} Signal:{signal} Position:{position} NetPosition:{net_position}" )
    if showTradingViewLive:
        global liveTVThread 
        if liveTVThread is not None: 
            liveTVThread.join()
        liveTVThread = threading.Thread(target=liveCharts.loadChart, 
                                                args=(t,)) # funky syntax needs (t,) if only one arg to signify its a tuple
        liveTVThread.start()
            #liveCharts.loadChart(t)

def generateSignalsAndTrade(df,positions,stock,options,dfStartTime=None):
    # Stuff to hack if this function is called from KitTicker instaed of 
    # time_loop
    global now
    if dfStartTime is not None:
        now = df.index[-1]
        dfStartTime = df.index[0]

    t = df['symbol'][0]

    #update moving averages and get signals
    dataPopulators = [signals.populateBB, signals.populateADX, signals.populateOBV]
    signalGenerators = [signals.getSig_BB_CX
                        ,signals.getSig_ADX_FILTER
                        ,signals.getSig_MASLOPE_FILTER
                        ,signals.getSig_OBV_FILTER
                        ,signals.getSig_exitAnyExtremeADX_OBV_MA20_OVERRIDE
                        ,signals.getSig_followAllExtremeADX_OBV_MA20_OVERRIDE
                                
                        ]
    overrideSignalGenerators = []   
    signals.applyIntraDayStrategy(df,dataPopulators,signalGenerators,
                                overrideSignalGenerators)
    df = perf.calculate_positions(df,close_at_end=False)
    
    downloader.cache_df(df, t, dfStartTime, now)
        
    ltp = df['Adj Close'][-1]
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
    if t in positions:
        net_position = positions[t]['net_position']
        # net_position can be 1,-1 or 'inconsistent'
        # if inconsistent, then we need to exit all positions
        if (np.isnan(df['signal'][-1]) and df['position'][-1] != net_position):
#                if (df['signal'][-1] != net_position):
            logging.info(f"{t}: Exiting all Positions.  Live Kite positions({positions[t]['net_position']} inconsistant with DF pos:{df['position'][-1]} signal: {df['signal'][-1]} ")
            ki.exit_given_positions(kite,positions[t]['positions'])
            net_position =0
            del positions[t]
#        elif (not math.isnan(df['signal'][-1])) and df['position'][-1] != 0:
    elif df['position'][-1] != 0:
            logging.info(f"{t}: No Live Kite positions inconsistant with DF ({df['position'][-1]})")
        
    # Get Put / CALL option tickerS
    if options:
        tput,tput_lot_size,tput_tick_size = db.get_option_ticker(t, ltp, 'PE')
        tcall,tcall_lot_size,tcall_tick_size = db.get_option_ticker(t, ltp, 'CE')
    
    if ('signal' not in df.columns or 'position' not in df.columns):
        logging.error('signal or position does not exist in dataframe')
        logging.error(df)
        return
            
    if (buyCondition(df,net_position)): 
        #Go Long --  Sell Put AND buy back any pre-sold Calls
        qToExit = \
            exitCurrentPosition(t,positions,net_position,1)           
        if stock:
            ki.nse_buy(kite,t,qToExit=qToExit) 
        if options:
            ki.nfo_sell(kite,tput,tput_lot_size,tput_tick_size,doubleQtoExit=False) 
        tradeNotification("LONG", t,ltp,df['signal'][-1],df['position'][-1],net_position)

    
    if (sellCondition(df,net_position)): 
        #Go Short --  Sell Call AND buy back any pre-sold Puts
        qToExit = \
            exitCurrentPosition(t,positions,net_position,-1)           
        if stock:
            ki.nse_sell(kite,t,qToExit=qToExit)
        if options:
            ki.nfo_sell(kite,tcall,tcall_lot_size,tcall_tick_size,doubleQtoExit=False)
        
        tradeNotification("SHORT", t,ltp,df['signal'][-1],df['position'][-1],net_position)
        

    if(exitCondition(df,net_position)):
        logging.info(f"EXITING {t} LastCandleClose:{ltp} Signal:{df['signal'][-1]} Position:{df['position'][-1]} NetPosition:{net_position}" )
        ki.exit_given_positions(kite,positions[t]['positions'])

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
    while (now.hour >= startHour and now.hour < exitHour):
        nxt_tick = now + timedelta(minutes=1) - timedelta(seconds=now.second)

        #Tick during market hours only
        positions = Tick(stock=True, options=False)
            
        now = datetime.datetime.now(ist)
        #Sleep for seconds until the next minute
        slp_time = (nxt_tick - now).total_seconds()
        
        if (now.hour == endHour):
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

