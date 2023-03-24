#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed Mar  1 19:42:07 2023

@author: nikhilsama
"""

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


def sigterm_handler(_signo, _stack_frame):
    # Raises SystemExit(0):
    print("SIGTERM -- EXITING POSITIONS")
    ki.exit_positions(kite)            
    sys.exit(0)
signal.signal(signal.SIGTERM, sigterm_handler)

## TEST FREEZESR START

from freezegun import freeze_time
freezer = freeze_time("Mar 24nd, 2023 11:00:00+0530", tick=True)
freezer.start()

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
time.sleep(60-now.second)
now = datetime.datetime.now(ist)

print(f'Tick @ {now.strftime("%I:%M %p")}')
logging.info(f'>>>>>>>>NEW RUN @ {now.strftime("%I:%M:%S %p")}>>>>>>>')
logging.debug(f">>>>>>>>DEBUG IS ON>>>>>>>")

db = DBBasic() 

nifty = td.get_nifty_tickers()
kite, kws = ki.initKiteTicker()


def Tick():

    logging.info("\n\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n")
    logging.info(f'Tick @ {now.strftime("%I:%M:%S %p")}')
    logging.info("\n\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n")
    positions = ki.get_positions(kite)

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
        if (len(df) == 0):
            continue
        df = downloader.zColsToDbCols(df)
        
        #Get data from db
        #df = td.get_ticker_data(t, frm_ma, now, incl_options=False)
        
        #update moving averages and get signals
        df = signals.bollinger_band_cx(df,startTime=startTime)
        df = perf.calculate_positions(df,close_at_end=False)
        
        downloader.cache_df(df, t, frm, now)
            
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
            if (df['position'][-1] != net_position):
                if (df['signal'][-1] != net_position):
                    logging.info(f"{t}: Exiting all Positions.  Live Kite positions({positions[t]['net_position']} inconsistant with DF pos:{df['position'][-1]} signal: {df['signal'][-1]} ")
                    ki.exit_given_positions(kite,positions[t]['positions'])
                    del positions[t]
#        elif (not math.isnan(df['signal'][-1])) and df['position'][-1] != 0:
        elif df['position'][-1] != 0:
                logging.info(f"{t}: No Live Kite positions inconsistant with DF ({df['position'][-1]})")
            
        # Get Put / CALL option tickerS
        tput,tput_lot_size,tput_tick_size = db.get_option_ticker(t, ltp, 'PE')
        tcall,tcall_lot_size,tcall_tick_size = db.get_option_ticker(t, ltp, 'CE')
        
        if ('signal' not in df.columns or 'position' not in df.columns):
            logging.error('signal or position does not exist in dataframe')
            logging.error(df)
            continue
        # if (math.isnan(df['signal'][-1])):
        #     continue
        
        #place orders
        if (df['signal'][-1] == 1 and df['position'][-1] != 1) or (df['position'][-1] == 1 and net_position != 1): 
            #Go Long --  Sell Put AND buy back any pre-sold Calls
            logging.info(f"GO LONG {t} LastCandleClose:{ltp}")
            if t in positions: 
                ki.exit_given_positions(kite,positions[t]['positions'])
#            ki.exit_positions(kite,t,tput_lot_size,tput_tick_size)            
            ki.nse_buy(kite,t)
            #ki.nfo_sell(kite,tput,tput_lot_size,tput_tick_size)

        if (df['signal'][-1] == -1 and df['position'][-1] != -1) or (df['position'][-1] == -1 and net_position != -1): 
            #Go Short --  Sell Call AND buy back any pre-sold Puts
            logging.info(f"GO SHORT {t} LastCandleClose:{ltp}" )
            if t in positions: 
                ki.exit_given_positions(kite,positions[t]['positions'])
#           ki.exit_positions(kite,t,tput_lot_size,tput_tick_size)            
            ki.nse_sell(kite,t)
            #ki.nfo_sell(kite,tcall,tcall_lot_size,tcall_tick_size)
        
        if(df['signal'][-1] == 0 and df['position'][-1] != 0):
            ki.exit_positions(kite,t,tput_lot_size,tput_tick_size)            
        
### MAIN LOOP RUNS 9:15 AM to 3:00 @ 3 PM MIS orders will be auto closed anyway (bad pricing)###

while (now.hour >= 9 and now.hour < 15):
    nxt_tick = now + timedelta(minutes=1) - timedelta(seconds=now.second)
    
    #Tick during market hours only
    Tick()
        
    now = datetime.datetime.now(ist)
    #Sleep for seconds until the next minute
    slp_time = (nxt_tick - now).total_seconds()
    
    logging.info(f'Done >>> It is {now.strftime("%I:%M:%S %p")} Sleeping for {slp_time}')
    
    ki.endOfTick()

    time.sleep(max(slp_time , 0))
    ki.startOfTick()

    #update now
    now = datetime.datetime.now(ist)
    
#Exit all positions at end of trading
ki.exit_positions(kite)            

