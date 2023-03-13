#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 19:42:07 2023

@author: nikhilsama
"""

import log_setup
import time
from datetime import date,timedelta,timezone,datetime
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

## TEST FREEZESR START

from freezegun import freeze_time
freezer = freeze_time("Mar 10th, 2023 13:00:00+0530", tick=True)
freezer.start()

## END 

# set timezone to IST
ist = pytz.timezone('Asia/Kolkata')

# get current datetime in IST
now = datetime.now(ist)
print(f"Tick @ {now}")
logging.info(f">>>>>>>>NEW RUN @ {now}>>>>>>>")
logging.debug(f">>>>>>>>DEBUG IS ON>>>>>>>")

db = DBBasic() 

nifty = td.get_nifty_tickers()
kite, kws = ki.initKiteTicker()


def Tick():

    logging.info("\n\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n")
    logging.info(f"Tick @ {now}")
    logging.info("\n\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n")
    positions = {}

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
        df = downloader.zColsToDbCols(df)
        
        #Get data from db
        #df = td.get_ticker_data(t, frm_ma, now, incl_options=False)
        
        if (len(df) <2):
            continue
        
        #update moving averages and get signals
        df = signals.bollinger_band_cx(df)
        df = perf.calculate_positions(df)
        
        ltp = df['Adj Close'][-1]
        
        # Get Put / CALL option tickerS
        tput,tput_lot_size,tput_tick_size = db.get_option_ticker(t, ltp, 'PE')
        tcall,tcall_lot_size,tcall_tick_size = db.get_option_ticker(t, ltp, 'CE')
        
        if ('signal' not in df.columns or 'position' not in df.columns):
            logging.error('signal or position does not exist in dataframe')
            logging.error(df)
            continue
        
        #place orders
        if (df['signal'][-1] == 1 and df['position'][-1] != 1): 
            #Go Long --  Sell Put AND buy back any pre-sold Calls
            ki.nse_exit(kite,t)
            ki.nse_buy(kite,t)
           # ki.nfo_exit(kite,t,tput_lot_size,tput_tick_size)            
           # ki.nfo_sell(kite,tput,tput_lot_size,tput_tick_size)

        if (df['signal'][-1] == -1 and df['position'][-1] != -1): 
            #Go Short --  Sell Call AND buy back any pre-sold Puts
            ki.nse_exit(kite,t)
            ki.nse_sell(kite,t)
          #  ki.nfo_exit(kite,t,tcall_lot_size,tcall_tick_size)            
          #  ki.nfo_sell(kite,tcall,tcall_lot_size,tcall_tick_size)
        
        
### MAIN LOOP RUNS 9:15 AM to 3:15###

while (now.hour >= 9 and now.hour < 15) or (now.hour == 15 and now.minute < 30):
    nxt_tick = now + timedelta(minutes=1)

    #Tick during market hours only
    Tick()
        
    now = datetime.now(ist)
    #Sleep for seconds until the next minute
    slp_time = (nxt_tick - now).total_seconds()
    
    logging.info(f"Done >>> It is {now} Sleeping for {slp_time}")
    
    ki.endOfTick()

    time.sleep(max(slp_time , 0))
    ki.startOfTick()

    #update now
    now = datetime.now(ist)

