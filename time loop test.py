#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 18:24:32 2023

@author: nikhilsama
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 19:42:07 2023

@author: nikhilsama
"""

import time
from datetime import date,timedelta,timezone,datetime
import tickerdata as td
import numpy as np
import signals as signals
import pandas as pd
import DownloadHistorical as downloader
import kite_init as ki 
from DatabaseLogin import DBBasic
from freezegun import freeze_time

freezer = freeze_time("Mar 10th, 2023 09:02:00", tick=True)
freezer.start()
print(f"Tick @ {datetime.now(timezone.utc).isoformat()}")

db = DBBasic() 


nifty = td.get_nifty_tickers()
kite, kws = ki.initKiteTicker()

def Tick():
    #print(f"Tick @ {datetime.now(timezone.utc).isoformat()}")
    
    positions = {}

    for t in nifty:

        to = datetime.now()
        frm = to - timedelta(minutes=2)
        #Get latest minute tick from zerodha
        downloader.zget(frm,to,t) 
        
        #Get data from db
        frm = to - timedelta(minutes=200)
        start =date.today() - timedelta(200)
        df = td.get_ticker_data(t, frm,to, incl_options=False)
        
        #update moving averages and get signals
        df = signals.bollinger_band_cx(df)
        
        ltp = df['Adj Close'][-1]
        
        # Get Put / CALL option tickerS
        tput = db.get_option_ticker(t, ltp, 'PE')
        tcall = db.get_option_ticker(t, ltp, 'PE')
        
        #place orders
        if (df['signal'][-1] == 1 and df['position'][-1] != 1): 
            #Go Long --  Sell Put AND buy back any pre-sold Calls
            
            positions[t]['sold_put_order_Id'] = ki.nfo_sell(kite,tput)
            positions[t]['sold_put_ticker'] = tput #Store this ticker as it may change

            if (positions[t]['sold_call_order_Id'] > 0): 
                # EXIT SOLD SHORT POSITIONS 
                ki.nfo.buy(kite,positions[t]['sold_call_ticker'])
                positions[t]['sold_call_ticker'] = ''
                positions[t]['sold_call_order_Id'] = 0

        if (df['signal'][-1] == -1 and df['position'][-1] != -1): 
            #Go Short --  Sell Call AND buy back any pre-sold Puts
            positions[t]['sold_call_order_Id'] = ki.nfo_sell(kite,tcall)
            positions[t]['sold_call_ticker'] = tcall #Store this ticker as it may change

            if (positions[t]['sold_put_order_Id'] > 0): 
                # EXIT SOLD LONG POSITIONS 
                ki.nfo.buy(kite,positions[t]['sold_put_ticker'])
                positions[t]['sold_put_ticker'] = ''
                positions[t]['sold_put_order_Id'] = 0


### MAIN LOOP RUNS 9:15 AM to 3:15###

hr = datetime.now().hour
mnt = datetime.now().minute

while (hr >= 9 and hr < 15) or (hr == 15 and mnt < 30):
    #Tick during market hours only
    Tick()
        
    #Sleep for seconds until the next minute
    slp_time = 60 - (time.time() % 60)
    time.sleep(slp_time)

    #update hr and minute for the next loop
    hr = datetime.now().hour
    mnt = datetime.now().minute


