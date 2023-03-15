#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 10:03:29 2023

@author: nikhilsama
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 20:18:29 2023

@author: nikhilsama
"""

from datetime import date,timedelta,datetime
import tickerdata as td
import performance as perf
import numpy as np
import signals as signals
import pandas as pd
import matplotlib.pyplot as plt
import pprint

import DownloadHistorical as downloader

tickers = td.get_sp500_tickers()

nifty = td.get_nifty_tickers()
index_tickers = td.get_index_tickers()

#td.get_all_ticker_data()
end =datetime.now()
start =end - timedelta(days=1)

niftydf = {}

for t in nifty:
    niftydf[t]= downloader.zget(start,end,t) 
    niftydf[t]=downloader.zColsToDbCols(niftydf[t])

results = pd.DataFrame()

def backtest(sl,ml,bw,sbw):
    performance = pd.DataFrame()
    global results 
    for t in nifty:
        df = niftydf[t]
        df = signals.bollinger_band_cx_w_flat_superTrend(df,sl,ml,bw,sbw)
        tearsheet,tearsheetdf = perf.tearsheet(df)
        tearsheetdf.insert(0, 'sl', sl)
        tearsheetdf.insert(0, 'ml', ml)
        tearsheetdf.insert(0, 'bw', bw)
        tearsheetdf.insert(0, 'sbw', sbw)
        tearsheetdf.insert(0, 'ticker', t)
        tearsheetdf.insert(0, 'type', 1)

        print(f"{datetime.now()}runngin {t} {sl} {ml} {bw} {sbw}")
        performance = pd.concat([performance, tearsheetdf])

    results = pd.concat([results,performance.mean().to_frame().T])
    performance.to_csv(f"Data/backtest/NS{sl}sl-{ml}ml-{bw}bw-{sbw}sbw-CX_Super.csv")
    
    for t in nifty:
        df = niftydf[t]
        df = signals.bollinger_band_cx(df,sl,ml,bw,sbw)
        tearsheet,tearsheetdf = perf.tearsheet(df)
        tearsheetdf.insert(0, 'sl', sl)
        tearsheetdf.insert(0, 'ml', ml)
        tearsheetdf.insert(0, 'bw', bw)
        tearsheetdf.insert(0, 'sbw', sbw)
        tearsheetdf.insert(0, 'ticker', t)
        tearsheetdf.insert(0, 'type', 2)
        print(f"runnging2 {t} {sl} {ml} {bw} {sbw}")

        performance = pd.concat([performance, tearsheetdf])

    results = pd.concat([results,performance.mean().to_frame().T])
    performance.to_csv("Data/backtest/NS{sl}sl-{ml}ml-{bw}bw-{sbw}sbw-CX.csv")
    
def combinator():
    for sl in [100,200,300]:
        print(sl)
        for ml in [10,20,30,50]:
            print(ml)
            for bw in [1,2,3]:
                print(bw)
                for sbw in [2,2.5,3,3,5,4]:
                    print(sbw)
                    backtest(sl,ml,bw,sbw)

def groupByDay(df):
    # Split the DataFrame by day using the 'groupby' function and list comprehension
    dfs_by_day = [group[1] for group in df.groupby(pd.Grouper(freq='D'))]

    # Print the split DataFrames
    for i, df_by_day in enumerate(dfs_by_day):
        print(f'DataFrame for Day {i+1}:')
        print(df_by_day)

#combinator()
backtest(200,20,2,2.5)
#results.to_csv("Data/backtest/NIFTY-TUNING-BACKTEST.csv")
# sl = 100
# ml = 10
# bw = 3
# sbw = 2
# t = 'WIPRO'
# df = td.get_ticker_data(t, start,end, interval='5min',incl_options=False)
#df = signals.bollinger_band_cx(df,sl,ml,bw,sbw)
# tearsheet,tearsheetdf = perf.tearsheet(df)
