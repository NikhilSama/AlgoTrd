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
import pytz


# set timezone to IST
ist = pytz.timezone('Asia/Kolkata')


import DownloadHistorical as downloader

tickers = td.get_sp500_tickers()

nifty = td.get_nifty_tickers()
index_tickers = td.get_index_tickers()

#td.get_all_ticker_data()

niftydf = {}
results = pd.DataFrame()

def zget(interval='minute'):
    global niftydf
    end =datetime.now()
    start =end - timedelta(days=60)
    niftydf = {}
    for t in nifty:
        niftydf[t]= downloader.zget(start,end,t,interval) 
        niftydf[t]=downloader.zColsToDbCols(niftydf[t])

def backtest(sl=200,ml=20,bw=2,sbw=2.5,sigGenerator=signals.bollinger_band_cx,
             name="bb-cx",type=1):
    performance = pd.DataFrame()
    global results 
    for t in nifty:
        #print(f"{datetime.now()}runngin {t} {sl} {ml} {bw} {sbw}")
        df = niftydf[t].copy()
        df = sigGenerator(df,sl,ml,bw,sbw)
        tearsheet,tearsheetdf = perf.tearsheet(df)
        tearsheetdf.insert(0, 'sl', sl)
        tearsheetdf.insert(0, 'ml', ml)
        tearsheetdf.insert(0, 'bw', bw)
        tearsheetdf.insert(0, 'sbw', sbw)
        tearsheetdf.insert(0, 'ticker', t)
        tearsheetdf.insert(0, 'type', type)

        performance = pd.concat([performance, tearsheetdf])

    results = pd.concat([results,performance.mean().to_frame().T])
    results.to_csv("Data/backtest/NIFTY-TUNING-BACKTEST.csv")
    performance.to_csv(f"Data/backtest/NS{sl}sl-{ml}ml-{bw}bw-{sbw}sbw-{name}.csv")
    

# def backtest_old(sl=200,ml=20,bw=2,sbw=2.5):
#     performance = pd.DataFrame()
#     global results 
#     for t in nifty:
#         print(f"{datetime.now()} runngin {t} {sl} {ml} {bw} {sbw}")
#         df = niftydf[t].copy()
#         df = signals.bollinger_band_cx(df,sl,ml,bw,sbw)
#         tearsheet,tearsheetdf = perf.tearsheet(df)
#         tearsheetdf.insert(0, 'sl', sl)
#         tearsheetdf.insert(0, 'ml', ml)
#         tearsheetdf.insert(0, 'bw', bw)
#         tearsheetdf.insert(0, 'sbw', sbw)
#         tearsheetdf.insert(0, 'ticker', t)
#         tearsheetdf.insert(0, 'type', 1)

#         performance = pd.concat([performance, tearsheetdf])

#     results = pd.concat([results,performance.mean().to_frame().T])
#     performance.to_csv(f"Data/backtest/NS{sl}sl-{ml}ml-{bw}bw-{sbw}sbw-CX_Super.csv")
    
#     performance = pd.DataFrame()

#     for t in nifty:
#         print(f"runnging2 {t} {sl} {ml} {bw} {sbw}")
#         df = niftydf[t].copy()
#         df = signals.bollinger_band_cx2(df,sl,ml,bw,sbw)
#         tearsheet,tearsheetdf = perf.tearsheet(df)
#         tearsheetdf.insert(0, 'sl', sl)
#         tearsheetdf.insert(0, 'ml', ml)
#         tearsheetdf.insert(0, 'bw', bw)
#         tearsheetdf.insert(0, 'sbw', sbw)
#         tearsheetdf.insert(0, 'ticker', t)
#         tearsheetdf.insert(0, 'type', 2)

#         performance = pd.concat([performance, tearsheetdf])

#     results = pd.concat([results,performance.mean().to_frame().T])
#     performance.to_csv(f"Data/backtest/NS{sl}sl-{ml}ml-{bw}bw-{sbw}sbw-CX.csv")

def combinator_variables():
    for sl in [100,200,300]:
        print(f'{datetime.now(ist).strftime("%I:%M:%S %p")} sl -> {sl}')
        for ml in [10,20,30]:
            print(f'{datetime.now(ist).strftime("%I:%M:%S %p")} ml -> {ml}')
            for bw in [1,2,3]:
                print(f'{datetime.now(ist).strftime("%I:%M:%S %p")} bw -> {bw}')
                for sbw in [2,2.5,3,3,5,4]:
                    print(f'{datetime.now(ist).strftime("%I:%M:%P %p")} sbw -> {sbw}')
                    backtest(sl,ml,bw,sbw,signals.bollinger_band_cx,"bb-cx",1)
                    backtest(sl,ml,bw,sbw,signals.bollinger_band_cx2,"bb-cx-basis",2)
                    backtest(sl,ml,bw,sbw,signals.bollinger_band_cx_w_flat_superTrend,"bb-cx-flatst",3)
                    backtest(sl,ml,bw,sbw,signals.bollinger_band_cx_w_flat_superTrend2,"bb-cx-flatst-basis",4)
                    backtest(sl,ml,bw,sbw,signals.bollinger_band_cx_w_basis_breakout,"bb-cx-stbreakout",5)
                    
def combinator_interval():
    for i in ['minute','5minute','15minute','30minute','60minute']:
        zget(i)
        backtest()
        
def groupByDay(df):
    # Split the DataFrame by day using the 'groupby' function and list comprehension
    dfs_by_day = [group[1] for group in df.groupby(pd.Grouper(freq='D'))]

    # Print the split DataFrames
    for i, df_by_day in enumerate(dfs_by_day):
        print(f'DataFrame for Day {i+1}:')
        print(df_by_day)

#combinator()
#combinator_interval()
zget()
combinator_variables()
# backtest(200,20,2,2.5,signals.bollinger_band_cx,"bb-cx",1)
# backtest(200,20,2,2.5,signals.bollinger_band_cx2,"bb-cx-basis",2)
# backtest(200,20,2,2.5,signals.bollinger_band_cx_w_flat_superTrend,"bb-cx-super",3)
print(results)
results.to_csv("Data/backtest/NIFTY-TUNING-BACKTEST.csv")
# sl = 100
# ml = 10
# bw = 3
# sbw = 2
# t = 'WIPRO'
# df = td.get_ticker_data(t, start,end, interval='5min',incl_options=False)
#df = signals.bollinger_band_cx(df,sl,ml,bw,sbw)
# tearsheet,tearsheetdf = perf.tearsheet(df)
