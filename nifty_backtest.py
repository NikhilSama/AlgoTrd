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
import random
import cfg
globals().update(vars(cfg))


# set timezone to IST
ist = pytz.timezone('Asia/Kolkata')


import DownloadHistorical as downloader

tickers = td.get_sp500_tickers()

nifty = td.get_nifty_tickers()
nifty_active = td.get_fo_active_nifty_tickers()
index_tickers = td.get_index_tickers()

#td.get_all_ticker_data()

niftydf = {}
results = pd.DataFrame()

# generate a random RGB color tuple for each series
colors = []
for i in range(50):
    r = random.random()
    g = random.random()
    b = random.random()
    colors.append((r, g, b))
legend = []

def zget(interval='minute'):
    global niftydf
#    end =datetime.now()
    end = datetime(2023, 3, 29, 15, 30, tzinfo=ist)
    start =end - timedelta(days=days)
    niftydf = {}
    for t in nifty_active:
        niftydf[t]= downloader.zget(start,end,t,interval,includeOptions=includeOptions) 
        niftydf[t]=downloader.zColsToDbCols(niftydf[t])

def backtest(type=1, name='test'):
    zget()
    performance = pd.DataFrame()
    global results 
    x = 0
    for t in nifty_active:
        #print(f"{datetime.now()}runngin {t} {sl} {ml} {bw} {sbw}")
        df = niftydf[t].copy()
        
        if df.empty:
            print(f"{t} is empty. Skipping.")
            continue
        
        dataPopulators = [signals.populateBB, signals.populateADX, signals.populateOBV]
        signalGenerators = [signals.getSig_BB_CX
                      ,signals.getSig_ADX_FILTER
                       ,signals.getSig_MASLOPE_FILTER
                       ,signals.getSig_OBV_FILTER
                       ,signals.getSig_exitAnyExtremeADX_OBV_MA20_OVERRIDE
                       ,signals.getSig_followAllExtremeADX_OBV_MA20_OVERRIDE
                       ,signals.exitTrendFollowing
                       ]
        overrideSignalGenerators = []
        signals.applyIntraDayStrategy(df,dataPopulators,signalGenerators,
                                  overrideSignalGenerators)


        tearsheet,tearsheetdf = perf.tearsheet(df)
        # tearsheetdf.insert(0, 'sl', sl)
        # tearsheetdf.insert(0, 'ml', ml)
        # tearsheetdf.insert(0, 'bw', bw)
        # tearsheetdf.insert(0, 'sbw', sbw)
        # tearsheetdf.insert(0, 'ticker', t)

        df = df[df['cum_strategy_returns']!=0]
        if plot:
            plt.plot(df['i'], df['cum_strategy_returns'], 
                     color=colors[x])
            x = x + 1
            legend.append(t)
        performance = pd.concat([performance, tearsheetdf])
    
    perfSummary = performance.mean()
    perfSummary['num_trades'] = performance['num_trades'].sum()
    perfSummary['num_winning_trades'] = performance['num_winning_trades'].sum()
    perfSummary['num_losing_trades'] = performance['num_losing_trades'].sum()
    perfSummary['std_dev_across_stocks'] = performance['return'].std()
    perfSummary.rename(
                {'std_dev_pertrade_return':'average_of_per_ticker_std_dev_across_trades'}, inplace=True)
    perfSummary = perfSummary.to_frame().T

    # Get command-line arguments and add them to the Series
    perfSummaryfname = 'niftyPerf-'
    args = sys.argv[1:]
    for arg in args:
        key, value = arg.split(':')
        perfSummary[key] = value
        perfSummaryfname = perfSummaryfname + '-' + key + '-' + value
    #results = pd.concat([results,perfSummary.to_frame().T])
    #results.to_csv("Data/backtest/NIFTY-TUNING-BACKTEST.csv")
    perfSummary.to_csv(f"Data/backtest/nifty/{perfSummaryfname}.csv")
    
    if plot:
        plt.legend(legend,loc='upper right')
        # set the x-axis scale to 'symlog'
        plt.show()

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
# combinator_variables()
dataPopulators = [signals.populateBB, signals.populateADX, signals.populateOBV]
signalGenerators = [signals.getSig_BB_CX
                    ,signals.getSig_ADX_FILTER
                    ,signals.getSig_MASLOPE_FILTER
                    ,signals.getSig_OBV_FILTER
                    ]
           
#backtest(200,20,2,2.5,signals.bollinger_band_cx,"bb-cx",1)
backtest(2, 'BB-CX-ADX30-MASLOPE1-OBV.25')
# backtest(200,20,2,2.5,signals.bollinger_band_cx2,"bb-cx-basis",2)
# backtest(200,20,2,2.5,signals.bollinger_band_cx_w_flat_superTrend,"bb-cx-super",3)
#print(results)
#results.to_csv("Data/backtest/NIFTY-TUNING-BACKTEST.csv")
# sl = 100
# ml = 10
# bw = 3
# sbw = 2
# t = 'WIPRO'
# df = td.get_ticker_data(t, start,end, interval='5min',incl_options=False)
#df = signals.bollinger_band_cx(df,sl,ml,bw,sbw)
# tearsheet,tearsheetdf = perf.tearsheet(df)
