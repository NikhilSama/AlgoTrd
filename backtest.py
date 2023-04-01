#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 20:18:29 2023

@author: nikhilsama
"""

from datetime import date,datetime,timedelta
import time
import tickerdata as td
import performance as perf
import numpy as np
import signals as signals
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import pprint
import DownloadHistorical as downloader
import pytz
import strategies15m as strat15m
import ppprint
from plotting import plot_backtest
import backtest_log_setup

#cfg has all the config parameters make them all globals here
import cfg
globals().update(vars(cfg))

tickers = td.get_sp500_tickers()
nifty = td.get_nifty_tickers()
index_tickers = td.get_index_tickers()

# set timezone to IST
ist = pytz.timezone('Asia/Kolkata')

def zget(t,s,e,i):
    #Get latest minute tick from zerodha
    df = downloader.zget(s,e,t,i,includeOptions=includeOptions)
    df = downloader.zColsToDbCols(df)
    return df

def zgetNDays(t,n,e=datetime.now(ist),i="minute"):
    s = e - timedelta(days=n)
    return zget(t, s, e, i)

def getTotalChange(df):
    return round(100*(df['Open'][0] - df['Adj Close'][-1])/df['Open'][0],2)


def test(t='ADANIENT',i='minute'):
    df = zgetNDays(t,days,i=i)
    df.insert(0, 'i', range(1, 1 + len(df)))
  
    dataPopulators = [signals.populateBB, signals.populateADX]
    signalGenerators = [signals.getSig_BB_CX_ADX_MASLOPE]

    signals.applyIntraDayStrategy(df1,dataPopulators,signalGenerators)

def perfProfiler(name,t):
    print (f"{name} took {round((time.time() - t)*1000,2)}ms")
    return time.time()

def backtest(t,i='minute',exportCSV=False):

    perfTIME = time.time()    
    startingTime = perfTIME
    zgetFrom = datetime(2023, 3, 27, 9, 0, tzinfo=ist)
    zgetTo = datetime(2023, 3, 31, 17, 30, tzinfo=ist)
    df = zget(t,zgetFrom,zgetTo,i=i)
    if df.empty:
        print(f"No data foc {t}")
        return
    #df = zgetNDays(t,days,i=i)
    perfTime = perfProfiler("ZGET", perfTIME)
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
    perfTIME = perfProfiler("SIGNAL GENERATION", perfTIME)


    tearsheet,tearsheetdf = perf.tearsheet(df)
    print(f'Total Return: {tearsheet["return"]*100}%')
    print(f'Num Trades: {tearsheet["num_trades"]}')
    print(f'Avg Return Per Trade: {tearsheet["std_dev_pertrade_return"]*100}%')
    print(f'Std Dev of Returns: {tearsheet["return"]}')
    print(f'Skewness: {tearsheet["skewness_pertrade_return"]}')
    print(f'Kurtosis: {tearsheet["kurtosis_pertrade_return"]}')
    perfTIME = perfProfiler("Tearsheet took", perfTIME)

    if (exportCSV == True):
        df.to_csv("export.csv")
    perfTIME = perfProfiler("to CSV", perfTIME)
    perfTIME = perfProfiler("TOTAL", startingTime)

    if (plot == False):
        return
    plot_backtest(df,tearsheet['trades'])
    
    # print (f"END Complete {datetime.now(ist)}")
    return tearsheetdf

# Plot the graph of closing prices for the array of tickers provided
# and the interval provided and the number of days provided
def plot_options(uticker, tickers,i='minute', 
         days=30, e=datetime.now(ist)):
    df={}
    j=0
    color=['blue','green','red','orange']
    fig, (ax1,ax2) = plt.subplots(2, 1, figsize=(8, 8))

    udf = zgetNDays(uticker,days,i=i,e=e)
    udf['pct_change'] = udf['Adj Close'].pct_change()
    udf['cum_pct_change']=((1 + udf['pct_change']).cumprod() - 1)*100
    udf.insert(0, 'i', range(1, 1 + len(udf)))
    ax1.plot(udf['i'], udf['Adj Close'], color='red', linewidth=2)
    legend = []
    for t in tickers:
        print(t)
        df[t] = zgetNDays(t,days,i=i,e=e)
        df[t]['pct_change'] = ldf[t]['Adj Close'].pct_change()
        df[t]['cum_pct_change']=((1 + df[t]['pct_change']).cumprod() - 1)*100/20
        df[t].insert(0, 'i', range(1, 1 + len(df[t])))
        ax2.plot(df[t]['i'], df[t]['cum_pct_change'], color=color[j], linewidth=2)
        j = j+1
        legend.append(t)

    plt.legend(legend,loc='upper right')
    plt.show()
    
def compareDayByDayPerformance(t,days=90):
    i = 0
    while i<days:
        i=i+1
        s = datetime.now(ist)-timedelta(days=i)
        df = zgetNDays(t,days,s)
        if(len(df)):
            df = signals.bollinger_band_cx(df)
            tearsheet,tearsheetdf = perf.tearsheet(df)
            change = getTotalChange(df)
            ret = round(tearsheet['return'] *100,2)
            print(f"{t} Day:{s} Return:{ret}% Change; {change}%")

# def backtestCombinator():
#     ma_slope_threshes = [0.5, 1, 1.5]
#     ma_slope_thresh_yellow_multipliers = [0.5, 0.7, 0.9]
#     ma_slope_slope_threshes = [0.005, 0.01, 0.015]
#     obv_osc_threshes = [0.1, 0.2, 0.3]
#     obv_osc_thresh_yellow_multipliers = [0.5,0.7,0.9]
#     obv_osc_slope_threshes = [0.1,0.3,0.5]
#     override_multipliers = [1,1.2,1.4]
#     for params in itertools.product(ma_slope_threshes, ma_slope_thresh_yellow_multipliers, ma_slope_slope_threshes,
#                                 obv_osc_slope_threshes, obv_osc_threshes, obv_osc_thresh_yellow_multipliers, override_multipliers):
        ma_slope_thresh, ma_slope_thresh_yellow_multiplier, ma_slope_slope_thresh, obv_osc_thresh, obv_osc_thresh_yellow_multiplier, ovc_osc_slope_thresh, \
            override_multiplier = params        
        # This loop will run 3^7 = 2187 times; each run will be about 
        # 1 second, so total 2187 seconds = 36 minutes
        # run a combo 
        
        # when done w for loop write results to csv and mark done in db 
        # add to fname string and csv file "maSlopeThresh:{ma_slope_thresh} maSlopeThreshYellowMultiplier:{ma_slope_thresh_yellow_multiplier} maSlopeSlopeThresh:{ma_slope_slope_thresh} obvOscThresh:{obv_osc_thresh} obvOscThreshYellowMultiplier:{obv_osc_thresh_yellow_multiplier} obvOscSlopeThresh:{ovc_osc_slope_thresh} overrideMultiplier:{override_multiplier}"
        # continue to next combo
        # output fname and csv row to contain timeframe in 
        # start and end times, symbol, and all the parameters
        
        
#plot_options(['ASIANPAINT'],10,'minute')
#backtest('HDFCLIFE','minute',adxThreh=30)
backtest('INFY','minute')
#backtest('HDFCLIFE','minute',adxThreh=25)
#backtest('ASIANPAINT','minute',adxThreh=25)
#backtest('HDFCLIFE','minute',adxThreh=30)
#backtest('ADANIENT','minute',adxThreh=30)
#compareDayByDayPerformance('ONGC')
 
#plot('INFY',['ASIANPAINT23MAR2840PE','ASIANPAINT23MAR2840CE'],i='minute', days=3,e=datetime.now(ist)-timedelta(days=15))   

    # print hello
# print hello
