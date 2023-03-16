#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 20:18:29 2023

@author: nikhilsama
"""

from datetime import date,datetime,timedelta
import tickerdata as td
import performance as perf
import numpy as np
import signals as signals
import pandas as pd
import matplotlib.pyplot as plt
import pprint
import DownloadHistorical as downloader
import pytz

tickers = td.get_sp500_tickers()
nifty = td.get_nifty_tickers()
index_tickers = td.get_index_tickers()

# set timezone to IST
ist = pytz.timezone('Asia/Kolkata')

def zget(t,s,e):
    #Get latest minute tick from zerodha
    df = downloader.zget(s,e,t) 
    df = downloader.zColsToDbCols(df)
    return df

def zgetNDays(t,n,e=datetime.now(ist)):
    s = e - timedelta(days=n)
    return zget(t, s, e)

def getTotalChange(df):
    return round(100*(df['Open'][0] - df['Adj Close'][-1])/df['Open'][0],2)



def backtest(t):

    df = zgetNDays(t,1)
    #df = td.get_ticker_data("NIFTY 50", start,end, incl_options=False)
    #ce_ticker = td.get_option_ticker("RELIANCE", df['Adj Close'][-1], 'CE')
    #ddf = td.get_ticker_data(ce_ticker, start,end)
    
    # Adding new column
    df.insert(0, 'i', range(1, 1 + len(df)))
    #ddf.insert(0, 'i', range(1, 1 + len(ddf)))
    #signals.eom_effect(df)
    #signals.sma50_bullish(df)
    #df = signals.bollinger_band_cx(df)
    
    df = signals.bollinger_band_cx(df)
    
    #df['Open'] = ddf['Open']
    #df['High'] = ddf['High']
    #df['Low'] = ddf['Low']
    #df['Close'] = ddf['Close']
    #df['Adj Close'] = ddf['Adj Close']
    #df.index[9]
    #signals.ema_cx(df)
    #signals.mystrat(df)
    tearsheet,tearsheetdf = perf.tearsheet(df)
    
    #df[['ma_superTrend', 'ma_slow', 'ma_fast']].plot(grid=True, figsize=(12, 8))
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(8, 8))
    
    # plot the first series in the first subplot
    #ax1.plot(df['i'], df['ma_superTrend'], color='green', linewidth=2)
    #ax1.plot(df['i'], df['ma20'], color='gold', linewidth=2)
    ax1.plot(df['i'], df['Adj Close'], color='red', linewidth=2)
    ax1.plot(df['i'], df['lower_band'], color='grey', linewidth=2)
    ax1.plot(df['i'], df['upper_band'], color='grey', linewidth=2)
    ax1.plot(df['i'], df['ma_superTrend'], color='orange', linewidth=4)
    
    # plot the second series in the second subplot
    ax2.plot(df['i'], df['ma_superTrend_pct_change'], color='red', linewidth=2)
    #ax2.plot(df.index, df['superTrend'], color='red', linewidth=2)
    #ax2.plot(df['i'], df['Adj Close-P'], color='blue', linewidth=2)
    #ax2.plot(df['i'], df['Adj Close-C'], color='red', linewidth=2)
    
    ax3.plot(df['i'], df['ma20_pct_change_ma'], color='green', linewidth=2)
    #ax3.plot(df['i'], df['ma20_pct_change_ma'], color='red', linewidth=2)
    ax4.plot(df['i'], df['cum_strategy_returns'], color='blue', linewidth=2)
    ax5.plot(df['i'], df['position'], color='green', linewidth=2)
    
    #ax3.plot(df.index, df['Volume'], color='red', linewidth=2)
    #ax3.plot(df.index, df['obv'], color='green', linewidth=2)
    #ax3.plot(df.index, df['ma_obv'], color='black', linewidth=2)
    
    # display the plots
    plt.show()
    pprint.pprint(tearsheet)
    
    df.to_csv("export.csv")

def plot(tickers,day=30):
    df={}
    i=0
    color=['blue','green','red']
    fig, (ax1) = plt.subplots(1, 1, figsize=(8, 8))
    for t in tickers:
        df[t] = zgetNDays(t,10)
        df[t]['pct_change'] = df[t]['Adj Close'].pct_change()
        df[t]['cum_pct_change']=(1 + df[t]['pct_change']).cumprod() - 1
        df[t].insert(0, 'i', range(1, 1 + len(df[t])))
        ax1.plot(df[t]['i'], df[t]['cum_pct_change'], color=color[i], linewidth=2)
        i = i+1
    
def compareDayByDayPerformance(t,days=90):
    i = 0
    while i<days:
        i=i+1
        s = datetime.now(ist)-timedelta(days=i)
        df = zgetNDays(t,1,s)
        if(len(df)):
            df = signals.bollinger_band_cx(df)
            tearsheet,tearsheetdf = perf.tearsheet(df)
            change = getTotalChange(df)
            ret = round(tearsheet['return'] *100,2)
            print(f"{t} Day:{s} Return:{ret}% Change; {change}%")

backtest('COALINDIA')
#compareDayByDayPerformance('ONGC')
 
#plot(['NIFTY23MAR17300PE','NIFTY23MAR16850CE'])       
    
