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

tickers = td.get_sp500_tickers()
nifty = td.get_nifty_tickers()
index_tickers = td.get_index_tickers()

# set timezone to IST
ist = pytz.timezone('Asia/Kolkata')

def zget(t,s,e,i,includeOptions=False):
    #Get latest minute tick from zerodha
    df = downloader.zget(s,e,t,i,includeOptions=includeOptions)
    df = downloader.zColsToDbCols(df)
    return df

def zgetNDays(t,n,e=datetime.now(ist),i="minute",includeOptions=False):
    s = e - timedelta(days=n)
    return zget(t, s, e, i,includeOptions=includeOptions)

def getTotalChange(df):
    return round(100*(df['Open'][0] - df['Adj Close'][-1])/df['Open'][0],2)


def test(t='ADANIENT',i='minute',days=1,adxThreh=30,includeOptions=False):
    df = zgetNDays(t,days,i=i,includeOptions=includeOptions)
    df.insert(0, 'i', range(1, 1 + len(df)))
  
    dataPopulators = [signals.populateBB, signals.populateADX]
    signalGenerators = [signals.getSig_BB_CX_ADX_MASLOPE]

    signals.applyIntraDayStrategy(df1,dataPopulators,signalGenerators)
    

def backtest(t='ADANIENT',i='minute',days=1,adxThresh=30,maThresh=1,obvOscThresh=.25,includeOptions=False, plot=False):

    print (f'Start {datetime.now(ist)}')
    df = zgetNDays(t,days,i=i,includeOptions=includeOptions)
    print (f"ZGET Complete {datetime.now(ist)}")
    
    # Adding new column
    df.insert(0, 'i', range(1, 1 + len(df)))
    
    dataPopulators = [signals.populateBB, signals.populateADX, signals.populateOBV]
    signalGenerators = [signals.getSig_BB_CX
                      ,signals.getSig_ADX_FILTER
                       ,signals.getSig_MASLOPE_FILTER
                       ,signals.getSig_OBV_FILTER
                        ]
    overrideSignalGenerators = [signals.getSig_Extreme_ADX_OBV_MA20_OVERRIDE]
    signals.applyIntraDayStrategy(df,dataPopulators,signalGenerators,
                                  overrideSignalGenerators,
                                  adxThresh=adxThresh, maThresh=maThresh,
                                  obvOscThresh=obvOscThresh)


    tearsheet,tearsheetdf = perf.tearsheet(df)
    print(f'Total Return: {tearsheet["return"]*100}%')
    df.to_csv("export.csv")

    if (plot == False):
        return
    ## PLOTTING CODE FOLLOWS 
    #pprint.pprint(tearsheet, indent=4)
    #df[['ma_superTrend', 'ma_slow', 'ma_fast']].plot(grid=True, figsize=(12, 8))
#    fig, (ax1, ax2, ax3, ax4, ax5, ax7) = plt.subplots(6, 1, figsize=(8, 8))
    fig, (ax1, ax2, ax3, ax4, ax5, ax7) = \
        plt.subplots(6, 1, figsize=(8, 8), sharex=True, 
                     gridspec_kw={'height_ratios': [4, 1, 1, 1, 1, 1]})
    plt.subplots_adjust(left=0.1, bottom=0.1)
    df['ma20_pct_change_ma'].fillna(0, inplace=True)

    # plot the first series in the first subplot
    #ax1.plot(df['i'], df['ma_superTrend'], color='green', linewidth=2)
    #ax1.plot(df['i'], df['ma20'], color='gold', linewidth=2)
    ax1.plot(df['i'], df['Adj Close'], color='red', linewidth=2)
    ax1.plot(df['i'], df['lower_band'], color='grey', linewidth=2)
    ax1.plot(df['i'], df['upper_band'], color='grey', linewidth=2)
    #ax1.plot(df['i'], df['ma_superTrend'], color='orange', linewidth=4)
    
    # plot the second series in the second subplot
    #ax2.plot(df['i'], df['ma_superTrend_pct_change'], color='red', linewidth=2)
    #ax2.plot(df.index, df['superTrend'], color='red', linewidth=2)
    #ax2.plot(df['i'], df['Adj Close-P'], color='blue', linewidth=2)
    #ax2.plot(df['i'], df['Adj Close-C'], color='red', linewidth=2)   
    #ax3.plot(df['i'], df['ma20_pct_change_ma'], color='red', linewidth=2)
    ax2.plot(df['i'], df['cum_strategy_returns'], color='blue', linewidth=2)
    ax2.set_title('Cumulative Strategy Returns ↓', loc='right')

    ax3.plot(df['i'], df['position'], color='green', linewidth=2)
    ax3.set_title('Position↓', loc='right')
    df['ADX'] .fillna(0, inplace=True)
    ax4.plot(df['i'], df['ADX'], color='green', linewidth=2)
    # draw a threshold line at y=0.5
    ax4.axhline(y=adxThresh, color='red', linestyle='--')
    ax4.axhline(y=-adxThresh, color='red', linestyle='--')
    ax4.set_title('ADX↓', loc='right')

    ax5.set_title('ma20_pct_change_ma - MA↓', loc='right')
    ax5.axhline(y=maThresh, color='red', linestyle='--')
    ax5.axhline(y=-maThresh, color='red', linestyle='--')
    ax5.plot(df['i'], df['ma20_pct_change_ma'], color='green', linewidth=2)

    # ax6.plot(df['i'], df['OBV'], color='green', linewidth=2)
    # ax6.set_title('MA OBV ↓', loc='right')

    ax7.plot(df['i'], df['OBV-OSC'], color='green', linewidth=2)
    ax7.set_title('OBV OSC↓', loc='right')
    ax7.axhline(y=obvOscThresh, color='red', linestyle='--')
    ax7.axhline(y=-obvOscThresh, color='red', linestyle='--')
    
    
    # Loop over each day in the DataFrame
    for day in np.unique(df.index.date):
        # Find the start and end times of the shaded region
        start_time = pd.Timestamp(year=day.year, month=day.month, day=day.day, hour=9, minute=15)
        end_time = pd.Timestamp(year=day.year, month=day.month, day=day.day, hour=10, minute=0)
        start_time = ist.localize(start_time)
        end_time = ist.localize(end_time)
        start_index=df['i'][start_time]
        end_index=df['i'][end_time]
        # Add a shaded rectangle for the time period between start_time and end_time
        ax1.axvspan(start_index, end_index, alpha=0.2, color='gray')
        ax2.axvspan(start_index, end_index, alpha=0.2, color='gray')
        ax3.axvspan(start_index, end_index, alpha=0.2, color='gray')
        ax4.axvspan(start_index, end_index, alpha=0.2, color='gray')
        ax5.axvspan(start_index, end_index, alpha=0.2, color='gray')
        ax7.axvspan(start_index, end_index, alpha=0.2, color='gray')

        start_time = pd.Timestamp(year=day.year, month=day.month, day=day.day, hour=14, minute=0)
        end_time = pd.Timestamp(year=day.year, month=day.month, day=day.day, hour=15, minute=29)
        start_time = ist.localize(start_time)
        end_time = ist.localize(end_time)
        try:

            start_index=df['i'][start_time]
            end_index=df['i'][end_time]
            # Add a shaded rectangle for the time period between start_time and end_time
            ax1.axvspan(start_index, end_index, alpha=0.2, color='yellow')
            ax2.axvspan(start_index, end_index, alpha=0.2, color='yellow')
            ax3.axvspan(start_index, end_index, alpha=0.2, color='yellow')
            ax4.axvspan(start_index, end_index, alpha=0.2, color='yellow')
            ax5.axvspan(start_index, end_index, alpha=0.2, color='yellow')
            ax7.axvspan(start_index, end_index, alpha=0.2, color='yellow')
        except KeyError:
            print(f"Label '{start_time}' not found in DataFrame index.")
        # Create a boolean mask where 'a' is greater than 'b'
        mask = (df['position'].shift(-2) == 0) & \
            (df.index.hour >=10) & \
            (df.index.hour <= 13) & \
            ((df['Adj Close'] > df['upper_band']) | \
            (df['Adj Close'] < df['lower_band']))
        # Use the shift method to get the start and end times of each region where the mask is True
        start_times = df.index[(mask & ~mask.shift(1, fill_value=False))].tolist()
        end_times = df.index[(mask & ~mask.shift(-1, fill_value=False))].tolist()
        # Loop over each start and end time and add a shaded rectangle
        for start_time, end_time in zip(start_times, end_times):
            start_index=df['i'][start_time]
            end_index=df['i'][end_time]
            
            # Add a shaded rectangle for the time period between start_time and end_time
            ax1.axvspan(start_index, end_index, alpha=0.2, color='red')
            ax2.axvspan(start_index, end_index, alpha=0.2, color='red')
            ax3.axvspan(start_index, end_index, alpha=0.2, color='red')
            ax4.axvspan(start_index, end_index, alpha=0.2, color='red')
            ax5.axvspan(start_index, end_index, alpha=0.2, color='red')
            ax7.axvspan(start_index, end_index, alpha=0.2, color='red')

    # create the slider widget
    axpos = plt.axes([0.1, 0.1, 0.65, 0.03])
    slider = Slider(axpos, 'Time', df['i'][0], df['i'].iloc[-1], 
                    valinit=df['i'][0])

    # define the function to update the plot when the slider is changed
    def update(val):
        pos = slider.val
        sliderLen = 300
        # Get the current y-value at pos
        yval1 = np.interp(pos, df['i'], df['Adj Close'])
        # Set the y-axis limits to ±5% of the current y-value at pos
        yrange1 = abs(0.02 * yval1)
        ax1.set_ylim(yval1 - yrange1, yval1 + yrange1)
        ax1.set_xlim(pos, pos+sliderLen)  # update the x-axis limits
        
        # Get the current y-value at pos
        # yval2 = np.interp(pos, df['i'], df['cum_strategy_returns'])
        # # Set the y-axis limits to ±5% of the current y-value at pos
        # print (f"yval2: {yval2}")
        # yrange2 = abs(0.01 * yval1)
        # ax2.set_ylim(yval2 - yrange2, yval2 + yrange2)
        y2lim = ax2.get_ylim( )
        ax2.set_ylim(y2lim[0],y2lim[1])
        ax2.set_xlim(pos, pos+sliderLen)  # update the x-axis limits
        
        # POSITIONgraph
        ax3.set_ylim(-1.1,1.1)
        ax3.set_xlim(pos, pos+sliderLen)  # update the x-axis limits
        
        # ADX
        ax4.set_ylim(0,100)
        ax4.set_xlim(pos, pos+sliderLen)  # update the x-axis limits
        
        # ma 20 pct change
        ax5.set_ylim(-2.5,2.5)
        ax5.set_xlim(pos, pos+sliderLen)  # update the x-axis limits
        
        # # Get the current y-value at pos
        # yval6 = np.interp(pos, df['i'], df['OBV'])
        # # Set the y-axis limits to ±5% of the current y-value at pos
        # yrange6 = abs(50* yval6)
        # ax6.set_ylim(yval6 - yrange6, yval6 + yrange6)
        # ax6.set_xlim(pos, pos+sliderLen)  # update the x-axis limits
        
        # OBV - OSC
        ax7.set_ylim(-1,1)
        ax7.set_xlim(pos, pos+sliderLen)  # update the x-axis limits
        
        fig.canvas.draw_idle()

    slider.on_changed(update)
    # display the plots
    plt.show()
    #pprint.pprint(tearsheet)
    #### END OF PLOTTING CODE
    # print (f"END Complete {datetime.now(ist)}")

# Plot the graph of closing prices for the array of tickers provided
# and the interval provided and the number of days provided
def plot(uticker, tickers,i='minute', 
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
        df[t]['pct_change'] = df[t]['Adj Close'].pct_change()
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

#plot(['ASIANPAINT'],10,'minute')
#backtest('HDFCLIFE','minute',adxThreh=30)
backtest('RELIANCE','minute',days=10, adxThresh=30,obvOscThresh=0.25,includeOptions=False, plot=True)
#backtest('HDFCLIFE','minute',adxThreh=25)
#backtest('ASIANPAINT','minute',adxThreh=25)
#backtest('HDFCLIFE','minute',adxThreh=30)
#backtest('ADANIENT','minute',adxThreh=30)
#compareDayByDayPerformance('ONGC')
 
#plot('INFY',['ASIANPAINT23MAR2840PE','ASIANPAINT23MAR2840CE'],i='minute', days=3,e=datetime.now(ist)-timedelta(days=15))   

    # print hello
# print hello
