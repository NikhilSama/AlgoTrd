#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 14:26:36 2023

@author: nikhilsama
"""
#cfg has all the config parameters make them all globals here
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import numpy as np
import pandas as pd
import pytz
import scipy.stats as stats
import tickerCfg
import utils
import matplotlib.dates as mdates

# set timezone to IST
ist = pytz.timezone('Asia/Kolkata')

import cfg
globals().update(vars(cfg))


def applyTickerSpecificCfg(ticker):
        
    tCfg = utils.getTickerCfg(ticker)
    
    for key, value in tCfg.items():
        globals()[key] = value
        #print(f"setting {key} to {value}")
def plot_trades (df):
    fig, (ax1) = \
        plt.subplots(1, 1, figsize=(8, 8), sharex=True, 
                    gridspec_kw={'height_ratios': [1]})
    ax1.plot(df['trade_num'], df['sum_return'], color='grey', linewidth=2)
    ax1.plot(df['trade_num'], df['prev_peak_sum_return'], color='green', linewidth=2)
    ax1.plot(df['trade_num'], df['drawdown_from_prev_peak_sum'], color='red', linewidth=2)


    # Set the x-axis label and title
    ax1.set_xlabel('Date')
    ax1.set_title('Trade Returns')

    # Add the month labels
    for d in pd.date_range(df.index[0], df.index[-1], freq='MS'):
        ax1.text(mdates.date2num(d), ax1.get_ylim()[0], d.strftime('%Y-%m'), ha='center', va='top', fontsize=8)

    plt.show()

def plot_trades_histogram(df):
    # create 20 bins based on the range of return values
    bins = np.linspace(df['return'].min(), df['return'].max(), 20)

    # group the data by the bins and count the frequency of returns in each bin
    freq, bins = np.histogram(df['return'], bins=bins)

    # plot the histogram
    plt.bar(bins[:-1], freq, width=(bins[1]-bins[0]))

    # check for normal distribution
    if df['return'].mean() == np.median(df['return']):
        print("The distribution is normal")
    else:
        print("The distribution is not normal")

    if 'adjCloseGraph' not in plot:
        plt.show()

def plot_stock_and_option(df):
    fig, (ax1, ax2, ax3) = \
        plt.subplots(3, 1, figsize=(8, 8), sharex=True, 
                    gridspec_kw={'height_ratios': [1,1,1]})
    df['pct_change_stock'] = df['Adj Close'].pct_change()
    df['pct_change_put'] = df['Adj Close-P'].pct_change()
    df['pct_change_call'] = df['Adj Close-C'].pct_change()
    ax1.plot(df['i'], df['pct_change_stock'], color='red', linewidth=2)
    ax2.plot(df['i'], -df['pct_change_put'], color='grey', linewidth=2)
    ax3.plot(df['i'], df['pct_change_call'], color='grey', linewidth=2)
    plt.show()


def plot_backtest(df,trades=None):
    applyTickerSpecificCfg(df['symbol'][0])
    # Plot trades 
    if 'trade_returns' in plot:
        plot_trades(trades)
    
    if 'adjCloseGraph' not in plot:
        return
        ## PLOTTING CODE FOLLOWS 
            
    #pprint.pprint(tearsheet, indent=4)
    #df[['ma_superTrend', 'ma_slow', 'ma_fast']].plot(grid=True, figsize=(12, 8))
#    fig, (ax1, ax2, ax3, ax4, ax5, ax7) = plt.subplots(6, 1, figsize=(8, 8))
    fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7,ax8,ax9) = \
        plt.subplots(9, 1, figsize=(8, 8), sharex=True, 
                     gridspec_kw={'height_ratios': [4, 1, 1, 1, 1, 1,1,1,1]})
    plt.subplots_adjust(left=0.1, bottom=0.1)
    # df['SLOPE-OSC'].fillna(0, inplace=True)
    df['ma20_pct_change_ma'].fillna(0, inplace=True)
    df['ADX-PCT-CHNG'].fillna(0, inplace=True)
    
    if 'OBV-OSC' not in df.columns:
        df['OBV-OSC'] = 0
        df['OBV-OSC-PCT-CHNG'] = 0
        
    df['OBV-OSC-PCT-CHNG'].fillna(0, inplace=True)

    # plot the first series in the first subplot
    #ax1.plot(df['i'], df['ma_superTrend'], color='green', linewidth=3)
    ax1.plot(df['i'], df['VWAP'], color='gold', linewidth=2)
    ax1.plot(df['i'], df['Adj Close'], color='black', linewidth=2)
    ax1.plot(df['i'], df['VWAP_upper'], color='red', linewidth=1)
    ax1.plot(df['i'], df['VWAP_lower'], color='green', linewidth=1)
    # ax1.plot(df['i'], df['SuperTrend'], color='green', linewidth=3)
    # ax1.plot(df['i'], df['ma20'], color='orange', linewidth=1)
    # ax1.plot(df['i'], df['MA-FAST'], color='green', linewidth=1)
    
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
    ax4.plot(df['i'], df['ADX'], color='black', linewidth=2)
    # ax4.plot(df['i'], df['+di'], color='green', linewidth=1)
    # ax4.plot(df['i'], df['-di'], color='red', linewidth=1)
    # draw a threshold line at y=0.5
    ax4.axhline(y=adxThresh, color='red', linestyle='--')
    ax4.axhline(y=-adxThresh, color='red', linestyle='--')
    ax4.axhline(y=adxThresh*adxThreshYellowMultiplier, color='blue', linestyle='--')
    ax4.axhline(y=-adxThresh*adxThreshYellowMultiplier, color='blue', linestyle='--')
    ax4.set_title('ADX↓', loc='right')

    ax5.plot(df['i'], df['ADX-PCT-CHNG'], color='green', linewidth=2)
    ax5.set_title('ADX-PCT-CHNG↓', loc='right')
    # ax5.axhline(y=adxSlopeThresh, color='red', linestyle='--')
    # ax5.axhline(y=-adxSlopeThresh, color='red', linestyle='--')

    ax6.plot(df['i'], df['ma20_pct_change'], color='green', linewidth=2)
    ax6.set_title('MA-SLOPE↓', loc='right')
    ax6.axhline(y=maSlopeThresh, color='red', linestyle='--')
    ax6.axhline(y=-maSlopeThresh, color='red', linestyle='--')

    ax7.plot(df['i'], df['SLOPE-OSC-SLOPE'], color='green', linewidth=2)
    ax7.set_title('SLOPE-OSC-SLOPE sq ↓', loc='right')
    # ax7.axhline(y=maSlopeSlopeThresh, color='red', linestyle='--')
    # ax7.axhline(y=-maSlopeSlopeThresh, color='red', linestyle='--')

    if 'OBV-OSC' in df.columns and df['OBV-OSC'][-1] != 0:
        ax8.plot(df['i'], df['OBV'], color='green', linewidth=2)
        ax8.set_title('OBV↓', loc='right')
        ax8.axhline(y=obvOscThresh, color='red', linestyle='--')
        ax8.axhline(y=-obvOscThresh, color='red', linestyle='--')
        ax8.axhline(y=obvOscThresh*obvOscThreshYellowMultiplier, color='blue', linestyle='--')
        ax8.axhline(y=-obvOscThresh*obvOscThreshYellowMultiplier, color='blue', linestyle='--')
    else:
        ax8.plot(df['i'], df['MA-FAST-SLP'], color='green', linewidth=2)
        ax8.set_title('MA-FAST-SLP↓', loc='right')
        ax8.axhline(y=cfgFastMASlpThresh, color='red', linestyle='--')
        ax8.axhline(y=-cfgFastMASlpThresh, color='red', linestyle='--')


    ax9.plot(df['i'], df['OBV-PCT-CHNG'], color='green', linewidth=2)
    ax9.set_title('OBV-PCT-CHNG ↓', loc='right')
    ax9.axhline(y=cfgObvSlopeThresh, color='red', linestyle='--')
    ax9.axhline(y=-cfgObvSlopeThresh, color='red', linestyle='--')
    
    if len(df) > 500:
        plt.show()
        return # dont plot the shaded region if there are too many rows
    
    xticks = []
    # print(df)
    # Loop over each day in the DataFrame
    for day in np.unique(df.index.date):
        day_rows = df.loc[df.index.date == day]

        # Find the start and end times of the shaded region
        start_time = pd.Timestamp(year=day.year, month=day.month, day=day.day, hour=9, minute=15)
        end_time = pd.Timestamp(year=day.year, month=day.month, day=day.day, hour=9, minute=59)
        start_time = ist.localize(start_time)
        end_time = ist.localize(end_time)
        start_index = 0
        end_index = 0
        try:
            start_index=day_rows.loc[day_rows.index[0],'i']
            end_index=df['i'][end_time]
        except:
            print(f"cant find start or end index(Likely cause data started or ended mid-day) for {start_time} to {end_time}")
        # Add a shaded rectangle for the time period between start_time and end_time
        xticks.extend([start_index,end_index])
        ax1.axvspan(start_index, end_index, alpha=0.2, color='gray')
        ax2.axvspan(start_index, end_index, alpha=0.2, color='gray')
        ax3.axvspan(start_index, end_index, alpha=0.2, color='gray')
        ax4.axvspan(start_index, end_index, alpha=0.2, color='gray')
        ax5.axvspan(start_index, end_index, alpha=0.2, color='gray')
        ax6.axvspan(start_index, end_index, alpha=0.2, color='gray')
        ax7.axvspan(start_index, end_index, alpha=0.2, color='gray')
        ax8.axvspan(start_index, end_index, alpha=0.2, color='gray')
        ax9.axvspan(start_index, end_index, alpha=0.2, color='gray')

        start_time = pd.Timestamp(year=day.year, month=day.month, day=day.day, hour=14, minute=0)
        end_time = pd.Timestamp(year=day.year, month=day.month, day=day.day, hour=15, minute=29)
        start_time = ist.localize(start_time)
        end_time = ist.localize(end_time)
        try:

            start_index=df['i'][start_time]
            end_index = day_rows.loc[day_rows.index[-1],'i']
            
            # Add a shaded rectangle for the time period between start_time and end_time
            ax1.axvspan(start_index, end_index, alpha=0.2, color='yellow')
            ax2.axvspan(start_index, end_index, alpha=0.2, color='yellow')
            ax3.axvspan(start_index, end_index, alpha=0.2, color='yellow')
            ax4.axvspan(start_index, end_index, alpha=0.2, color='yellow')
            ax5.axvspan(start_index, end_index, alpha=0.2, color='yellow')
            ax6.axvspan(start_index, end_index, alpha=0.2, color='yellow')
            ax7.axvspan(start_index, end_index, alpha=0.2, color='yellow')
            ax8.axvspan(start_index, end_index, alpha=0.2, color='yellow')
            ax9.axvspan(start_index, end_index, alpha=0.2, color='yellow')
        except KeyError:
            print(f"Label '{start_time}' not found in DataFrame index.")
        # Create a boolean mask where 'a' is greater than 'b'
        xticks.extend([start_index,end_index])
        mask = (df['position'].shift(-1) == 0) & \
        (df.index.hour >=10) & \
        (df.index.hour <= 13) & \
        ((df['Adj Close'] >= df['upper_band']) | \
        (df['Adj Close'] <= df['lower_band']))
        # Use the shift method to get the start and end times of each region where the mask is True
        start_times = df.index[(mask & ~mask.shift(1, fill_value=False))].tolist()
        end_times = df.index[(mask & ~mask.shift(-1, fill_value=False))].tolist()
        # Loop over each start and end time and add a shaded rectangle
        for start_time, end_time in zip(start_times, end_times):
            start_index=df['i'][start_time]
            end_index=df['i'][end_time]
            xticks.extend([start_index,end_index])

            # Add a shaded rectangle for the time period between start_time and end_time
            ax1.axvspan(start_index, end_index, alpha=0.2, color='red')
            ax2.axvspan(start_index, end_index, alpha=0.2, color='red')
            ax3.axvspan(start_index, end_index, alpha=0.2, color='red')
            ax4.axvspan(start_index, end_index, alpha=0.2, color='red')
            ax5.axvspan(start_index, end_index, alpha=0.2, color='red')
            ax6.axvspan(start_index, end_index, alpha=0.2, color='red')
            ax7.axvspan(start_index, end_index, alpha=0.2, color='red')
            ax8.axvspan(start_index, end_index, alpha=0.2, color='red')
            ax9.axvspan(start_index, end_index, alpha=0.2, color='red')

        mask = (df['position'].shift(-1) != 0)
        # Use the shift method to get the start and end times of each region where the mask is True
        start_times = df.index[(mask & ~mask.shift(1, fill_value=False))].tolist()
        end_times = df.index[(mask & ~mask.shift(-1, fill_value=False))].tolist()
        # Loop over each start and end time and add a shaded rectangle
        for start_time, end_time in zip(start_times, end_times):
            start_index=df['i'][start_time]
            end_index=df['i'][end_time]
            xticks.extend([start_index,end_index])
            # Add a shaded rectangle for the time period between start_time and end_time
            ax1.axvspan(start_index, end_index, alpha=0.2, color='green')
            ax2.axvspan(start_index, end_index, alpha=0.2, color='green')
            ax3.axvspan(start_index, end_index, alpha=0.2, color='green')
            ax4.axvspan(start_index, end_index, alpha=0.2, color='green')
            ax5.axvspan(start_index, end_index, alpha=0.2, color='green')
            ax6.axvspan(start_index, end_index, alpha=0.2, color='green')
            ax7.axvspan(start_index, end_index, alpha=0.2, color='green')
            ax8.axvspan(start_index, end_index, alpha=0.2, color='green')
            ax9.axvspan(start_index, end_index, alpha=0.2, color='green')
    # create the slider widget
    plt.xticks(xticks,rotation=90)
    axpos = plt.axes([0.25, 0.01, 0.65, 0.03])
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
        
        #ADX slope
        ax5.set_ylim(-.15,.15)
        ax5.set_xlim(pos, pos+sliderLen)  # update the x-axis limits

        # ma 20 pct change
        ax6.set_ylim(-3.5,3.5)
        ax6.set_xlim(pos, pos+sliderLen)  # update the x-axis limits
        # OBV - OSC
        ax7.set_ylim(-1,1)
        ax7.set_xlim(pos, pos+sliderLen)  # update the x-axis limits
        
        # # Get the current y-value at pos
        # yval6 = np.interp(pos, df['i'], df['OBV'])
        # # Set the y-axis limits to ±5% of the current y-value at pos
        # yrange6 = abs(50* yval6)
        # ax6.set_ylim(yval6 - yrange6, yval6 + yrange6)
        # ax6.set_xlim(pos, pos+sliderLen)  # update the x-axis limits
        # OBV - OSC
        
        # OBV - OSC
        ax8.set_ylim(-1,1)
        ax8.set_xlim(pos, pos+sliderLen)  # update the x-axis limits
        # OBV - OSC
        
        #OBV SLOPE
        ax9.set_ylim(-1,1)
        ax9.set_xlim(pos, pos+sliderLen)  # update the x-axis limits
        
        fig.canvas.draw_idle()

    slider.on_changed(update)
    # display the plots
    plt.show()
    #pprint.pprint(tearsheet)
    #### END OF PLOTTING CODE
    