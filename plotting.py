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
import DownloadHistorical as downloader

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

def plot_option_intrinsic(df):
    fig, (ax1, ax2, ax3) = \
        plt.subplots(3, 1, figsize=(8, 8), sharex=True, 
                    gridspec_kw={'height_ratios': [1,1,1]})
    df['closelessstrike'] = df['Adj Close'] - df['Strike-C']
    df['callclose'] = df['Adj Close-P']
    df['timevalue'] = df['callclose'] - df['closelessstrike']
    ax1.plot(df['i'], df['closelessstrike'], color='red', linewidth=2)
    ax2.plot(df['i'], -df['callclose'], color='green', linewidth=2)
    ax3.plot(df['i'], df['timevalue'], color='yellow', linewidth=2)
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

def plot_returns_on_nifty(df):
    fig, (ax1, ax2) = \
    plt.subplots(2, 1, figsize=(8, 8), sharex=True, 
                gridspec_kw={'height_ratios': [3,1]})
    niftyDf = downloader.zget(df.index[0],df.index[-1],'NIFTY 50','day',includeOptions=False)
    ax1.plot(niftyDf.index, niftyDf['Adj Close'], color='red', linewidth=2)
    ax2.plot(df.index, df['strategy_returns'], color='red', linewidth=2)
    plt.show()


def plot_option_vs_stock(df):
    fig, (ax1, ax2) = \
        plt.subplots(2, 1, figsize=(8, 8), sharex=True, 
                     gridspec_kw={'height_ratios': [1, 1]})
    plt.subplots_adjust(left=0.1, bottom=0.1)
    iv = abs(df['nifty'] - int(df['strike'][-1]))

    ax1.plot(df['i'], df['Adj Close'], color='black', linewidth=1)
    ax1.plot(df['i'], iv, color='red', linewidth=1)

    # diffClose = df['Adj Close'].diff(1)
    # diffIv = iv.diff(1)
    diff = df['Adj Close'] - iv.shift
    madiff = diff.rolling(window=2000,min_periods=10).mean()
    ax2.plot(df['i'], diff, color='black', linewidth=1)
    ax2.plot(df['i'], madiff, color='red', linewidth=1)
    
def plot_backtest(df,trades=None):
    applyTickerSpecificCfg(df['symbol'][0])
    # Plot trades 
    if 'trade_returns' in plot:
        plot_trades(trades)
    
    if 'adjCloseGraph' not in plot:
        return
        ## PLOTTING CODE FOLLOWS 
    if len(df) > 500 :
        return
    #pprint.pprint(tearsheet, indent=4)
    #df[['ma_superTrend', 'ma_slow', 'ma_fast']].plot(grid=True, figsize=(12, 8))
#    fig, (ax1, ax2, ax3, ax4, ax5, ax7) = plt.subplots(6, 1, figsize=(8, 8))
    fig, (ax1, ax2, ax3, ax4,ax5,ax6) = \
        plt.subplots(6, 1, figsize=(8, 8), sharex=True, 
                     gridspec_kw={'height_ratios': [6, 1, 1, 4,1,1]})
    plt.subplots_adjust(left=0.1, bottom=0.1)
    # df['SLOPE-OSC'].fillna(0, inplace=True)
    # df['ma20_pct_change_ma'].fillna(0, inplace=True)
    # df['ADX-PCT-CHNG'].fillna(0, inplace=True)
    
    if 'OBV-OSC' not in df.columns:
        df['OBV-OSC'] = 0
        df['OBV-OSC-PCT-CHNG'] = 0
        
    df['OBV-OSC-PCT-CHNG'].fillna(0, inplace=True)

    iv = abs(df['nifty'] - int(df['strike'][-1]))
    # plot the first series in the first subplot
    #ax1.plot(df['i'], df['ma_superTrend'], color='green', linewidth=3)
    # ax1.plot(df['i'], df['VWAP'], color='gold', linewidth=2)
    ax1.plot(df['i'], df['Adj Close'], color='black', linewidth=2)
    ax1.plot(df['i'], iv, color='red', linewidth=2)
    # ax1.plot(df['i'], df['renko_brick_high'], color='green', linewidth=1)
    # ax1.plot(df['i'], df['renko_brick_low'], color='red', linewidth=1)
    # ax1.plot(df['i'], df['pocShrtTrm'], color='black', linewidth=1)
    # ax1.plot(df['i'], df['ShrtTrmHigh'], color='green', linewidth=1)
    # ax1.plot(df['i'], df['ShrtTrmLow'], color='red', linewidth=1)
    # ax1.plot(df['i'], abs(df['sl1']), color='red', linewidth=3)
    # ax1.plot(df['i'], df['val']+(df['slpVal']*10), color='red', linewidth=1)
    # ax1.plot(df['i'], df['ma20'], color='orange', linewidth=1)
    # ax1.plot(df['i'], df['MA-FAST'], color='green', linewidth=1)
    ax1.plot(df['i'], df['ma20'], color='blue', linewidth=1)
    ax1.plot(df['i'], df['upper_band'], color='green', linewidth=1)
    ax1.plot(df['i'], df['lower_band'], color='red', linewidth=1)
    # ax1.plot(df['i'], df['SuperTrendUpper'], color='green', linewidth=1)
    # ax1.plot(df['i'], df['SuperTrendLower'], color='red', linewidth=1)
    # ax1.plot(df['i'], df['vah'], color='green', linewidth=1)
    # ax1.plot(df['i'], df['val'], color='red', linewidth=1)
    # ax1.plot(df['i'], df['poc'], color='yellow', linewidth=1)

    #Draw horizontal lines at renko bricks
    # start = df['renko_brick_low'].min()
    # end = df['renko_brick_high'].max()
    # for y in range(int(start), int(end), 8):
    #     ax1.axhline(y, color='grey', linestyle='--')  # adjust color and linestyle as necessary

    # plot the second series in the second subplot
    #ax2.plot(df['i'], df['ma_superTrend_pct_change'], color='red', linewidth=2)
    #ax2.plot(df.index, df['superTrend'], color='red', linewidth=2)
    #ax2.plot(df['i'], df['Adj Close-P'], color='blue', linewidth=2)
    #ax2.plot(df['i'], df['Adj Close-C'], color='red', linewidth=2)   
    #ax3.plot(df['i'], df['ma20_pct_change_ma'], color='red', linewidth=2)
    ax2.plot(df['i'], df['cum_strategy_returns'], color='blue', linewidth=2)
    ax2.set_title('returns ↓', loc='right')
    # ax2.plot(df['i'], df['SuperTrendUpper'], color='green', linewidth=1)
    # ax2.plot(df['i'], df['SuperTrendLower'], color='red', linewidth=1)

    # ax2.axhline(round(df['nifty'][-1]/100)*100, color='grey', linestyle='--')  # adjust color and linestyle as necessary
    # ax2.axhline((round(df['nifty'][-1]/100)*100) -100, color='grey', linestyle='--')  # adjust color and linestyle as necessary
    # ax2.axhline((round(df['nifty'][-1]/100)*100) +100, color='grey', linestyle='--')  # adjust color and linestyle as necessary

    ax3.plot(df['i'], df['position'], color='green', linewidth=2)
    ax3.set_title('Position↓', loc='right')
    # df['ADX'] .fillna(0, inplace=True)
    # ax4.plot(df['i'], abs(df['Adj Close'] - iv), color='black', linewidth=2)
    ax4.plot(df['i'], df['RSI'], color='black', linewidth=2)
    ax4.axhline(y=75, color='red', linestyle='--')
    ax4.axhline(y=25, color='red', linestyle='--')

    # threshold = df['volDeltaThreshold'] * 2

    # Clip the data using the calculated threshold
    # clippedVolDelta = np.clip(df['volDelta'], -threshold, threshold)
    # clippedMaxDelta = np.clip(df['maxVolDelta'], -threshold, threshold)
    # clippedMinDelta = np.clip(df['minVolDelta'], -threshold, threshold)

    # ax4.plot(df['i'], clippedMaxDelta, color='red', linewidth=1)
    # ax4.plot(df['i'], clippedMinDelta, color='green', linewidth=1)
    # ax4.plot(df['i'], clippedVolDelta, color='black', linewidth=1)
    # ax4.plot(df['i'], df['volDeltaThreshold'], color='green', linewidth=1)
    # ax4.plot(df['i'], -df['volDeltaThreshold'], color='red', linewidth=1)
    # draw a threshold line at y=0.5
    # ax4.axhline(y=-1, color='red', linestyle='--')
    # ax4.axhline(y=-2, color='blue', linestyle='--')
    # ax4.axhline(y=-50000, color='red', linestyle='--')
    # ax4.axhline(y=-.2, color='blue', linestyle='--')
    # ax4.axhline(y=1, color='green', linestyle='--')
    # ax4.axhline(y=2, color='blue', linestyle='--')
    # ax4.axhline(y=.1, color='green', linestyle='--')
    # ax4.axhline(y=.2, color='blue', linestyle='--')
    ax4.set_title('RSI↓', loc='right')    
    
    ax5.plot(df['i'], df['ma20_pct_change'], color='red', linewidth=1)
    # ax5.plot(df['i'], df['stCumVolDelta'], color='red', linewidth=1)
    # # ax5.plot(df['i'], df['slpSTPoc'], color='black', linewidth=1)
    # # ax5.plot(df['i'], df['slpVah'], color='green', linewidth=1)
    # # ax5.plot(df['i'], df['slpVal'], color='red', linewidth=1)
    ax5.set_title('Slope↓', loc='right')
    # ax5.axhline(y=0, color='red', linestyle='--')
    ax5.axhline(y=0.02, color='red', linestyle='--')
    ax5.axhline(y=-0.02, color='red', linestyle='--')
    # ax5.axhline(y=-50000, color='red', linestyle='--')

    ax6.plot(df['i'], df['Volume'], color='green', linewidth=2)
    # ax6.set_title('VolDelta ↓', loc='right')
    # ax6.axhline(y=av, color='red', linestyle='--')
    # ax6.axhline(y=-av, color='red', linestyle='--')
    # # ax6.axhline(y=4*av, color='red', linestyle='--')
    # # ax6.axhline(y=-4*av, color='red', linestyle='--')

    # df['VolDeltaRatio'] = df['VolDeltaRatio'].clip(0, 2)
    # ax7.plot(df['i'], df['VolDeltaRatio'], color='green', linewidth=2)
    # ax7.set_title('VolDeltaRatio ↓', loc='right')
    # ax7.axhline(y=cfgFastMASlpThresh, color='red', linestyle='--')
    # ax7.axhline(y=-cfgFastMASlpThresh, color='red', linestyle='--')

    # if 'OBV-OSC' in df.columns and df['OBV-OSC'][-1] != 0:
    #     ax8.plot(df['i'], df['OBV'], color='green', linewidth=2)
    #     ax8.set_title('OBV↓', loc='right')
    #     ax8.axhline(y=obvOscThresh, color='red', linestyle='--')
    #     ax8.axhline(y=-obvOscThresh, color='red', linestyle='--')
    #     ax8.axhline(y=obvOscThresh*obvOscThreshYellowMultiplier, color='blue', linestyle='--')
    #     ax8.axhline(y=-obvOscThresh*obvOscThreshYellowMultiplier, color='blue', linestyle='--')
    # else:
    #     ax8.plot(df['i'], df['MA-FAST-SLP'], color='green', linewidth=2)
    #     ax8.set_title('MA-FAST-SLP↓', loc='right')
    #     ax8.axhline(y=cfgFastMASlpThresh, color='red', linestyle='--')
    #     ax8.axhline(y=-cfgFastMASlpThresh, color='red', linestyle='--')


    # ax9.plot(df['i'], df['OBV-PCT-CHNG'], color='green', linewidth=2)
    # ax9.set_title('OBV-PCT-CHNG ↓', loc='right')
    # ax9.axhline(y=cfgObvSlopeThresh, color='red', linestyle='--')
    # ax9.axhline(y=-cfgObvSlopeThresh, color='red', linestyle='--')
    
    if len(df) > 500 :
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
        ax5.axvspan(start_index, end_index, alpha=0.2, color='gray') if 'ax5' in locals() else None
        ax6.axvspan(start_index, end_index, alpha=0.2, color='gray') if 'ax6' in locals() else None
        ax7.axvspan(start_index, end_index, alpha=0.2, color='gray') if 'ax7' in locals() else None
        ax8.axvspan(start_index, end_index, alpha=0.2, color='gray') if 'ax8' in locals() else None
        ax9.axvspan(start_index, end_index, alpha=0.2, color='gray') if 'ax9' in locals() else None

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
            ax5.axvspan(start_index, end_index, alpha=0.2, color='yellow') if 'ax5' in locals() else None
            ax6.axvspan(start_index, end_index, alpha=0.2, color='yellow') if 'ax6' in locals() else None
            ax7.axvspan(start_index, end_index, alpha=0.2, color='yellow') if 'ax7' in locals() else None
            ax8.axvspan(start_index, end_index, alpha=0.2, color='yellow') if 'ax8' in locals() else None
            ax9.axvspan(start_index, end_index, alpha=0.2, color='yellow') if 'ax9' in locals() else None
        except KeyError:
            print(f"Label '{start_time}' not found in DataFrame index.")
        # Create a boolean mask where 'a' is greater than 'b'
        xticks.extend([start_index,end_index])
        # mask = (df['position'].shift(-1) == 0) & \
        # (df.index.hour >=10) & \
        # (df.index.hour <= 13) & \
        # ((df['Adj Close'] >= df['upper_band']) | \
        # (df['Adj Close'] <= df['lower_band']))
        # # Use the shift method to get the start and end times of each region where the mask is True
        # start_times = df.index[(mask & ~mask.shift(1, fill_value=False))].tolist()
        # end_times = df.index[(mask & ~mask.shift(-1, fill_value=False))].tolist()
        # # Loop over each start and end time and add a shaded rectangle
        # for start_time, end_time in zip(start_times, end_times):
        #     start_index=df['i'][start_time]
        #     end_index=df['i'][end_time]
        #     xticks.extend([start_index,end_index])

        #     # Add a shaded rectangle for the time period between start_time and end_time
        #     ax1.axvspan(start_index, end_index, alpha=0.2, color='red')
        #     ax2.axvspan(start_index, end_index, alpha=0.2, color='red')
        #     ax3.axvspan(start_index, end_index, alpha=0.2, color='red')
        #     ax4.axvspan(start_index, end_index, alpha=0.2, color='red')
        #     ax5.axvspan(start_index, end_index, alpha=0.2, color='red')
        #     ax6.axvspan(start_index, end_index, alpha=0.2, color='red')
        #     ax7.axvspan(start_index, end_index, alpha=0.2, color='red')
        #     ax8.axvspan(start_index, end_index, alpha=0.2, color='red')
        #     ax9.axvspan(start_index, end_index, alpha=0.2, color='red')

        # mask = ((df['maVolDelta'] > df['volDeltaThreshold']) | (df['maVolDelta'] < -df['volDeltaThreshold']))
        # # Use the shift method to get the start and end times of each region where the mask is True
        # start_times = df.index[(mask & ~mask.shift(1, fill_value=False))].tolist()
        # end_times = df.index[(mask & ~mask.shift(-1, fill_value=False))].tolist()
        # # Loop over each start and end time and add a shaded rectangle
        # for start_time, end_time in zip(start_times, end_times):
        #     start_index=df['i'][start_time]
        #     end_index=df['i'][end_time]
        #     xticks.extend([start_index,end_index])
        #     # Add a shaded rectangle for the time period between start_time and end_time
        #     ax1.axvspan(start_index, end_index, alpha=0.2, color='green')
        #     ax2.axvspan(start_index, end_index, alpha=0.2, color='green')
        #     ax3.axvspan(start_index, end_index, alpha=0.2, color='green')
        #     ax4.axvspan(start_index, end_index, alpha=0.2, color='green')
        #     ax5.axvspan(start_index, end_index, alpha=0.2, color='green') if 'ax5' in locals() else None
        #     ax6.axvspan(start_index, end_index, alpha=0.2, color='green')   if 'ax6' in locals() else None
        #     ax7.axvspan(start_index, end_index, alpha=0.2, color='green') if 'ax7' in locals() else None
        #     ax8.axvspan(start_index, end_index, alpha=0.2, color='green') if 'ax8' in locals() else None
        #     ax9.axvspan(start_index, end_index, alpha=0.2, color='green') if 'ax9' in locals() else None
    
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
            ax1.axvspan(start_index, end_index, alpha=0.2, color='red')
            ax2.axvspan(start_index, end_index, alpha=0.2, color='red')
            ax3.axvspan(start_index, end_index, alpha=0.2, color='red')
            ax4.axvspan(start_index, end_index, alpha=0.2, color='red')
            ax5.axvspan(start_index, end_index, alpha=0.2, color='red') if 'ax5' in locals() else None
            ax6.axvspan(start_index, end_index, alpha=0.2, color='red')  if 'ax6' in locals() else None
            ax7.axvspan(start_index, end_index, alpha=0.2, color='red') if 'ax7' in locals() else None
            ax8.axvspan(start_index, end_index, alpha=0.2, color='red') if 'ax8' in locals() else None
            ax9.axvspan(start_index, end_index, alpha=0.2, color='red') if 'ax9' in locals() else None

    
    
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
    