#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 12:43:21 2023

@author: nikhilsama
"""

#Functions to calculate the performace of an algo strategy 
# All functions expect as argument a dataframe w collumns: 
# "datetime", "open", "high", "low", "close", "Adj Close", "volume", 
# "signal",
# signal is 1 if we went long, 0 if we exited, -1 if we went short
# Additional columns 'bnh_returns' (buy and hold returns)
# and 'cum_bnh_returns' cummulative bnh returns are calculated for the Adj CLose
# in this library itself
# "position" is calculated ffill the signal until it changes
#  "strategy_return" are calculated as a product of position and return
# "cum_strategy_return" is cummulated along strategy return
# bnh_ln_return strategy_ln_returns are natural log returns

import pandas as pd
import numpy as np
import csv

pd.options.mode.chained_assignment = None  # default='warn'

### Global Data frame prep functions that calculate positions and returns
### based on signal column

def calculate_bnh_returns (df):
    df['bnh_returns'] = df["Adj Close"].pct_change()
    df['cum_bnh_returns'] = (1+df["bnh_returns"]).cumprod() - 1
    df['bnh_ln_returns'] = np.log(df['Adj Close']/df['Adj Close'].shift(1))
    df['cum_bnh_ln_returns'] = df['bnh_ln_returns'].cumsum()

    ## IF Call and Put option data exists
    if "Adj Close-C" in df.columns:
        df['bnh_returns-C'] = df["Adj Close-C"].pct_change()
    if "Adj Close-P" in df.columns:
        df['bnh_returns-P'] = df["Adj Close-P"].pct_change()


def calculate_positions (df,close_at_end=True):
    
    
    # creating long and short positions 
    #df['position'] = df['signal'].replace(to_replace=10, method='ffill')
    
    #Set first signal to zero, we always calculating signals from second 
    #row based on data in first row/candle
    df.iloc[0,df.columns.get_loc('signal')] = 0
    df['position'] = df['signal'].fillna(method='ffill')
    
    # if signal is generated on the nth bar, then position will be taken
    # on next n+1th bar, and return of n+1th bar is what we will get
    df['position'] = df['position'].shift(1)

    #CLOSE all positions at end of df; this will always be the last trade
    #return on last candle will always be zero
    if (close_at_end):
        df.iloc[-1, df.columns.get_loc('position')] = 0

    #signal (eg crossover) can be buy or 1 for several consecutive periods.  If we already
    #bought in the first period, then the second signal is not a trade as such
    #therefore we use position chanage variable below to distinguish actual 
    #change in position, from the signal itself.  Additionally, signal can be 0
    #in two situations .. first, just do nothing, second, exit the the 
    #previous trade (exit trade).  Position_change variable helps distinguish
    #these two situations
    #
    #position change will be 1 or -1 when exiting, or 2 or -2 when transit
    #from long to short or vice versa.  0 when we are staying in the same 
    #position as before
    df['position_change'] = df['position']-df.shift()['position']
    df['position_change'] = df['position_change'].fillna(0)
    return df

def calculate_strategy_returns (df):
    # calculating stretegy returns
    if "Adj Close-C" in df.columns and "Adj Close-P" in df.columns:
        df['strategy_returns'] = np.where( (df['position'] > 0),
                                          df['bnh_returns-P'] * -1,0)
        df['strategy_returns'] = np.where( (df['position'] < 0),
                                          df['bnh_returns-C'] * -1,df['strategy_returns'])
        # if position > 0 sell Put, returns will be negative put option returns'
        # if position < 0 sell Call, returns will be negative call options returns    
    else:
        # If not options
        df['strategy_returns'] = df['bnh_returns'] * (df['position'])

    df['cum_strategy_returns'] = (1+df['strategy_returns']).cumprod() - 1
    df['strategy_ln_returns'] = df['bnh_ln_returns'] * (df['position'])
    df['cum_strategy_ln_returns'] = df['strategy_ln_returns'].cumsum()

def prep_dataframe (df):
    #calc returns if not already calculated
    if not set(['position','cum_bnh_returns']).issubset(df.columns):
        calculate_bnh_returns(df)
        
    #calc positions if not already calculated
    if not set(['position']).issubset(df.columns):
        calculate_positions (df)
    
    #calc strategy returns if not already calculated
    if not set(['strategy_returns','cum_strategy_returns']).issubset(df.columns):
        calculate_strategy_returns(df)

#### END OF GLOBAL prep functions

# Return CAGR for strategy
def get_n_for_cagr(rows, timeframe):
    #Assume 252 trading days in a year
    n = 0
    if timeframe == 'm':
        n = rows/(252*8*60)
    elif timeframe == 'd':
        n = rows/252
    return n

def CAGR (df, timeframe='d'):
    prep_dataframe(df)
    n = get_n_for_cagr(len(df), 'd')
    CAGR = (df["cum_return"][-1])**(1/n) - 1
    return CAGR

# MAX Drawdown
def max_drawdown (df):
    max = 0
    return max


# Get Trades -- Gets trades 
# trades should be df w columns 
# ['Entry date', 'Exit date', 'abs_return', 'cagr', 'max_loss', 'max_drawdown', 'avg_return', 'median_return', 'stddev_return']

def calc_trade_returns (date, trades, ticker_data): 
    #last row of tickers is really the first day a new position was taken
    #therfore returns from that candle should not be included in this trade
    #they will be included in next trade.  Therefore set strategy returns for 
    #last candle here to 0
    ticker_data.iloc[-1, ticker_data.columns.get_loc('strategy_returns')] = 0
    ticker_data['cum_trade_returns'] =  (1+ticker_data['strategy_returns']).cumprod() - 1
    trades.loc[date, 'return'] = ticker_data.iloc[-1]['cum_trade_returns']

    return

def get_trades (df):
    trades = df[df["position_change"]!=0]
    
    if (len(trades) <1):
        return trades

    prev_date = trades.index[0]
    i = 0
    
    for date in trades.index:

        #identify the trade by trade num i in main dataframe
        df.loc[prev_date:date, 'trade_num'] = i
        trades.loc[date, 'trade_num'] = i
        
        #Calculate returns for the trade
        calc_trade_returns(date, trades, df.loc[prev_date:date].copy())
        
        #Move the iterators forward
        i = i + 1
        prev_date = date
    
    #return on each trade row here, are really returns for the trade that
    #started on the date of the previous row, and ended on this date
    #better to have those returns listed with previous rows by shifting them up 
    #by one and then filling the last row nan with 0
    #assumption here is that the last row is anyway the closing trade wehere we
    #exit all positions, therefore returns there are zero anyway
    trades['return'] = trades['return'].shift(-1).fillna(0)
    trades["cum_return"] = (1+trades['return']).cumprod() - 1

    return trades

def get_trade_stats (df):
    stats = {}
    stats['mean'] = df.mean()
    stats['stddev'] = df.std()
    stats['median'] = df.median()
    stats['max'] = df.max()
    stats['min'] = df.min()

    return stats
    
def tearsheet (df):
    prep_dataframe(df)
    tearsheet = {}
    trades = get_trades(df)
    tearsheet["trading_days"] = len(np.unique(df.index.date))
    tearsheet["days_in_trade"] = len(np.unique(trades.index.date))
    tearsheet["num_trades"] = len(trades)
    if (tearsheet["num_trades"] > 0):
        
        trades['prev_peak_cum_return'] = trades['cum_return'].cummax()
        trades['drawdown_from_prev_peak'] = trades['cum_return'] - trades['prev_peak_cum_return']
        tearsheet["num_winning_trades"] = len(trades[trades['return'] > 0])
        tearsheet["num_losing_trades"] = len(trades[trades['return'] < 0])
        tearsheet["win_pct"] = tearsheet["num_winning_trades"]/tearsheet["num_trades"]
        tearsheet["return"] = trades.iloc[-1]["cum_return"]
        tearsheet["return per day in trade"] =     tearsheet["return"] / tearsheet["days_in_trade"]
        tearsheet["annualized return"] = tearsheet["return per day in trade"] * 250
        #tearsheet["return_fixed_bet"] = df["cum_bnh_returns"][-1]
        tearsheet['max_drawdown_from_0'] = trades['cum_return'].min()
        tearsheet['max_drawdown_from_peak'] = trades['drawdown_from_prev_peak'].min()
        tearsheet["average_per_trade_return"] = trades['return'].mean()
        tearsheet["std_dev_pertrade_return"] = trades['return'].std()
        tearsheet["sharpe_ratio"] = tearsheet['average_per_trade_return'] / tearsheet['std_dev_pertrade_return']
        tearsheet["skewness_pertrade_return"] = trades['return'].skew()
        tearsheet["kurtosis_pertrade_return"] = trades['return'].kurtosis()
        tearsheet["wins"] = get_trade_stats(trades.loc[trades['return'] > 0, 'return'])
        tearsheet["loss"] = get_trade_stats(trades.loc[trades['return'] < 0, 'return'])
        tearsheetdf = pd.DataFrame(tearsheet,index=[0])
        # drop the 'wins' and 'loss' columns
        tearsheetdf = tearsheetdf.drop(['wins', 'loss'], axis=1)
        
        tearsheet['trades'] = trades
    else:
        tearsheet["num_winning_trades"] = 0
        tearsheet["num_losing_trades"] = 0
        tearsheet["win_pct"] = 0
        tearsheet["return"] = 0
        tearsheet["return per day in trade"] = 0
        tearsheet["annualized return"] = 0
        tearsheet["return_fixed_bet"] = 0
        tearsheet["average_return"] = 0
        tearsheet["std_dev_return"] = 0
        tearsheet["skewness_pertrade_return"] = 0
        tearsheet["kurtosis_pertrade_return"] = 0
        tearsheetdf = pd.DataFrame(tearsheet,index=[0])
        
    trades.to_csv('trades.csv')
    df.to_csv('df.csv')
    return tearsheet,tearsheetdf
    