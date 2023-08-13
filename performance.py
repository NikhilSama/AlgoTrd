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
import math 
#cfg has all the config parameters make them all globals here
import cfg
globals().update(vars(cfg))

pd.options.mode.chained_assignment = None  # default='warn'

### Global Data frame prep functions that calculate positions and returns
### based on signal column

def calculate_bnh_returns (df):
    df['trade_price'] = df['exit_price'].fillna(df['entry_price']).fillna(df['Open'])
    df['bnh_returns'] = df["trade_price"].pct_change()
    #pct change gets the pct change from prev row open to this row open
    #we want this row bnh_returns to contain the returns for holding from this row open
    #to next row open, so we shift the column down by one
    df['bnh_returns'] = df['bnh_returns'].shift(-1)
    df['cum_bnh_returns'] = (1+df["bnh_returns"]).cumprod() - 1
    df['bnh_ln_returns'] = np.log(df['trade_price']/df['trade_price'].shift(1))
    df['cum_bnh_ln_returns'] = df['bnh_ln_returns'].cumsum()

    ## IF Call and Put option data exists
    if "Adj Close-C" in df.columns:
        df['bnh_returns-C'] = df["Open-C"].pct_change().shift(-1)
    if "Adj Close-P" in df.columns:
        df['bnh_returns-P'] = df["Open-P"].pct_change().shift(-1)
    return df


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
    return df

def prep_dataframe (df, close_at_end=True):
    df = duplicateDFRowsWithMultipleTrades(df)
    
    #calc returns if not already calculated
    if not set(['position','cum_bnh_returns']).issubset(df.columns):
        df = calculate_bnh_returns(df)
        
    #calc positions if not already calculated
    if not set(['position']).issubset(df.columns):
        df = calculate_positions (df, close_at_end)
    
    #calc strategy returns if not already calculated
    if not set(['strategy_returns','cum_strategy_returns']).issubset(df.columns):
        df = calculate_strategy_returns(df)
    return df
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


def insertSyntheticExitSignalforAbrubtDirectionChange(df):
    #Trade Calculators dont like position change from 1 to -1 directly, or vice versa
    # without an exit 0 signal in between.  So we add a 0 signal in between
    #artificially
    df['signal_change'] = df['signal']-df.shift()['signal']
    df['signal_change'] = df['signal_change'].fillna(0)

    valid_rows = df[(df['signal_change'] == 2) | (df['signal_change'] == -2)]

    # Create a copy of those rows
    duplicated_rows = valid_rows.copy()

    # Apply the rules to the original and duplicated rows
    for idx, row in duplicated_rows.iterrows():
        duplicated_rows.loc[idx, 'signal'] = 0
        # Set Exit price to next candle open, unless there is already an SL exit price there
        if np.isnan(df.loc[idx, 'exit_price']):
            df['Open_shifted'] = df['Open'].shift(-1)
            duplicated_rows.loc[idx, 'exit_price'] = df.loc[idx, 'Open_shifted']
            df = df.drop('Open_shifted', axis=1)

        df.loc[idx, 'exit_price'] = df.loc[idx, 'entry_price'] = np.nan
        
    # Change the index of duplicated rows to be one microsecond before the original row
    duplicated_rows.index = duplicated_rows.index - pd.Timedelta(microseconds=1)

    # Concatenate the original dataframe with the duplicated rows
    df = pd.concat([df, duplicated_rows]).sort_index()  
    df = df.drop('signal_change', axis=1)

    return df

def duplicateDFRowsWithMultipleTrades(df):
    # Some DF rows have multiple trades, exit_price and entry_price are set
    # by limit or SL order entries set in the previous row
    # Either we exited (via stoploss or target limit order trigger), 
    # and then entered in the same minute via CheckLongEntry or CheckShortEntry
    # or Vice versa (entered and then exited)
    # these rows become hard to handle in performance calculations, so we split them
    # into multiple rows, one for the first trade and one for the second trade.
    # Assert if both entry_price and exit_price are not 'nan' for any row
    # These prices are set by checkOrderStatus, which should only set one, not both
    assert not df[(~df['entry_price'].isna()) & (~df['exit_price'].isna())].shape[0], "Both entry_price and exit_price are not 'nan' for some rows"

    # Select rows where either entry_price or exit_price are not nan
    valid_rows = df[~(df['entry_price'].isna() & df['exit_price'].isna())]
    
    # Create a copy of those rows
    duplicated_rows = valid_rows.copy()

    # Apply the rules to the original and duplicated rows
    for idx, row in duplicated_rows.iterrows():
        if not pd.isna(row['exit_price']):
            duplicated_rows.loc[idx, 'signal'] = 0
        elif row['entry_price'] < 0:
            duplicated_rows.loc[idx, 'signal'] = -1
        elif row['entry_price'] > 0:
            duplicated_rows.loc[idx, 'signal'] = 1

        df.loc[idx, 'entry_price'] = np.nan
        df.loc[idx, 'exit_price'] = np.nan
    
        if df.loc[idx, 'signal'] == duplicated_rows.loc[idx, 'signal']:
            df.loc[idx, 'signal'] = np.nan 
    
    # Change the index of duplicated rows to be one microsecond before the original row
    duplicated_rows.index = duplicated_rows.index - pd.Timedelta(microseconds=10)

    # Concatenate the original dataframe with the duplicated rows
    df = pd.concat([df, duplicated_rows]).sort_index()  
    
    #Trade Calculators dont like position change from 1 to -1, or vice versa
    # without an exit 0 signal in between.  So we add a 0 signal in between
    #artificially
    df = insertSyntheticExitSignalforAbrubtDirectionChange(df)

    #Shift entry and exit prices down one row. they are currently in the signal row, but the trade starts at the next row
    #where position change happens. So we shift them down one row
    df['entry_price'] = abs(df['entry_price']).shift(1)
    df['exit_price'] = df['exit_price'].shift(1)
  
    return df
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
    #they will be included in ne 0xt trade.  Therefore set strategy returns for 
    #last candle here to 0


# Theoretically this method below shouuld work, in practice it has some errors
# when row returns are negative.
# for Example look at NIFTY WEEKLY OPTION DATA @
#        27-02 10:01 AM 47 Position: -1.0 @ 125.2        Return:-2.88%   CUM=>-2.88%
#                11:02:00 AM: EXIT @ 128.80(-3.6 or -2.88%)
# This cumprod calculation gives a return or -7% for this .. debug later, for now just 
# use the simpler alternate method belwo
#
#    ticker_data.iloc[-1, ticker_data.columns.get_loc('strategy_returns')] = 0
#    ticker_data['cum_trade_returns'] =  (1+ticker_data['strategy_returns']).cumprod() - 1
#   trades.loc[date, 'return'] = ticker_data.iloc[-1]['cum_trade_returns']

    tradeEntry = ticker_data.iloc[0, ticker_data.columns.get_loc('entry_price')] if 'entry_price' in ticker_data.columns else float('nan')
    tradeEntry = ticker_data.iloc[0, ticker_data.columns.get_loc('Open')] if np.isnan(tradeEntry) else tradeEntry
    tradeEntry = abs(tradeEntry)
    
    
    tradeExit  = ticker_data.iloc[-1, ticker_data.columns.get_loc('exit_price')] if 'exit_price' in ticker_data.columns else float('nan')
    tradeExit  = ticker_data.iloc[-1, ticker_data.columns.get_loc('Open')] if np.isnan(tradeExit) else tradeExit
    
    tradePNL = (tradeExit - tradeEntry) * ticker_data.iloc[0, ticker_data.columns.get_loc('position')]
    
    tradeReturn = tradePNL/tradeEntry
    # subtract trade costs from tradePNL
    tradeReturn = tradeReturn - (cfgTradingCost*2) if tradeReturn != 0 else tradeReturn#2 trades, entry and exit
    trades.loc[date, 'return'] = tradeReturn
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
    
    # remove trades where we just exited, and didnt take a position
    # also removes the last exit which na we just filled with 0
    #trades = trades[trades['return'] != 0]

    trades["cum_return"] = (1+trades['return']).cumprod() - 1
    trades["sum_return"] = trades['return'].cumsum() #more appropriate since our bet_size doesnt change based on prev trade performance

    return trades

def get_trade_stats (df):
    stats = {}
    stats['mean'] = df.mean()
    stats['stddev'] = df.std()
    stats['median'] = df.median()
    stats['max'] = df.max()
    stats['min'] = df.min()

    return stats

def addDailyReturns(trades,tearsheet):
    # assuming 'trades' is the name of your DataFrame
    daily_returns = trades['return'].resample('D').sum()
    daily_returns = daily_returns[daily_returns != 0]
    
    avg_return = daily_returns.mean()
    best_return = daily_returns.max()
    worst_return = daily_returns.min()
    days = daily_returns.count()
    winning_days = daily_returns[daily_returns > 0].count()
    losing_days = daily_returns[daily_returns < 0].count()

    # get the date of the best and worst daily returns
    best_date = daily_returns.idxmax().strftime('%Y-%m-%d')
    worst_date = daily_returns.idxmin().strftime('%Y-%m-%d')

    # print the results
    #print(f"Daily return: avg:{avg_return:.2%} sharpe:{avg_return/daily_returns.std():.2f} kurtosis:{daily_returns.kurtosis():.2f} skew:{daily_returns.skew():.2f} max:{best_return:.2%} min:{worst_return:.2%}")

    # print(f"Best daily return ({best_date}): {best_return:.2%}")
    # print(f"Worst daily return ({worst_date}): {worst_return:.2%}")
    
    tearsheet["avg_daily_return"] = avg_return
    tearsheet["std_daily_return"] = daily_returns.std()
    tearsheet["sharpe_daily_return"] = math.sqrt(252) * avg_return/tearsheet["std_daily_return"]
    tearsheet["kurtosis_daily_return"] = daily_returns.kurtosis()
    tearsheet["skew_daily_return"] = daily_returns.skew()

    tearsheet["best_daily_return"] = best_return
    tearsheet["best_daily_return_date"] = best_date
    tearsheet["worst_daily_return"] = worst_return
    tearsheet["worst_daily_return_date"] = worst_date

    tearsheet["num_days"] = days
    tearsheet["numWinningDays"] = winning_days
    tearsheet["numLosingDays"] = losing_days
    
    tearsheet["daily_returns"] = daily_returns
    
    return tearsheet



def tearsheet (df):
    df = prep_dataframe(df)

    tearsheet = {}
    trades = get_trades(df)
    tearsheet["trading_days"] = len(np.unique(df.index.date))
    tearsheet["days_in_trade"] = len(np.unique(trades.index.date))
    if (len(trades) > 0):
        
        tearsheet["num_trades"] = len(trades[trades['return'] != 0])
        trades['prev_peak_cum_return'] = trades['cum_return'].cummax()
        trades['drawdown_from_prev_peak'] = trades['cum_return'] - trades['prev_peak_cum_return']
        trades['prev_peak_sum_return'] = trades['sum_return'].cummax()
        trades['drawdown_from_prev_peak_sum'] = trades['sum_return'] - trades['prev_peak_sum_return']
        tearsheet = addDailyReturns(trades, tearsheet)
        tearsheet["num_winning_trades"] = len(trades[trades['return'] > 0])
        tearsheet["num_losing_trades"] = len(trades[trades['return'] < 0])
        tearsheet["win_pct"] = tearsheet["num_winning_trades"]/tearsheet["num_trades"]
        tearsheet["return"] = trades.iloc[-1]["sum_return"]
        tearsheet["return per day in trade"] =     tearsheet["return"] / tearsheet["days_in_trade"]
        tearsheet["annualized return"] = tearsheet["return per day in trade"] * math.sqrt(252)
        #tearsheet["return_fixed_bet"] = df["cum_bnh_returns"][-1]
        tearsheet['max_drawdown_from_0'] = trades['cum_return'].min()
        tearsheet['max_drawdown_from_0_sum'] = trades['sum_return'].min()
        tearsheet['max_drawdown_from_peak'] = trades['drawdown_from_prev_peak'].min()
        tearsheet['max_drawdown_from_prev_peak_sum'] = trades['drawdown_from_prev_peak_sum'].min()
        tearsheet["average_per_trade_return"] = trades[trades['return'] != 0]['return'].mean()
        tearsheet["std_dev_pertrade_return"] = trades[trades['return'] != 0]['return'].std()
        tearsheet["sharpe_ratio"] = round(tearsheet['avg_daily_return'] * math.sqrt(252) / tearsheet['std_daily_return'],1)
        tearsheet['calamar_ratio'] = -round(tearsheet['avg_daily_return'] * 252 / tearsheet['max_drawdown_from_prev_peak_sum'],1)
        tearsheet["skewness_pertrade_return"] = trades[trades['return'] != 0]['return'].skew()
        tearsheet["kurtosis_pertrade_return"] = trades[trades['return'] != 0]['return'].kurtosis()
        tearsheet["wins"] = get_trade_stats(trades.loc[trades['return'] > 0, 'return'])
        tearsheet["loss"] = get_trade_stats(trades.loc[trades['return'] < 0, 'return'])
        tearsheet["normalized_hit_ratio"] = (tearsheet["num_winning_trades"]*tearsheet['wins']['mean']) / ((tearsheet["num_winning_trades"]*tearsheet['wins']['mean']) + (tearsheet["num_losing_trades"]*abs(tearsheet['loss']['mean'])))

        tearsheetdf = pd.DataFrame(tearsheet,index=[0])
        # drop the 'wins' and 'loss' columns
        tearsheetdf = tearsheetdf.drop(['wins', 'loss'], axis=1)
        
        tearsheet['trades'] = trades
    else:
        tearsheet["num_trades"] = 0
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
        
    trades.to_csv('trades-renko.csv')
    # df.to_csv('df.csv')
    return tearsheet,tearsheetdf,df
    