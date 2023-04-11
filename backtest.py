#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 20:18:29 2023

@author: nikhilsama
"""
isMain = True if __name__ == '__main__' else False

from datetime import date,timedelta
import datetime
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
from plotting import plot_backtest,plot_stock_and_option
import itertools 
from sqlalchemy import create_engine
import mysql.connector
import backtest_log_setup



import utils
import sys

mydb = None

def getTaskNameFromArgs():
    argString = ''
    args = sys.argv[1:]
    arg_dict = {}
    for arg in args:
        key, value = arg.split(':')
        if key in ["cacheTickData","zerodha_access_token","dbhost","dbuser", "dbpass" ,"dbname"]:
            continue
        argString = argString + ' ' + arg
    return argString.strip()

def mark_task_complete():
    global mydb
    mydb = mysql.connector.connect(
        host="trading.ca6bwmzs39pr.ap-south-1.rds.amazonaws.com",
        user="trading",
        password="trading123",
        database="trading"
    )

    task_name = getTaskNameFromArgs()
    mycursor = mydb.cursor()
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sql = "UPDATE tasks SET status = 1, completed_time = %s WHERE task_name = %s"
    val = (now,task_name)
    mycursor.execute(sql, val)
    #print(mycursor.statement)
    mydb.commit()
    mydb.close()


#cfg has all the config parameters make them all globals here
import cfg
globals().update(vars(cfg))

# set timezone to IST
ist = pytz.timezone('Asia/Kolkata')

tickers = td.get_sp500_tickers()
nifty = td.get_nifty_tickers()
index_tickers = td.get_index_tickers()
firstTradeTime = datetime.datetime(2023, 3,31, 9, 0, tzinfo=ist)
zgetFrom = firstTradeTime - timedelta(days=cfgHistoricalDaysToGet)
zgetTo = datetime.datetime(2023, 4,11, 15, 30, tzinfo=ist)

def zget(t,s,e,i):
    #Get latest minute tick from zerodha
    df = downloader.zget(s,e,t,i,includeOptions=includeOptions)
    df = downloader.zColsToDbCols(df)
    if utils.isOption(t):
        df['Volume'] = 0
    return df
def zgetNDays(t,n,e=datetime.datetime.now(ist),i="minute"):
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

def printTearsheet(tearsheet):
    print(f"Total Return: {tearsheet['return']:.2%}")
    print("Sharpe: ", tearsheet['sharpe_ratio'])
    print("Num Trades: ", tearsheet['num_trades'])
    print(f"Avg Return per day: {tearsheet['avg_daily_return']:.2%}")
    print("TRADES: ")
    for index,trade in tearsheet['trades'].iterrows():
        if trade['position'] == 0:
            diff = round(trade['Open'] - entry_price,2)
            print(f"\t\t{utils.timeToString(index,date=False,time=True)}: EXIT @ {trade['Open']:.2f}({diff} or {diff/entry_price:.2%})")
        else:
            entry_price = trade['Open']
            print(f"\t{utils.timeToString(index,date=True)} Position: {trade['position']} @ {trade['Open']} Return:{trade['return']:.2%} \tCUM=>{trade['sum_return']:.2%}")
    print("Days: ")
    daily_returns = tearsheet['trades']['return'].resample('D').sum()
    for index,day in daily_returns.iteritems():
        print(f"\t {utils.timeToString(index,date=True,time=False)} Return:{day:.2%}")
    
def backtest(t,i='minute',start = zgetFrom, end = zgetTo, \
            exportCSV=False, tradingStartTime = firstTradeTime, \
            applyTickerSpecificConfig = True):
    #perfTIME = time.time()    
    #startingTime = perfTIME
    df = zget(t,start,end,i=i)
    if df.empty:
        print(f"No data foc {t}")
        return
    if len(df) < cfgMaxLookbackCandles:
        print(f"Skipping {t} as it has {len(df)} less than {cfgMaxLookbackCandles} candles at {tradingStartTime}")
        print(df)
    else:#trim the df to maxlookbackcandles
        df_head = df.loc[:firstTradeTime].iloc[-cfgMaxLookbackCandles:]
        df_tail = df.loc[firstTradeTime:]
        df = pd.concat([df_head, df_tail])

    # print (len(df))
    # print(len(df_head))
    # print(len(df_tail))
    # exit(0)
    #df = zgetNDays(t,days,i=i)
    #perfTime = perfProfiler("ZGET", perfTIME)
    dataPopulators = [signals.populateBB, signals.populateADX, signals.populateOBV]
    signalGenerators = [
                        signals.getSig_BB_CX
                        ,signals.getSig_ADX_FILTER
                        ,signals.getSig_MASLOPE_FILTER
                        ,signals.getSig_OBV_FILTER
                        ,signals.getSig_exitAnyExtremeADX_OBV_MA20_OVERRIDE
                        ,signals.getSig_followAllExtremeADX_OBV_MA20_OVERRIDE
                        #,signals.followTrendReversal
                        ,signals.exitTrendFollowing
                        ]
    overrideSignalGenerators = []   
    
    signals.applyIntraDayStrategy(df,dataPopulators,signalGenerators,
                                  overrideSignalGenerators, 
                                  tradingStartTime=tradingStartTime,
                                  applyTickerSpecificConfig=applyTickerSpecificConfig)
    #perfTIME = perfProfiler("SIGNAL GENERATION", perfTIME)


    tearsheet,tearsheetdf = perf.tearsheet(df)
    printTearsheet(tearsheet) if isMain else None
    # print(f'Total Return: {tearsheet["return"]*100}%')
    # print(f'Sharpe: {tearsheet["sharpe_ratio"]}')
    #print(f'Num Trades: {tearsheet["num_trades"]}')
    # print(f'Avg Return Per Trade: {tearsheet["average_per_trade_return"]*100}%')
    # print(f'Std Dev of Returns: {tearsheet["std_dev_pertrade_return"]*100}%')
    # print(f'Skewness: {tearsheet["skewness_pertrade_return"]}')
    # print(f'Kurtosis: {tearsheet["kurtosis_pertrade_return"]}')
    # pp = pprint.PrettyPrinter(indent=4)
    # pp.pprint(tearsheet)
    #perfTIME = perfProfiler("Tearsheet took", perfTIME)

    if (exportCSV == True):
        df.to_csv("export.csv")
   # perfTIME = perfProfiler("to CSV", perfTIME)
    #perfTIME = perfProfiler("Backtest:", startingTime)

    if (plot == [] or (not 'trades' in tearsheet.keys())):
        return tearsheetdf
    
    if 'options' in plot:
        plot_stock_and_option(df.loc[firstTradeTime:])
        
    if 'adjCloseGraph' in plot:
        plot_backtest(df.loc[firstTradeTime:],tearsheet['trades'])
    
    # print (f"END Complete {datetime.datetime.now(ist)}")
    return tearsheetdf

def backtest_daybyday(t,i='minute',exportCSV=False):
    startingTime = time.time()  

    start_time = datetime.time(hour=9)
    end_time = datetime.time(hour=16, minute=30)

    dates = []
    curr_date = firstTradeTime.date()
    tearsheets = pd.DataFrame()
    while curr_date <= zgetTo.date():
        start_date = datetime.datetime.combine(curr_date, start_time, tzinfo=ist)
        zget_start = start_date - timedelta(days=10)
        end_date = datetime.datetime.combine(curr_date, end_time, tzinfo=ist)

        tearsheetdf = backtest(t,i=i,start=zget_start, end=end_date, tradingStartTime = start_date, exportCSV=exportCSV)
        tearsheets = pd.concat([tearsheets, tearsheetdf])
        dates.append((start_date, end_date))
        curr_date += timedelta(days=1)
    # pp = pprint.PrettyPrinter(indent=4)
    # pp.pprint(dates)
    print(f"Total Return: {tearsheets['return'].sum()*100}%")
    print(f"max day return: {tearsheets['return'].max()*100}%")
    print(f"min day return: {tearsheets['return'].min()*100}%")
    print(f"avg day return: {tearsheets['return'].mean()*100}%")
    print(f"std dev day return: {tearsheets['return'].std()*100}%")
    print(f"kurtosis day return: {tearsheets['return'].kurtosis()}")
    print(f"skew day return: {tearsheets['return'].skew()}")
    perfProfiler("Total:", startingTime)

# Plot the graph of closing prices for the array of tickers provided
# and the interval provided and the number of days provided
def plot_options(uticker, tickers,i='minute', 
         days=30, e=datetime.datetime.now(ist)):
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
        s = datetime.datetime.now(ist)-timedelta(days=i)
        df = zgetNDays(t,days,s)
        if(len(df)):
            df = signals.bollinger_band_cx(df)
            tearsheet,tearsheetdf = perf.tearsheet(df)
            change = getTotalChange(df)
            ret = round(tearsheet['return'] *100,2)
            print(f"{t} Day:{s} Return:{ret}% Change; {change}%")

def performanceToCSV(performance):
    ## Add in the config variables it was called with from process combinator 
    args = sys.argv[1:]
    for arg in args:
        key, value = arg.split(':')
        if key in ['zerodha_access_token','dbuser','dbpass','cacheTickData', 'dbname', 'dbhost']:
            continue
        performance[key] = value
    #perfFileName = utils.fileNameFromArgs('Data/backtest/combo/niftyPerf-')
    #performance.to_csv(perfFileName)
    return performance

def backtestCombinator():
        
    performance = pd.DataFrame()
    
    ma_slope_threshes = [0.5, 1, 1.5]
    ma_slope_thresh_yellow_multipliers = [0.5,0.7,0.9]
    # obv_osc_threshes = [0.1, 0.2, 0.4]
    # obv_osc_thresh_yellow_multipliers = [0.7, 0.9, 1]
    # obv_ma_lens = [10,20,30]

    # ma_slope_threshes = [0.5]
    # ma_slope_thresh_yellow_multipliers = [0.5]
    # ma_slope_slope_threshes = [0.1]
    obv_osc_threshes = [0.1]
    obv_osc_thresh_yellow_multipliers = [0.9]
    obv_ma_lens = [20]

    # This loop will run 3^4 = 89 times; each run will be about 
    # 3 second, so total 267 seconds = 4.5 minutes
    # run a combo 
    
    # when done w for loop write results to csv and mark done in db 
    # add to fname string and csv file "maSlopeThresh:{ma_slope_thresh} maSlopeThreshYellowMultiplier:{ma_slope_thresh_yellow_multiplier} maSlopeSlopeThresh:{ma_slope_slope_thresh} obvOscThresh:{obv_osc_thresh} obvOscThreshYellowMultiplier:{obv_osc_thresh_yellow_multiplier} obvOscSlopeThresh:{ovc_osc_slope_thresh} overrideMultiplier:{override_multiplier}"
    # continue to next combo
    # output fname and csv row to contain timeframe in 
    # start and end times, symbol, and all the parameters
        
    for params in itertools.product(ma_slope_threshes, ma_slope_thresh_yellow_multipliers,
                                 obv_osc_threshes, obv_osc_thresh_yellow_multipliers,obv_ma_lens):
        ma_slope_thresh, ma_slope_thresh_yellow_multiplier, obv_osc_thresh, obv_osc_thresh_yellow_multiplier, obv_ma_len \
             = params

        signals.updateCFG(ma_slope_thresh, ma_slope_thresh_yellow_multiplier, \
                         obv_osc_thresh, \
                         obv_osc_thresh_yellow_multiplier, obv_ma_len)
        tearsheetdf = backtest(cfgTicker,'minute',exportCSV=False,
                               applyTickerSpecificConfig=False)
        
        # Add in config variables we are looping through to the tearsheetdf
        tearsheetdf['ma_slope_thresh'] = ma_slope_thresh
        tearsheetdf['ma_slope_thresh_yellow_multiplier'] = ma_slope_thresh_yellow_multiplier
        tearsheetdf['obv_osc_thresh'] = obv_osc_thresh
        tearsheetdf['obv_osc_thresh_yellow_multiplier'] = obv_osc_thresh_yellow_multiplier
        tearsheetdf['obv_ma_len'] = obv_ma_len
        tearsheetdf['ticker'] = 'NIFTY23APRFUT'
        tearsheetdf['interval'] = 'minute'
        tearsheetdf['startTime'] = firstTradeTime
        tearsheetdf['endTime'] = zgetTo
        tearsheetdf['duration_in_days'] = (zgetTo - zgetFrom).days
        
        performance = pd.concat([performance, tearsheetdf])

    
    # write the DataFrame to a SQL table
    # Connect to the MySQL database
    #HACK THIS performanceTOCSV also parses arv, and adds it to the performance df
    performance = performanceToCSV(performance)
    mark_task_complete()
    # create a database connection (performance to CSV adds the 
    # argv variables as well to the performance df)
    engine = create_engine('mysql+pymysql://trading:trading123@trading.ca6bwmzs39pr.ap-south-1.rds.amazonaws.com/trading')
    performance.to_sql('performancev4', engine, if_exists='append')
    engine.dispose()

if isMain:
    #backtestCombinator()       
    #plot_options(['ASIANPAINT'],10,'minute')
    backtest('NIFTY23APR17300CE','minute')
    #backtest(cfgTicker,'minute')
    #backtest_daybyday('NIFTY23APRFUT','minute')

    #backtest('HDFCLIFE','minute',adxThreh=25)
    #backtest('ASIANPAINT','minute',adxThreh=25)
    #backtest('HDFCLIFE','minute',adxThreh=30)
    #backtest('ADANIENT','minute',adxThreh=30)
    #compareDayByDayPerformance('ONGC')
    
    #plot('INFY',['ASIANPAINT23MAR2840PE','ASIANPAINT23MAR2840CE'],i='minute', days=3,e=datetime.datetime.now(ist)-timedelta(days=15))   

    # print hello
    # print hello

#NIFTY2341317500PE