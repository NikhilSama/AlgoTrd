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
from plotting import plot_backtest,plot_stock_and_option,plot_trades,plot_returns_on_nifty
import itertools 
from sqlalchemy import create_engine
import mysql.connector
import backtest_log_setup
import os
import pickle
from DatabaseLogin import DBBasic
from dateutil.relativedelta import relativedelta

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
firstTradeTime = datetime.datetime(2022,4,1,9,0) if cfgZGetStartDate == None else cfgZGetStartDate
# firstTradeTime = datetime.datetime(2023,3,14,9,0) if cfgZGetStartDate == None else cfgZGetStartDate
firstTradeTime = ist.localize(firstTradeTime)
zgetFrom = firstTradeTime - timedelta(days=cfgHistoricalDaysToGet)
zgetTo = datetime.datetime(2023,4,1,15,30) if cfgZGetStartDate == None else cfgZGetStartDate +  relativedelta(months=11)
# zgetTo = datetime.datetime(2023,3,14,15,30) if cfgZGetStartDate == None else cfgZGetStartDate +  relativedelta(months=11)
zgetTo = ist.localize(zgetTo)


# def mmReturns(row,df):
#     i = df.index.get_loc(row.name)

#     lastCandleHigh = df.iloc[i-1, df.columns.get_loc('High')]
#     lastCandleLow = df.iloc[i-1, df.columns.get_loc('Low')]
    
#     High = row['High']
#     Low = row['Low']
    
#     if High >= lastCandleHigh 
# def stratMarketMakeLastCandle():
#     df = dbget('NIFTYWEEKLYOPTION',firstTradeTime,zgetTo)
#     df['signal'] = df.apply(mmReturns, 
#         args=(df), axis=1)


    
def dbget(t,s,e,offset=None):
    df = downloader.getCachedTikerData(f'NIFTYITMCALL{offset if offset is not None else ""}',s,e,'minute')
    if not df.empty:
        return df 
    print("getting from db")
    db = DBBasic()
    q = f'select * from niftyITMCall{offset if offset is not None else ""} where date between "'+s.strftime('%Y-%m-%d %H:%M:%S')+'" and "'+e.strftime('%Y-%m-%d %H:%M:%S')+'"'
    df = db.frmDB(q)
    df.drop('Open Interest', axis = 1, inplace = True) if 'Open Interest' in df.columns else None

    df['symbol'] = t+(str(offset) if offset is not None else '')
    downloader.loadTickerCache(df,'NIFTYITMCALL{offset}',s,e,'minute')
    return df
    
def zget(t,s,e,i):
    
    if t == 'NIFTYWEEKLYOPTION': 
        pfname = 'Data/NIFTYOPTIONSDATA/contNiftyWeeklyOptionDF.pickle'
        # if os.path.isfile(pfname):
        #     with open(pfname, "rb") as f:
        #         df = pickle.load(f)
        #         return df

        fname = 'Data/NIFTYOPTIONSDATA/contNiftyWeeklyOptionDF.csv'
        df = pd.read_csv(fname)
        df.set_index('date', inplace=True)
        df.index = pd.to_datetime(df.index)
        # Set the seconds values to 0 in the DateTimeIndex
        df.index = df.index.floor('T')
        df.insert(0, 'i', range(1, 1 + len(df)))
        # Filter the DataFrame between start_date and end_date
        # print(df)

        df = df[(df.index >= s) & (df.index <= e)]
        with open(pfname,"wb") as f:
            pickle.dump(df,f)
        return df
    #Get latest minute tick from zerodha
    df = downloader.zget(s,e,t,i,includeOptions=includeOptions)
    df = downloader.zColsToDbCols(df)
    
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
    if tearsheet is None or tearsheet['num_trades'] == 0:
        print("No trades")
        return
    print("TRADES: ")
    prevPos = 0
    for index,trade in tearsheet['trades'].iterrows():
        if trade['position'] == 0:
            diff = round(trade['Open'] - entry_price,2)*prevPos
            print(f"\t\t{utils.timeToString(index,date=False,time=True)}: EXIT @ {trade['Open']:.2f}({diff} or {diff/entry_price:.2%})")
        else:
            entry_price = trade['Open']
            print(f"\t{utils.timeToString(index,date=True)} {trade['i']} Position: {trade['position']} @ {trade['Open']} \tReturn:{trade['return']:.2%} \tCUM=>{trade['sum_return']:.2%}")
        prevPos = trade['position']
    print("Days: ")
    daily_returns = tearsheet['trades']['return'].resample('D').sum()
    dailyDF = pd.DataFrame(columns=['Date','Day','Open','dayReturn','return'])
    for index,day in daily_returns.iteritems():
        if day == 0:
            continue
        trades_on_d = tearsheet['trades'].loc[tearsheet['trades'].index.date == index.date()]
        first_row = trades_on_d.iloc[0]
        last_row = trades_on_d.iloc[-1]
        dayOpen = first_row['Open']
        dayClose = last_row['Open']
        dailyDF.loc[len(dailyDF)] = [index,index.strftime('%a'),dayOpen,(dayClose-dayOpen)/dayOpen,day]
        print(f"\t {utils.timeToString(index,date=True,time=False)}({index.strftime('%a')}) Open:{dayOpen} Close:{dayClose} ({(dayClose-dayOpen)/dayOpen:.1%}) Return:{day:.2%}") if day != 0 else None
    dailyDF.to_csv("dailyReturns.csv")
    print(f"Total Return: {tearsheet['return']:.2%}")
    # print(f"Drawdown: {tearsheet['max_drawdown_from_0_sum']:.2%}")
    print(f"Drawdown from Prev Peak: {tearsheet['max_drawdown_from_prev_peak_sum']:.2%}")
    print("Sharpe: ", tearsheet['sharpe_ratio'])
    print("Calamar: ", tearsheet['calamar_ratio'])
    print("Num Trades: ", tearsheet['num_trades'])
    print(f"Avg Return per day: {tearsheet['avg_daily_return']:.2%}")
    print(f"Worst Day ({tearsheet['worst_daily_return_date']}): {tearsheet['worst_daily_return']:.2%}")
    print(f"Best Day ({tearsheet['best_daily_return_date']}): {tearsheet['best_daily_return']:.2%}")
    
def backtest(t,i='minute',start = zgetFrom, end = zgetTo, \
            exportCSV=True, tradingStartTime = firstTradeTime, \
            applyTickerSpecificConfig = True, signalGenerators = None,
            src = 'z'):
    perfTIME = time.time()    
    startingTime = perfTIME
    if src == 'db':
        df = dbget(t,start,end)
    else:
        df = zget(t,start,end,i=i)
    if df.empty:
        print(f"No data for {t} start:{start} end:{end} i:{i}")
        return

    if len(df) < cfgMaxLookbackCandles:
        print(f"Skipping {t} as it has {len(df)} less than {cfgMaxLookbackCandles} candles at {tradingStartTime}")
        print(df)
    else:#trim the df to maxlookbackcandles
        df_head = df.loc[:firstTradeTime].iloc[-cfgMaxLookbackCandles:]
        df_tail = df.loc[firstTradeTime:]
        df = pd.concat([df_head, df_tail])
    #df['Volume']=0
    # print (len(df))
    # print(len(df_head))
    # print(len(df_tail))
    # exit(0)
    #df = zgetNDays(t,days,i=i)
    perfTime = perfProfiler("ZGET", perfTIME)
    dataPopulators = {
        'daily': [
            signals.populateATR,
            signals.populateRenko,
            # signals.populateBB,     
            # signals.populateADX, 
            # signals.populateSuperTrend,
            # signals.populateOBV,
            signals.vwap,
            signals.populateSVP
            # signals.populateCandleStickPatterns
        ], 
        'hourly': [
        ]
    }
    
    signalGenerators = [
                    #    signals.randomSignalGenerator
                    #    signals.followSuperTrend
                        signals.followSVP
                        # signals.followRenkoWithOBV
                        #signals.followRenkoWithTargetedEntry
                        # signals.followObvMA
                        # ,signals.exitObvAdxMaTrend
                        # signals.followMAandADX
                     #   signals.justFollowFastMA
                    #       signals.getSig_BB_CX
                    #      ,signals.getSig_ADX_FILTER
                    #      ,signals.getSig_MASLOPE_FILTER
                    # #      ,signals.getSig_OBV_FILTER
                    # #     ,signals.getSig_exitAnyExtremeADX_OBV_MA20_OVERRIDE
                    #       ,signals.getSig_followAllExtremeADX_OBV_MA20_OVERRIDE
                    # # # #     #,signals.followTrendReversal
                    #       ,signals.exitTrendFollowing
                           # signals.fastSlowMACX
                    #       ,signals.exitCandleStickReversal
                        #    ,signals.exitTargetOrSL

                        ] if signalGenerators is None else signalGenerators
    overrideSignalGenerators = []   
    
    df = signals.applyIntraDayStrategy(df,dataPopulators,signalGenerators,
                                  overrideSignalGenerators, 
                                  tradeStartTime=tradingStartTime,
                                  applyTickerSpecificConfig=applyTickerSpecificConfig)
    perfTIME = perfProfiler("SIGNAL GENERATION", perfTIME)


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
   # perfTIME = perfProfiler("to CSV", perfTIME)expo
    #perfTIME = perfProfiler("Backtest:", startingTime)
  
    if (plot == []):
        return tearsheetdf
    
    if 'options' in plot:
        plot_stock_and_option(df.loc[firstTradeTime:])
        
    if 'adjCloseGraph' in plot:
        plot_backtest(df.loc[firstTradeTime:],tearsheet['trades'] if 'trades' in tearsheet else None)
    
    if 'trade_returns' in plot:
        plot_trades(tearsheet['trades'])

    if 'plot_returns_on_nifty' in plot:
        plot_returns_on_nifty(df)
    # print (f"END Complete {datetime.datetime.now(ist)}")
    return tearsheetdf

def oneThousandRandomTests():
#     Mean: -11.99%
#     std dev: 95.73%
    signalGenerators = [signals.randomSignalGenerator]
    results = []
    for i in range(1000):
        ts = backtest('NIFTY2341317700CE', signalGenerators=signalGenerators)
        if ts is not None and 'return' in ts.keys():
            results.append(ts['return'])
    print(f"Mean: {np.mean(results):.2%}")
    print(f"std dev: {np.std(results):.2%}")
    
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
        if key in ['zerodha_access_token','dbuser','dbpass','cacheTickData', 'dbname', 'dbhost','cfgZGetStartDate','cfgZGetStartDate']:
            continue
        performance[key] = value
    #perfFileName = utils.fileNameFromArgs('Data/backtest/combo/niftyPerf-')
    #performance.to_csv(perfFileName)
    return performance
def backtestCombinator2():
    performance = pd.DataFrame()
    ma20lens = [10]
    fastMALens  = [2,5,7,10,15,20,25,30]
    adxLens = [10,14,20]
    adxThresholds = [10,15,20,25,30,35,40,45,90]
    fastMAThreshs = [0,0.005,0.01,0.02,0.03,0.04,.05,0.1]
    for ma20len in ma20lens:
        for fastMALen in fastMALens:
            for adxLen in adxLens:
                for adxThreshold in adxThresholds:
                    for fastMaThresh in fastMAThreshs:
                        cfg = {'maLen':ma20len, 'fastMALen':fastMALen,
                            'adxLen':adxLen, 'adxThresh':adxThreshold,
                            'cfgFastMASlpThresh': fastMaThresh}
                        signals.updateCFG2(cfg)
                        tearsheetdf = backtest(cfgTicker,'minute',exportCSV=False,
                                        applyTickerSpecificConfig=False)
                        tearsheetdf['fastMALen'] = fastMALen
                        tearsheetdf['cfgFastMASlpThresh'] = fastMaThresh
                        tearsheetdf['adxLen'] = adxLen
                        tearsheetdf['adxThresh'] = adxThresh
                        performance = pd.concat([performance, tearsheetdf])
    engine = create_engine('mysql+pymysql://trading:trading123@trading.ca6bwmzs39pr.ap-south-1.rds.amazonaws.com/trading')
    performance.to_sql('PerfNiftyFollowFast2', engine, if_exists='append')
    engine.dispose()

def backtestCombinator(src='z'):
        
    performance = pd.DataFrame()    
    #ma slopes threshold overloaded and used as fastma slope threshold
    #for justFOllwoFastMA strategy
    ma_slope_threshes = [0]
    ma_slope_periods = [1]
    cfgRenkoBrickMultipliers = [1,1.5,2,3,4]
    atrlens = [10]
    cfgRenkoBrickMultiplierLongTargets = [1,2,3,4,5,6]
    cfgRenkoBrickMultiplierLongSLs = [0.5,1,1.5,2]
    cfgRenkoBrickMultiplierShortTargets = [1,2,3,4,5,6]
    cfgRenkoBrickMultiplierShortSLs = [0.5,1,1.5,2]
    #ma slopes for bb + trending normal strategy 
    #ma_slope_threshes = [0.01,0.05,0.1,0.3,0.7,1] #for MA 15 0.3 seems the best, with 0.1 a close second; For MA go with 1 (experiment with higher values)
    ma_slope_thresh_yellow_multipliers = [0.6]
    # obv_osc_threshes = [0.1, 0.2, 0.4]
    # obv_osc_thresh_yellow_multipliers = [0.7, 0.9, 1]
    # obv_ma_lens = [10,20,30]

    # ma_slope_threshes = [0.5]
    #ma_slope_thresh_yellow_multipliers = [0]
    #ma_slope_slope_threshes = [0.1]
    obv_osc_threshes = [0.1]
    obv_osc_thresh_yellow_multipliers = [0]
    obv_ma_lens = [20]
    signal_generators = [ # Most returns come from trending only; although bb can add some cherry on top for some cases
                        [signals.followRenkoWithOBV,signals.exitTargetOrSL]
                        # [signals.getSig_BB_CX
                        #  ,signals.getSig_ADX_FILTER
                        #  ,signals.getSig_MASLOPE_FILTER],
                        # [signals.getSig_followAllExtremeADX_OBV_MA20_OVERRIDE
                        #   ,signals.exitTrendFollowing],
                        # [signals.getSig_BB_CX
                        #  ,signals.getSig_ADX_FILTER
                        #  ,signals.getSig_MASLOPE_FILTER
                        #  ,signals.getSig_followAllExtremeADX_OBV_MA20_OVERRIDE
                        #  ,signals.exitTrendFollowing]
                        ]

    # This loop will run 3^4 = 89 times; each run will be about 
    # 3 second, so total 267 seconds = 4.5 minutes
    # run a combo 
    
    # when done w for loop write results to csv and mark done in db 
    # add to fname string and csv file "maSlopeThresh:{ma_slope_thresh} maSlopeThreshYellowMultiplier:{ma_slope_thresh_yellow_multiplier} maSlopeSlopeThresh:{ma_slope_slope_thresh} obvOscThresh:{obv_osc_thresh} obvOscThreshYellowMultiplier:{obv_osc_thresh_yellow_multiplier} obvOscSlopeThresh:{ovc_osc_slope_thresh} overrideMultiplier:{override_multiplier}"
    # continue to next combo
    # output fname and csv row to contain timeframe in 
    # start and end times, symbol, and all the parameters
        
    for params in itertools.product(ma_slope_threshes, ma_slope_thresh_yellow_multipliers, \
                                obv_osc_threshes, obv_osc_thresh_yellow_multipliers,obv_ma_lens, \
                                signal_generators, ma_slope_periods, cfgRenkoBrickMultipliers, atrlens, \
                                cfgRenkoBrickMultiplierLongTargets, cfgRenkoBrickMultiplierLongSLs, \
                                cfgRenkoBrickMultiplierShortTargets, cfgRenkoBrickMultiplierShortSLs):
        ma_slope_thresh, ma_slope_thresh_yellow_multiplier, obv_osc_thresh, obv_osc_thresh_yellow_multiplier, obv_ma_len, sigGen, ma_slope_period,  cfgRenkoBrickMultiplier, atrlen \
            , cfgRenkoBrickMultiplierLongTarget, cfgRenkoBrickMultiplierLongSL, cfgRenkoBrickMultiplierShortTarget, cfgRenkoBrickMultiplierShortSL \
             = params

        signals.updateCFG(ma_slope_thresh, ma_slope_thresh_yellow_multiplier, \
                         obv_osc_thresh, \
                         obv_osc_thresh_yellow_multiplier, obv_ma_len, ma_slope_period, cfgRenkoBrickMultiplier, atrlen \
                             , cfgRenkoBrickMultiplierLongTarget, cfgRenkoBrickMultiplierLongSL, cfgRenkoBrickMultiplierShortTarget, cfgRenkoBrickMultiplierShortSL)
        tearsheetdf = backtest(cfgTicker,'minute',exportCSV=False,
                               applyTickerSpecificConfig=False, signalGenerators=sigGen, src=src)
        if tearsheetdf is None:
            print("No data for this ticker")
            return
        # Add in config variables we are looping through to the tearsheetdf
        tearsheetdf['ma_slope_thresh'] = ma_slope_thresh
        print(f"Doing for start:{firstTradeTime} end: {zgetTo} ma_slope_thresh: {ma_slope_thresh} ma_slope_thresh_yellow_multiplier: {ma_slope_thresh_yellow_multiplier} obv_osc_thresh: {obv_osc_thresh} obv_osc_thresh_yellow_multiplier: {obv_osc_thresh_yellow_multiplier} obv_ma_len: {obv_ma_len}")
        tearsheetdf['ma_slope_thresh_yellow_multiplier'] = ma_slope_thresh_yellow_multiplier
        tearsheetdf['obv_osc_thresh'] = obv_osc_thresh
        tearsheetdf['obv_osc_thresh_yellow_multiplier'] = obv_osc_thresh_yellow_multiplier
        tearsheetdf['obv_ma_len'] = obv_ma_len
        tearsheetdf['ma_slope_period'] = ma_slope_period
        tearsheetdf['cfgRenkoBrickMultiplier'] = cfgRenkoBrickMultiplier
        tearsheetdf['atrlen'] = atrlen
        #tearsheetdf['ticker'] = 'NIFTY23APRFUT'
        tearsheetdf['interval'] = 'minute'
        tearsheetdf['startTime'] = firstTradeTime
        tearsheetdf ['endTime'] = zgetTo
        tearsheetdf['duration_in_days'] = (zgetTo - firstTradeTime).days
        if len(sigGen) == 2:
            tearsheetdf['signalGenerators'] = 'TrendingOnly'
        elif len(sigGen) == 3:
            tearsheetdf['signalGenerators'] = 'BBOnly'
        elif len(sigGen) == 5:
            tearsheetdf['signalGenerators'] = 'TrendPlusBB'
        elif len(sigGen) == 1:
            tearsheetdf['signalGenerators'] = 'JustFollowMA'
        else:
            print("WIERD NUMBER OF SIGNAL GENERATORS")
            exit(0)
        performance = pd.concat([performance, tearsheetdf]) if tearsheetdf['num_trades'][0] > 0 else performance

    
    # write the DataFrame to a SQL table
    # Connect to the MySQL database
    #HACK THIS performanceTOCSV also parses arv, and adds it to the performance df
    performance = performanceToCSV(performance)
    mark_task_complete()
    # create a database connection (performance to CSV adds the 
    # argv variables as well to the performance df)
    if performance is None or performance.empty:
        print("No trades to write to DB")
        return
    engine = create_engine('mysql+pymysql://trading:trading123@trading.ca6bwmzs39pr.ap-south-1.rds.amazonaws.com/trading')
    performance.to_sql('PerfNiftyRenkov2', engine, if_exists='append')
    engine.dispose()

if isMain:
    #backtestCombinator()       
    #backtestCombinator()       
    #plot_options(['ASIANPAINT'],10,'minute')
    #backtest('NIFTY23APRFUT','minute')
    # isMain = False
    t = perfProfiler("Start", time.time())
    #backtest('NIFTY2351118100CE','minute')

    backtest('NIFTYWEEKLYOPTION','minute',src="db")
    t = perfProfiler("End", t)

    #oneThousandRandomTests()

#    backtest('NIFTY23APR17750CE','minute') 
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