import calendar
import os 
import pandas as pd
import numpy as np 
from datetime import date,timedelta, datetime, time
import pytz
import utils
import math
import zipfile
import plotting

ist = pytz.timezone('Asia/Kolkata')

niftyDir ='/Users/nikhilsama/Dropbox/Coding/AlgoTrading/Data/HISTORICAL/NIFTY/second' 
optionDir = '/Users/nikhilsama/Dropbox/Coding/AlgoTrading/Data/HISTORICAL/Options/second'

niftySubDirPrefix = 'GFDLCM_INDICES_TICK_'
niftyOptSubDirPrefix = 'GFDLNFO_NF_OPT_TICK_'
niftyFile = 'NIFTY 50.NSE_IDX.csv'

def getDataFromFile(file):
    try:
        data = pd.read_csv(file)
    except FileNotFoundError:
        print(f"File not found: {file}")
        #exit(0)
        return None
    if (data.empty):
        print(f"No data found  in {file}")
        return None
    return data

def cleanTickDf(df,isOption=False):
    df['date'] = df['Date'] + ' ' + df['Time']

    # Convert the 'DateTime' column to a datetime object and set it as the index
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y %H:%M:%S')
    df.set_index('date', inplace=True)
    df.index = df.index.tz_localize(ist)

    df.rename(columns={"Ticker":"symbol"},inplace=True)
    
    # Remove the '.NFO' suffix from the 'symbol' column
    df['symbol'] = df['symbol'].str.replace('.NFO', '')
    df['symbol'] = df['symbol'].str.replace('.NSE_IDX', '')

    df.rename(columns={"LTP":"Adj Close"},inplace=True)
    # Drop the original Date and Time columns
    df.drop(columns=['Date', 'Time'], inplace=True) if isOption else \
    df.drop(columns=['Date', 'Time', 'BuyPrice', 'BuyQty', 'SellPrice', 'SellQty', "LTQ", "OpenInterest"], inplace=True)

    if isOption:
        niftyDF = getNiftyDF(df.index[0])
        df['nifty'] = niftyDF['Adj Close']
        
        # Extract 'option_type' from the 'symbol' column
        df['option_type'] = df['symbol'].str[-2:]

        # Extract 'strike' from the 'symbol' column
        df['strike'] = df['symbol'].str[12:-2]

        # Convert 'expiry' column to date object
        df['expiry'] = pd.to_datetime(df['symbol'].str[5:12], format='%d%b%y')
        df['expiry'] = pd.to_datetime(df['expiry'], format='%d%b%y')

        df.rename(columns={"LTQ":"Volume"},inplace=True)
    df.insert(0, 'i', range(1, 1 + len(df)))

    df = utils.cleanDF(df)
    return df

def getNiftyDF(t):
    (y,m,d,h,mnt) = (t.year,t.month,t.day,t.hour,t.minute)
    mStr = calendar.month_abbr[m]
    mStr = mStr.upper()
    file = f'{niftyDir}/{y}/{mStr}_{y}/{niftySubDirPrefix}{str(d).zfill(2)}{str(m).zfill(2)}{y}/{niftyFile}'
    
    df = getDataFromFile(file)
    df = cleanTickDf(df)
    df = df[~df.index.duplicated(keep='first')]

    if (h == 9 and mnt < 18):
        print("Can only get prices after 9:18 am, updating time to 18")
        t = t.replace(minute=18)
    return df

def getNiftyPrice(t):
    df = getNiftyDF(t)
    return df.loc[t]['Adj Close']

def getDFFromCSV(t,date):
    (y,m,d,h,mnt) = (date.year,date.month,date.day,date.hour,date.minute)
    mStr = calendar.month_abbr[m]
    mStr = mStr.upper()
    file = f'{optionDir}/{y}/{mStr}_{y}/{niftyOptSubDirPrefix}{str(d).zfill(2)}{str(m).zfill(2)}{y}/{t}.NFO.csv'
    df = getDataFromFile(file)
    df = cleanTickDf(df,isOption=True)
    return df
def getStrikeForDate(dt,offset=0):
    dt = datetime.combine(dt.date(), time(9,18))
    dt = ist.localize(dt)
    open_value = getNiftyPrice(dt)
    # print(f"open_value: {open_value} offset: {offset} strike: {math.floor(open_value/100)*100 - offset}")
    callStrike = math.floor(open_value/100)*100
    putStrike = math.ceil(open_value/100)*100
    return (callStrike - int(offset),putStrike + int(offset))

def getExpiry(date):
    current_day_of_week = date.weekday()

    # Calculate the number of days to add to reach the next Thursday (3)
    days_to_add = (3 - current_day_of_week) % 7
    # Get the date of the Thursday following the given date
    next_thursday = date + timedelta(days=days_to_add)
    expiry = next_thursday
    while not utils.isTradingDay(expiry):
        expiry = expiry - timedelta(days=1)

    if date.date() == expiry.date():
        expiry = expiry + timedelta(days=1)
        while not utils.isTradingDay(expiry):
            expiry = expiry + timedelta(days=1)
        expiry = getExpiry(expiry)
    return expiry

def getWeeklyTicker(date,offset=0):
    (call_strike,put_strike) = getStrikeForDate(date,offset)
    expiry = getExpiry(date)
    call_optionTicker = f'NIFTY{expiry.strftime("%d%b%y").upper()}{call_strike}CE'
    put_optionTicker = f'NIFTY{expiry.strftime("%d%b%y").upper()}{put_strike}PE'
    return (call_optionTicker,put_optionTicker)

def getOptionDF(day,offset=0,type='Call'):
    (t_call,t_put)=getWeeklyTicker(day,offset)
    return getDFFromCSV(t_call,day) if type == 'Call' else getDFFromCSV(t_put,day)

def csvGetOption(t,s,e,targetCallPrice=200,type="Call"):
    thisDay = s
    df = pd.DataFrame()
    offset = 0

    while thisDay<=e:
        targetCallOptClose = 200 if thisDay.date().weekday() != 4 else 150
        targetPutOptClose = 200 if thisDay.date().weekday() != 4 else 150
        targetOptClose = targetCallOptClose if type == 'Call' else targetPutOptClose

        if  utils.isTradingDay(thisDay):
            #for each day 
            day_df = getOptionDF(thisDay,type=type)
            close = day_df.iloc[2]["Adj Close"]
            if close < targetOptClose:
                # print(f"offsetting {thisDay} by 100, Adj Close is {close}")
                while day_df.iloc[2]["Adj Close"] < targetOptClose:
                    offset = offset + 100
                    day_df = getOptionDF(thisDay,offset=offset,type=type) 

                # print(f"New close is {day_df.iloc[2]['Close']}")
            elif close > targetOptClose+100:
                while day_df.iloc[2]["Adj Close"] > targetOptClose+100:
                    offset = offset - 100
                    day_df = getOptionDF(thisDay,offset=offset,type=type)
            df = df.append(day_df)
        thisDay = thisDay + timedelta(days=1)

    return df
    
        
df = csvGetOption('NIFTY',datetime(2023,5,31,9,15), datetime(2023,5,31,9,15))
df.to_csv('optiontick.csv')
plotting.plot_backtest(df)
print(df)