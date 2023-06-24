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
from DatabaseLogin import DBBasic

ist = pytz.timezone('Asia/Kolkata')

niftyDir ='/Users/nikhilsama/Dropbox/Coding/AlgoTrading/Data/HISTORICAL/NIFTY' 
optionDir = '/Users/nikhilsama/Dropbox/Coding/AlgoTrading/Data/HISTORICAL/Options/second'

niftySubDirPrefix = 'GFDLCM_INDICES_TICK_'
niftyOptSubDirPrefix = 'GFDLNFO_NF_OPT_TICK_'
niftyFile = 'NIFTY 50.NSE_IDX.csv'

def getDataFromFile(file):
    try:
        data = pd.read_csv(file)
    except FileNotFoundError:
        print(f"File not found: {file}")
        exit(0)
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
    df.drop(columns=['Date', 'Time'], inplace=True)

    if not isOption:
        df.drop(columns=['OpenInterest'], inplace=True) if 'OpenInterest' in df.columns else None
        df.drop(columns=['Open Interest'], inplace=True) if 'Open Interest' in df.columns else None        
    if isOption:
        niftyDF = getNiftyDF(df.index[0])
        df['nifty'] = niftyDF['Adj Close']
        df['niftyHigh'] = niftyDF['Adj Close']
        df['niftyLow'] = niftyDF['Adj Close']
        
        # Extract 'option_type' from the 'symbol' column
        df['option_type'] = df['symbol'].str[-2:]

        # Extract 'strike' from the 'symbol' column
        df['strike'] = df['symbol'].str[12:-2]

        # Convert 'expiry' column to date object
        df['expiry'] = pd.to_datetime(df['symbol'].str[5:12], format='%d%b%y')
        df['expiry'] = pd.to_datetime(df['expiry'], format='%d%b%y')

        df.rename(columns={"LTQ":"Volume"},inplace=True)

    df = utils.cleanDF(df)
    return df

def getNiftyDF(t):
    (y,m,d,h,mnt) = (t.year,t.month,t.day,t.hour,t.minute)
    mStr = calendar.month_abbr[m]
    mStr = mStr.upper()
    file = f'{niftyDir}/{mStr}_{y}/{niftySubDirPrefix}{str(d).zfill(2)}{str(m).zfill(2)}{y}/{niftyFile}'
    
    df = getDataFromFile(file)
    df = cleanTickDf(df)
    df = df[~df.index.duplicated(keep='first')]

    if (h == 9 and mnt < 18):
        print("Can only get prices after 9:18 am, updating time to 18")
        t = t.replace(minute=18)
    return df

def getNiftyPrice(t):
    df = getNiftyDF(t)
    if t in df.index:
        return df.loc[t]['Adj Close']
    elif t+timedelta(seconds=1) in df.index:
        return df.loc[t+timedelta(seconds=1)]['Adj Close']
    elif t+timedelta(seconds=59) in df.index:
        return df.loc[t+timedelta(seconds=59)]['Adj Close']
    else:
        print(f"Could not find Nifty price for {t}")
        print(df.head(60))
        return None

def getDFFromCSV(t,date):
    (y,m,d,h,mnt) = (date.year,date.month,date.day,date.hour,date.minute)
    mStr = calendar.month_abbr[m]
    mStr = mStr.upper()
    file = f'{optionDir}/{y}/{mStr}_{y}/{niftyOptSubDirPrefix}{str(d).zfill(2)}{str(m).zfill(2)}{y}/{t}.NFO.csv'
    df = getDataFromFile(file)
    print(file)
    df = cleanTickDf(df,isOption=True)
    return df
def getStrikeForDate(dt,offset=0):
    dt = datetime.combine(dt.date(), time(9,18))
    dt = ist.localize(dt)
    open_value = getNiftyPrice(dt)
    print(f"open_value: {open_value} offset: {offset} strike: {math.floor(open_value/100)*100 - offset}")
    callStrike = math.floor(open_value/100)*100
    putStrike = math.ceil(open_value/100)*100
    return (callStrike - int(offset),putStrike + int(offset))


def get_last_thursday_of_month(date):
    # Extract the year and month from the input date
    year = date.year
    month = date.month

    # Get the last day of the month
    last_day = calendar.monthrange(year, month)[1]

    # Iterate backwards from the last day of the month
    for day in range(last_day, 0, -1):
        # Create a datetime object for the current day
        current_day = datetime(year, month, day)

        # Get the weekday of the current day (0 = Monday, 1 = Tuesday, ..., 6 = Sunday)
        weekday = current_day.weekday()
        if weekday == calendar.THURSDAY:
            last_thursday = current_day
            break

    return last_thursday

def get_last_thursday_of_week(date):
    current_day_of_week = date.weekday()

    # Calculate the number of days to add to reach the next Thursday (3)
    days_to_add = (3 - current_day_of_week) % 7
    # Get the date of the Thursday following the given date
    next_thursday = date + timedelta(days=days_to_add)
    return next_thursday

def getExpiry(date):
    expiry = get_last_thursday_of_week(date) if date > datetime(2019, 2, 11) else get_last_thursday_of_month(date)
    while not utils.isTradingDay(expiry):
        expiry = expiry - timedelta(days=1)
    print(f"Date: {date} Expiry: {expiry}")
    if date.date() == expiry.date():
        expiry = expiry + timedelta(days=1)
        while not utils.isTradingDay(expiry):
            expiry = expiry + timedelta(days=1)
        expiry = getExpiry(expiry)
    return expiry

def getWeeklyTicker(date,offset=0):
    # NIFTY18JUL11100PE.NFO.CSV
    (call_strike,put_strike) = getStrikeForDate(date,offset)
    expiry = getExpiry(date)
    if date > datetime(2019, 2, 11):
        call_optionTicker = f'NIFTY{expiry.strftime("%d%b%y").upper()}{call_strike}CE'
        put_optionTicker = f'NIFTY{expiry.strftime("%d%b%y").upper()}{put_strike}PE'
    else: # Only Monthly options before this date
        call_optionTicker = f'NIFTY{expiry.strftime("%y%b").upper()}{call_strike}CE'
        put_optionTicker = f'NIFTY{expiry.strftime("%y%b").upper()}{put_strike}PE'
    return (call_optionTicker,put_optionTicker)

def getOptionDF(day,offset=0,type='Call'):
    (t_call,t_put)=getWeeklyTicker(day,offset)
    return getDFFromCSV(t_call,day) if type == 'Call' else getDFFromCSV(t_put,day)

def resample (df, interval):
    df_resampled = df
    df_resampled['Open'] = df_resampled['High'] = df_resampled['Low'] = df_resampled['Adj Close']
    
    df_resampled = df.resample(interval).agg({
        'symbol': 'last',
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Adj Close': 'last',
        'Volume': 'sum',
        'OpenInterest': 'last',
        'nifty': 'last',
        'niftyHigh': 'max',
        'niftyLow': 'min',
        'option_type': 'last',
        'strike': 'last',
        'expiry': 'last',
        'buyVol': 'sum',
        'sellVol': 'sum',
        'VolDelta': 'sum'
    })
    df_resampled['VolDeltaRatio'] = np.where(df_resampled['sellVol'] != 0, df_resampled['buyVol'] / df_resampled['sellVol'], 10)
    # df_resampled.drop(columns=['BuyPrice', 'BuyQty','SellPrice','SellQty'], inplace=True)
    return df_resampled
def addVolDelta(df):
    # input df is in second tick format
    df['midPrice'] = (df['BuyPrice'] + df['SellPrice'])/2
    df['buyVol'] = np.where(df['Adj Close'] >= df['midPrice'].shift(), df['Volume'], 0)
    df['sellVol'] = np.where(df['Adj Close'] <= df['midPrice'].shift(), df['Volume'], 0)
    df['VolDelta'] = df['buyVol'] - df['sellVol']
    df.drop(columns=['midPrice'], inplace=True)
    df.drop(columns=['BuyPrice', 'BuyQty', 'SellPrice', 'SellQty'], inplace=True)

    return df

# Interval examples 
# 'B': Business day frequency
# 'D': Calendar day frequency
# 'W': Weekly frequency
# 'M': Month end frequency
# 'Q': Quarter end frequency
# 'A': Year end frequency
# 'H': Hourly frequency
# 'T' or 'min': Minute frequency
# 'S' or 'S': Second frequency
# 'L' or 'ms': Millisecond frequency
# 'U' or 'us': Microsecond frequency
# 'N': Nanosecond frequency
    
## Can also do 2S for 2 seconds, or or 2T for 2 minutes
def csvGetOption(t,s,e,targetCallPrice=200,type="Call",interval='1s'):
    thisDay = s
    df = pd.DataFrame()

    while thisDay<=e:
        if thisDay.date() == datetime(2021, 4, 28).date() or \
            thisDay.date()== datetime(2021, 4, 29).date() or \
                thisDay.date()== datetime(2021, 4, 30).date() :
            thisDay = thisDay + timedelta(days=1)
            continue
        offset = 0
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
            print(day_df)
            day_df = addVolDelta(day_df)
            day_df = resample(day_df, interval) if interval != '1s' else day_df
            
            df = df.append(day_df)
        thisDay = thisDay + timedelta(days=1)
    df.insert(0, 'i', range(1, 1 + len(df)))
    return df    

def csvToDb(s,e,interval='T',type='Call',offset=0):
    df = csvGetOption('NIFTY',s,e,interval=interval)
    print(df.columns)
    df.drop(columns=['i'], inplace=True) if 'i' in df.columns else None
    df.rename(columns={"OpenInterest":"Open Interest"},inplace=True) if 'OpenInterest' in df.columns else None

    # df.to_csv('test.csv')
    # exit()
    db = DBBasic()
    print(f"saving to DB niftyITMVD{type}{offset if offset !=0 else ''}")
    db.toDB(f'niftyITMVD{type}{interval}{offset if offset !=0 else ""}',df)

def splitTimeFramesAndSaveToDb():
    # csvToDb(datetime(2018,7,1,9,15), datetime(2019,28,2,9,15),interval='T',type='Call',offset=0)
    # csvToDb(datetime(2019,3,1,9,15), datetime(2019,12,31,9,15),interval='T',type='Call',offset=0)
    # csvToDb(datetime(2020,1,1,9,15), datetime(2020,12,31,9,15),interval='T',type='Call',offset=0)
    # csvToDb(datetime(2021,1,1,9,15), datetime(2021,12,31,9,15),interval='T',type='Call',offset=0)
    # csvToDb(datetime(2023,5,1,9,15), datetime(2023,5,31,9,15),interval='3S',type='Call',offset=0)
    csvToDb(datetime(2023,5,1,9,15), datetime(2023,5,31,9,15),interval='T',type='Call',offset=0)


splitTimeFramesAndSaveToDb()