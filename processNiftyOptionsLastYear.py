#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 20:18:29 2023

@author: nikhilsama
"""
from datetime import date,timedelta
import datetime
import time
import DownloadHistorical as downloader
import pytz
import pandas as pd
import numpy as np 
import math 
import utils
import os 
from DatabaseLogin import DBBasic

# set timezone to IST
ist = pytz.timezone('Asia/Kolkata')
zgetFrom = datetime.datetime(2022, 5,1, 9, 30, tzinfo=ist)
zgetTo = datetime.datetime(2023, 4,1, 15, 30, tzinfo=ist)
niftyDF = pd.DataFrame()

def zget(t,s,e,i):
    #Get latest minute tick from zerodha
    df = downloader.zget(s,e,t,i)
    df = downloader.zColsToDbCols(df)
    return df


def getStrikeForDate(dt,offset=0):
    if niftyDF.index[0].date() < dt.date():
        last_row_before_dt = niftyDF[niftyDF.index.date == dt.date()]
        last_row_before_dt = last_row_before_dt.iloc[0]
    else:
        last_row_before_dt = niftyDF.iloc[0]
    open_value = last_row_before_dt['Adj Close'] 
    # print(f"open_value: {open_value} offset: {offset} strike: {math.floor(open_value/100)*100 - offset}")  
    return math.floor(open_value/100)*100 - offset

# def getNiftyYear():
#     df = zget('NIFTY 50',zgetFrom,zgetTo,'60minute')
#     return df

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
    strike = getStrikeForDate(date,offset)
    expiry = getExpiry(date)
    optionTicker = f'NIFTY{expiry.strftime("%d%b%y").upper()}{strike}CE'
    return optionTicker
def getDFFromCSV(t,date):
    # Convert the given date to datetime object and extract day, month, and year
    #date = pd.to_datetime(date.date(), format='%d/%m/%Y')
    day = date.strftime('%d')
    month = date.strftime('%m')
    year = date.strftime('%Y')
    month_year = date.strftime('%b_%Y').upper()

    # Create the file path
    if month == '09' and year == '2022':
        prefix = 'GFDLNFO_OPTIONS_'
    else:
        prefix = 'GFDLNFO_NIFTY_OPT_'
    file_name = f"{prefix}{day}{month}{year}.csv"
    file_path = os.path.join("Data/NIFTYOPTIONSDATA",year, month_year, file_name)

    # Read the csv file into a dataframe
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        #exit(0)
        return None
    if (data.empty):
        print(f"No data found  in {file_path}")
        exit(0)        
        return None

    # Filter the dataframe for the given ticker and date
    filtered_data = data[(data['Ticker'] == t+'.NFO') & (data['Date'] == date.strftime('%d/%m/%Y'))]
    if (filtered_data.empty):
        print(f"No data found for {t} on {date.strftime('%d/%m/%Y')} in {file_path}")
        exit(0)
        return None
    return filtered_data

def getOptionDF(day,offset=0):
    t=getWeeklyTicker(day,offset)
    return getDFFromCSV(t,day)

def cleanDF(df):
    df['date'] = df['Date'] + ' ' + df['Time']

    # Convert the 'DateTime' column to a datetime object and set it as the index
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y %H:%M:%S')
    df.set_index('date', inplace=True)
    df.index = df.index.tz_localize(ist)
    # Set the seconds values to 0 in the DateTimeIndex
    df.index = df.index.floor('T')
    df.rename(columns={"Ticker":"symbol"},inplace=True)
    df.rename(columns={"Close":"Adj Close"},inplace=True)
    # Drop the original Date and Time columns
    df.drop(columns=['Date', 'Time'], inplace=True)
    return df

def constructDF(offset=0):
    global niftyDF
    niftyDF = zget('NIFTY 50',zgetFrom,zgetTo,'60minute') # Get NIFTY Data
    thisDay = zgetFrom
    df = pd.DataFrame()
    while thisDay<zgetTo:
        if  utils.isTradingDay(thisDay):
            #for each day 
            day_df = getOptionDF(thisDay,offset)
            df = df.append(day_df)
        thisDay = thisDay + timedelta(days=1)
    df = cleanDF(df)
    db = DBBasic()
    db.toDB(f'niftyITMCall{offset if offset !=0 else None}',df)
    df.to_csv(f'Data/NIFTYOPTIONSDATA/contNiftyWeeklyOptionDF{offset if offset !=0 else None}.csv')
    
constructDF(100)