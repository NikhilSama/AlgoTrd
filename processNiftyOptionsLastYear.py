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
    callStrike = math.floor(open_value/100)*100
    putStrike = math.ceil(open_value/100)*100
    return (callStrike - offset,putStrike + offset)

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
    (call_strike,put_strike) = getStrikeForDate(date,offset)
    expiry = getExpiry(date)
    call_optionTicker = f'NIFTY{expiry.strftime("%d%b%y").upper()}{call_strike}CE'
    put_optionTicker = f'NIFTY{expiry.strftime("%d%b%y").upper()}{put_strike}PE'
    return (call_optionTicker,put_optionTicker)
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
    # Filter the dataframe for the given ticker and date
    filtered_data = data[(data['Ticker'] == t+'.NFO') & (data['Date'] == date.strftime('%d/%m/%Y'))]
    if (filtered_data.empty):
        print(f"No data found for {t} on {date.strftime('%d/%m/%Y')} in {file_path}")
        exit(0)
        return None
    return filtered_data

def getOptionDF(day,offset=0,type='Call'):
    (t_call,t_put)=getWeeklyTicker(day,offset)
    return getDFFromCSV(t_call,day) if type == 'Call' else getDFFromCSV(t_put,day)

def cleanDF(df):
    df['date'] = df['Date'] + ' ' + df['Time']

    # Convert the 'DateTime' column to a datetime object and set it as the index
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y %H:%M:%S')
    df.set_index('date', inplace=True)
    df.index = df.index.tz_localize(ist)
    # Set the seconds values to 0 in the DateTimeIndex
    df.index = df.index.floor('T')
    df.rename(columns={"Ticker":"symbol"},inplace=True)
    
    # Remove the '.NFO' suffix from the 'symbol' column
    df['symbol'] = df['symbol'].str.replace('.NFO', '')

    # Extract 'option_type' from the 'symbol' column
    df['option_type'] = df['symbol'].str[-2:]

    # Extract 'strike' from the 'symbol' column
    df['strike'] = df['symbol'].str[12:-2]

    # Convert 'expiry' column to date object
    print(df['symbol'])
    df['expiry'] = pd.to_datetime(df['symbol'].str[5:12], format='%d%b%y')
    df['expiry'] = pd.to_datetime(df['expiry'], format='%d%b%y')

    df.rename(columns={"Close":"Adj Close"},inplace=True)
    # Drop the original Date and Time columns
    df.drop(columns=['Date', 'Time'], inplace=True)
    return df

def constructDF(offset=0,type='Call'):
    global niftyDF
    niftyDF = zget('NIFTY 50',zgetFrom,zgetTo,'60minute') # Get NIFTY Data
    thisDay = zgetFrom
    df = pd.DataFrame()
    while thisDay<zgetTo:
        targetCallOptClose = 200 if thisDay.date().weekday() != 4 else 150
        targetPutOptClose = 200 if thisDay.date().weekday() != 4 else 150
        targetOptClose = targetCallOptClose if type == 'Call' else targetPutOptClose
        targetOptClose = 100
        offset = 0
        if  utils.isTradingDay(thisDay):
            #for each day 
            day_df = getOptionDF(thisDay,offset,type)
            close = day_df.iloc[2]["Close"]
            if close < targetOptClose:
                # print(f"offsetting {thisDay} by 100, Adj Close is {close}")
                while day_df.iloc[2]["Close"] < targetOptClose:
                    offset = offset + 100
                    day_df = getOptionDF(thisDay,offset=offset,type=type) 

                # print(f"New close is {day_df.iloc[2]['Close']}")
            elif close > targetOptClose+100:
                while day_df.iloc[2]["Close"] > targetOptClose+100:
                    offset = offset - 100
                    day_df = getOptionDF(thisDay,offset=offset,type=type)
            print(f"Close is {day_df.iloc[2]['Close']}")
            df = df.append(day_df)
        thisDay = thisDay + timedelta(days=1)
    df = cleanDF(df)
    db = DBBasic()
    print(f"saving to DB niftyITMN{type}{offset if offset !=0 else ''}")
    db.toDB(f'niftyITMN{type}100{offset if offset !=0 else ""}',df)
    df.to_csv(f'Data/NIFTYOPTIONSDATA/contNiftyWeeklyOptionDF{offset if offset !=0 else ""}.csv')

def splitDatesIntoChunks(s,e):
    # Calculate the total number of days between the two dates
    total_days = (e - s).days

    # Calculate the number of 60-day chunks
    num_chunks = total_days // 60

    # Initialize a list to store the chunks
    date_chunks = []

    # Split the date range into 60-day chunks
    for i in range(num_chunks):
        start_date = s + datetime.timedelta(days=i * 60)
        end_date = start_date + datetime.timedelta(days=60)
        date_chunks.append((start_date, end_date))

    # Handle the remaining days (if any)
    remaining_days = total_days % 60
    if remaining_days > 0:
        start_date = s + datetime.timedelta(days=num_chunks * 60)
        end_date = start_date + datetime.timedelta(days=remaining_days)
        date_chunks.append((start_date, end_date))
    return date_chunks

def optionFileToDB(file_path):
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
    data = cleanDF(data)
    db = DBBasic()
    db.toDB(f'nifty_options',data)
    

def niftyDFToDB():
    df = pd.DataFrame()
    dateChunks = splitDatesIntoChunks(zgetFrom,zgetTo)
    for (s,e) in dateChunks:
        df = df.append(zget('NIFTY 50', s,e,'minute'))
    df['Volume'] = 1
    db = DBBasic()
    db.toDB(f'nifty',df)

# Recursive function to find all files in a directory
def find_files(directory):
    file_list = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                file_list.append(file_path)
    return file_list

def migrateCSVToDb():
    files = find_files('Data/NIFTYOPTIONSDATA/2022')
    files.extend(find_files('Data/NIFTYOPTIONSDATA/2023'))
    for file in files:
        print(file)
        optionFileToDB(file)
# print(find_files('Data/NIFTYOPTIONSDATA/2022'))
# # niftyDFToDB()
# exit()
# niftyDF = zget('NIFTY 50',zgetFrom,zgetTo,'minute') # Get NIFTY Data

# migrateCSVToDb()

constructDF(type='Call')