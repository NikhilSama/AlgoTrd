#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 20:22:14 2023

@author: nikhilsama
"""
# Library to do with getting tickers, and ticker data from yfinance
# Also includes the ability to store local file copies of any data gotten from
# yf so we are not making repeat calls if a copy exists. 

import bs4 as bs
import pickle
import requests
import pandas as pd
import os
import yfinance as yf
import datetime as dt
import warnings
from DatabaseLogin import DBBasic
import logging
import math 
import DownloadHistorical as downloader
import datetime


db = DBBasic() 

def get_ticker_data(ticker, start, end, interval='1m', incl_options=False,strike=0):
    df = db.get_ticker_data(ticker,interval,start,end)
    
    if (incl_options):
        put_option_ticker =db.get_option_ticker(ticker,df['Adj Close'][-1], 'PE',strike)
        p_df = db.get_ticker_data(put_option_ticker,interval,start,end)
        df['Open-P'] = p_df['Open']
        df['High-P'] = p_df['High']
        df['Low-P'] = p_df['Low']
        df['Adj Close-P'] = p_df['Adj Close']
        df['Volume-P'] = p_df['Volume']
        call_option_ticker = db.get_option_ticker(ticker,df['Adj Close'][-1], 'CE',strike)
        c_df = db.get_ticker_data(call_option_ticker,interval,start,end)
        df['Open-C'] = c_df['Open']
        df['High-C'] = c_df['High']
        df['Low-C'] = c_df['Low']
        df['Adj Close-C'] = c_df['Adj Close']
        df['Volume-C'] = c_df['Volume']

    return df

def yFin_get_ticker_data(ticker, period='1y', interval='1d'):
    #create directory if none exists
    path = "Data/"+ticker
    if not os.path.exists(path):
        try: 
            os.mkdir("Data/"+ticker)
        except OSError as error:
            print(error)

    #check if file exists
    file_name=path+"/ohlv-"+period+"-"+interval+".pickle"
    if os.path.isfile(file_name):
        #return file contents
        with open(file_name, "rb") as f:
                temp = pickle.load(f)
    else:
        #get from yf
        temp=yf.download(ticker, period=period, interval=interval)
        temp.dropna(how="any", inplace=True)
        with open(file_name,"wb") as f:
            pickle.dump(temp,f)
    return temp



def get_sp500_tickers():
    file_name = "Data/sp500tickers.pickle"
    tickers = []

    if os.path.isfile(file_name):
        with open(file_name, "rb") as f:
                tickers = pickle.load(f)
    else:
        resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        soup = bs.BeautifulSoup(resp.text, 'lxml')
        table = soup.find('table', {'class': 'wikitable sortable'})
        for row in table.findAll('tr')[1:]:
            ticker = row.findAll('td')[0].text
            tickers.append(ticker)
    
        with open(file_name,"wb") as f:
            pickle.dump(tickers,f)

    return tickers

def getTickerPrice(t,zgetTO):
    zgetFROM = zgetTO - datetime.timedelta(minutes=1)
    df = downloader.zget(zgetFROM,zgetTO,'NIFTY 50','minute',includeOptions=False)
    return df.iloc[0]['Adj Close'] 

def getStrike(t,offset=0):
    niftyOpen = getTickerPrice(t,datetime.datetime.combine(datetime.date.today(), datetime.time(9,16))) #if cfgNiftyOpen == 0 else cfgNiftyOpen 
    strikeFloor = (math.floor(niftyOpen/100)*100) - offset  
    strikeCiel = (math.ceil(niftyOpen/100)*100) + offset

    #print(f"Strike Floor: {strikeFloor}, Strike Ciel: {strikeCiel}")
    return (strikeFloor,strikeCiel)
    
def getActiveOptionTickers(t,offset=0):
    (strikeFloor,strikeCiel) = getStrike(t,offset)
    (itmCall,lot,tick) = db.get_option_ticker(t,0,'CE',None,strike=strikeFloor)
    (otmCall,lot,tick) = db.get_option_ticker(t,0,'CE',None,strike=strikeCiel)
    (otmPut,lot,tick) = db.get_option_ticker(t,0,'PE',None,strike=strikeFloor)
    (itmPut,lot,tick) = db.get_option_ticker(t,0,'PE',None,strike=strikeCiel)
#    print(f"ITM Call: {itmCall}, OTM Call: {otmCall}, OTM Put: {otmPut}, ITM Put: {itmPut}")
    return(itmCall,otmCall,otmPut,itmPut)

def get_fo_active_nifty_tickers(offset=0):
    (itmCall,otmCall,otmPut,itmPut) = getActiveOptionTickers('NIFTY 50',offset)
    return [itmCall]
    return ['HDFCBANK', 'ICICIBANK', 'RELIANCE', 'KOTAKBANK']
    return ['NIFTY 50','NIFTY23APRFUT','RELIANCE','INFY','BAJFINANCE','SBIN', 'TCS', 'KOTAKBANK', 'ICICIBANK', 'HDFCBANK', 'MARUTI' ]

def get_nifty_tickers():
    
#    return ['HDFC','HDFCBANK','HDFCLIFE','TATASTEEL','POWERGRID','ONGC','COALINDIA',
 #          'NIFTY 50']
#'ITC','BPCL','HINDALCO','WIPRO','JSWSTEEL','NTPC','TATAMOTORS',
    # return['NTPC','SBILIFE']
    #return['UPL']
    file_name = "Data/niftytickers.pickle"
    tickers = []
    

    if os.path.isfile(file_name):
        with open(file_name, "rb") as f:
                tickers = pickle.load(f)
    else:
        URL = 'https://www1.nseindia.com/content/indices/ind_nifty50list.csv'
        df = pd.read_csv(URL)
    
        tickers = df["Symbol"].tolist()
        with open(file_name,"wb") as f:
            pickle.dump(tickers,f)
    
    return tickers

def get_index_tickers():
    index_tickers = ["SPY", "qqq","^NSEI", "^NSEBANK"]
    with open("Data/indextickers.pickle","wb") as f:
        pickle.dump(index_tickers,f)
    return index_tickers

def get_all_ticker_data(period='10y', interval='1d'):
    tickers = get_sp500_tickers()
    tickers = tickers + get_nifty_tickers()
    tickers = tickers + get_index_tickers()
    
    for ticker in tickers: 
        get_ticker_data(ticker, period, interval)
        
    return 

