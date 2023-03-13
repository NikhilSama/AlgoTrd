#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 20:22:14 2023

@author: nikhilsama
"""

# Library to do with 

import pickle
import yfinance as yf
import os

period='12mo'
interval = '1h'

with open("Data/indextickers.pickle", "rb") as f:
    tickers = pickle.load(f)

for ticker in tickers: 
    temp=yf.download(ticker, period=period, interval=interval)
    temp.dropna(how="any", inplace=True)
    try: 
        os.mkdir("Data/"+ticker)
    except OSError as error:
        print(error)
    file_name="Data/"+ticker+"/ohlv-"+period+"-"+interval+".pickle"
    with open(file_name,"wb") as f:
        pickle.dump(temp,f)
