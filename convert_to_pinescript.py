#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 22:44:21 2023

@author: nikhilsama
"""

#d := t == timestamp(2023, 01, 02, 0, 0, 0) ? 10 : d

from datetime import date,timedelta,datetime
import tickerdata as td
import numpy as np
import pandas as pd
import DownloadHistorical as downloader


def convert(t,days=1,i="minute"):
    e =datetime.now()
    s =e - timedelta(days=days)
    df = downloader.zget(s,e,t,i)
    
    for index, row in df.iterrows():
        
        print(f"d := t == timestamp({index.year}, {index.month}, {index.day}, {index.hour}, {index.minute}, {index.second}) ? {row.close} : d")

convert('NIFTY23MAR17000PE',10)