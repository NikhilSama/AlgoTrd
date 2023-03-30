#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 14:26:36 2023

@author: nikhilsama
"""
import sys
cfg = {
    'startTime': 0,
    'startHour': 10,
    'endHour': 14,
    'exitHour': 15,
    'days': 60,
    'superLen': 200,
    'maLen': 20,
    'bandWidth': 2,
    'superBandWidth': 2.5,
    'fastMALen': 7,
    'atrLen': 14,
    'adxLen': 14,
    'adxThresh': 30,
    'adxThreshYellowMultiplier': 0.7,
    'numCandlesForSlopeProjection':2,
    'adxSlopeThresh': 0.06,
    'maSlopeThresh': 1,
    'maSlopeThreshYellowMultiplier': 0.7,
    'maSlopeSlopeThresh': 0.01,
    'obvOscThresh': 0.2,
    'obvOscThreshYellowMultiplier': 0.7,
    'obvOscSlopeThresh': 0.3,
    'includeOptions': False,
    'plot': [
        #'trade_returns'
#        ,'adjCloseGraph'
        ],
    'overrideMultiplier': 1.2,
# google cloud specific stuff 
    'dbhost' : 'localhost',
    'dbname' : 'trading',
    'zerodha_access_token': False,
    'dbhost' : '34.131.115.155',
    'dbuser' : 'trading',
    'dbpass' : 'trading',
    'showTradingViewLive' : True,
    'cacheTickData' : False, 
    'bet_size': 1000
}

  
args = sys.argv[1:]
arg_dict = {}
for arg in args:
    key, value = arg.split(':')
    if key not in cfg:
        print(f'Invalid argument: {key}')
        sys.exit()
    # Convert value to int or float if possible
    try:
        value = int(value)
    except ValueError:
        try:
            value = float(value)
        except ValueError:
            if value == 'False':
                value = False
            elif value == 'True':
                value = True
            pass
             
    cfg[key] = value
               
# Create variables with names as dict keys and values as dict values
for key, value in cfg.items():
    locals()[key] = value
