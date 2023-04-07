#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 14:26:36 2023

@author: nikhilsama
"""
import sys
cfgDict = {
    'cfgFreezeGun': False,
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
    'adxThresh': 20,
    'adxThreshYellowMultiplier': 0.7,
    'numCandlesForSlopeProjection':2,
    'maSlopeThresh': 1,
    'maSlopeThreshYellowMultiplier': 0.7,
    'obvOscThresh': 0.2,
    'obvOscThreshYellowMultiplier': 0.7,
    'cfgObvMaxMinDiff_MaxLookbackCandles': 200,
    'cfgNumConditionsForTrendFollow': 3,
    'includeOptions': False,
    'plot': [
        
        #'trade_returns'
        #,
        #'adjCloseGraph'
        # ,
        # 'options'
        ],
# google cloud specific stuff 
    'dbhost' : 'localhost',
    'dbname' : 'trading',
    'zerodha_access_token': False,
    'dbhost' : 'localhost',
#    'dbhost' : '34.131.115.155',
    'dbuser' : 'trading',
    'dbpass' : 'trading',
    'showTradingViewLive' : False,
    'cacheTickData' : False, 
    'bet_size': 100000
}

  
args = sys.argv[1:]
arg_dict = {}
for arg in args:
    key, value = arg.split(':')
    if key not in cfgDict:
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
             
    cfgDict[key] = value
               
# Create variables with names as dict keys and values as dict values
for key, value in cfgDict.items():
    locals()[key] = value