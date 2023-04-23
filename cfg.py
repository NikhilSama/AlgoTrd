#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 14:26:36 2023

@author: nikhilsama
"""
import sys
import datetime
cfgDict = {
    'cfgZGetStartDate': None,
    'cfgZGetEndDate': None,    
    'cfgMaxLotsForTrade' : 36,
    'cfgTicker': 'NIFTYWEEKLYOPTION',
    'cfgHistoricalDaysToGet': 10,
    'cfgFreezeGun': False,
    'cfgUseVolumeDataForOptions': False,
    'cfgStartTimeOfDay': datetime.datetime.strptime("09:30+05:30", "%H:%M%z").time(),
    'cfgEndNewTradesTimeOfDay': datetime.datetime.strptime("14:00+05:30", "%H:%M%z").time(),
    'cfgEndExitTradesOnlyTimeOfDay': datetime.datetime.strptime("15:00+05:30", "%H:%M%z").time(),
    'cfgMinCandlesForMA': 5,
    'cfgStopLoss': 0.05,
    'days': 60,
    'superLen': 200,
    'maLen': 20,
    'cfgMiniBandWidthMult': 0.75, 
    'bandWidth': 2,
    'cfgSuperBandWidthMult': 1.25,
    'fastMALen': 7,
    'cfgFastMASlpThresh': 0.01,
    'atrLen': 7,
    'adxLen': 14,
    'adxThresh': 25,
    'adxThreshYellowMultiplier': 0.6,
    'numCandlesForSlopeProjection':2,
    'maSlopeThresh': 0,
    'maSlopeThreshYellowMultiplier': 0.6,
    'cfgObvMaLen': 20,
    'obvOscThresh': 0.2,
    'obvOscThreshYellowMultiplier': 0.7,
    'cfgMaxLookbackCandles': 200,
    'cfgNumConditionsForTrendFollow': 3,
    'includeOptions': False,
    'plot': [
        
        #'trade_returns'
        # ,
        #    'adjCloseGraph'
        # ,
        # 'options'
        ],
# google cloud specific stuff 
    'dbhost' : 'localhost', 
    'dbname' : 'trading',
    'zerodha_access_token': False,
#    'dbhost' : '34.131.115.155',
    'dbuser' : 'trading',
    'dbpass' : 'trading',
    'showTradingViewLive' : False,
    'cacheTickData' : False, 
    'bet_size': 10000
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
    if key == 'cfgZGetStartDate' or key == 'cfgZGetEndDate':
        value = datetime.datetime.strptime(value, "%Y-%m-%d")
    cfgDict[key] = value
               
# Create variables with names as dict keys and values as dict values
for key, value in cfgDict.items():
    locals()[key] = value