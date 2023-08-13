#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 14:26:36 2023

@author: nikhilsama
"""
import sys
import datetime
cfgDict = {
    'cfgLazyDataFilePath': '/Users/nikhilsama/Dropbox/Coding/AlgoTrading/Data/lazyData',
    'cfgNiftyVolDeltaThreshold': 140000,
    'cfgFutVolDeltaThreshold': 25000,
    'cfgSlippage': 0.001,
    'cfgTradingCost': 0.001, #0.1% of every trade
    'cfgVolDeltaThresholdMultiplier':0.5,
    'cfgGoodTradeProfitPct': 0.5,
    'cfgNiftyOpen': 18500,
    'cfgIsBackTest': False,
    'cfgZGetStartDate': None,
    'cfgZGetEndDate': None,    
    'cfgMaxLotsForTrade' : 36,
    'cfgTicker': 'NIFTYWEEKLYOPTION',
    'cfgHistoricalDaysToGet': 0,
    'cfgFreezeGun': False,
    'cfgUseVolumeDataForOptions': True,
    'cfgMinPriceForOptions': 10,
    'cfgStartTimeOfDay': datetime.datetime.strptime("9:20+05:30", "%H:%M%z").time(),
    'cfgEndNewTradesTimeOfDay': datetime.datetime.strptime("15:09+05:30", "%H:%M%z").time(),
    'cfgEndExitTradesOnlyTimeOfDay': datetime.datetime.strptime("15:19+05:30", "%H:%M%z").time(),
    'cfgTimeToCheckDayTrendInfo': datetime.datetime.strptime("11:00+05:30", "%H:%M%z").time(),
    'cfgMeanRevStartTime': datetime.datetime.strptime("10:30+05:30", "%H:%M%z").time(),
    'cfgMeanRevEndTime': datetime.datetime.strptime("13:30+05:30", "%H:%M%z").time(),
    'cfgMinCandlesForMA': 5,
    'cfgStopLoss': 0.03,
    'cfgStopLossFromPeak': 0.03,
    'cfgTarget': 0.06,
    'days': 60,
    'superLen': 200,
    'maLen': 20,
    'cfgMiniBandWidthMult': 0.75, 
    'bandWidth': 2,
    'cfgSuperBandWidthMult': 1.25,
    'fastMALen': 5,
    'cfgFastMASlpThresh': 0.01,
    'atrLen': 100,
    'adxLen': 2,
    'adxThresh': 25,
    'cfgAdxThreshExitMultiplier': 0.5,
    'adxThreshYellowMultiplier': 0.6,
    'numCandlesForSlopeProjection':2,
    'maSlopeThresh': 0.02,
    'cfgMASlopePeriods': 3,
    'maSlopeThreshYellowMultiplier': 0.6,
    'cfgObvMaLen': 20,
    'cfgObvLen':3,
    'obvOscThresh': 0.01,
    'obvOscThreshYellowMultiplier': 0.7,
    'cfgObvSlopeThresh': 0.01,
    'cfgMaxLookbackCandles': 400,
    'cfgNumConditionsForTrendFollow': 2,
    'cfgMinStaticCandlesForMeanRev':120,
    'cfgRenkoBrickSize': 10,
    'cfgRenkoBrickMultiplierLongTarget': 2,
    'cfgRenkoBrickMultiplierLongSL': 1,
    'cfgRenkoBrickMultiplierShortTarget': 2,
    'cfgRenkoBrickMultiplierShortSL': 1,
    'cfgTargetPercentageFromResistance': 0.01,
    'cfgSLPercentageFromSupport': 0.02,
    'cfgRenkoNumBricksForTrend': 2,
    'cfgFastSVPWindowSize': 15,
    'cfgSVPSlopeCandles': 2,
    'cfgSVPSlopeProjectionCandles': 1,
    'cfgSVPSlopeThreshold': 1,
    'cfgSVPSessionSlopeThreshold': 0.1,
    'cfgEnoughReturnForTheDay': 0.05,
    'cfgEnoughLossForTheDay': 1,
    'cfgPartialExitPercent': 1,
    'includeOptions': False,
    'plot': [
        
        # 'trade_returns'
        # ,
         'adjCloseGraph'
        #  ,
        #  'plot_returns_on_nifty'
        # ,
        #'option_intrinsic'
        #,
        # 'options'
        ],
# google cloud specific stuff 
    # 'dbhost' : 'localhost', 
    'dbhost' : 'trading.ca6bwmzs39pr.ap-south-1.rds.amazonaws.com',
    'dbname' : 'trading',
    'zerodha_access_token': False,
#    'dbhost' : '34.131.115.155',
    'dbuser' : 'trading',
    'dbpass' : 'trading123',
    'showTradingViewLive' : False,
    'cacheTickData' : False, 
    'bet_size': 200000   
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