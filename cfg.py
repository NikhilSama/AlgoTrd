#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 14:26:36 2023

@author: nikhilsama
"""
cfg = {
    'startTime': 0,
    'startHour': 10,
    'endHour': 14,
    'exitHour': 15,
    'days': 4,
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
    'plot': True,
    'overrideMultiplier': 1.2,
# google cloud specific stuff 
    'dbhost' : 'localhost',
    'dbname' : 'trading',
#   'zerodha_access_token': 'Evr9jmY6dG4T1n7fQGtwRN4lzNAERgJy',
#    'dbhost' : '34.131.115.155',
    'dbuser' : 'trading',
    'dbpass' : 'trading',
    'showTradingViewLive' : False

}

# Create variables with names as dict keys and values as dict values
for key, value in cfg.items():
    locals()[key] = value
