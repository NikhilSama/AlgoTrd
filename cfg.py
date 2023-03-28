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
    'days': 2,
    'atrLen': 14,
    'adxLen': 14,
    'adxThresh': 30,
    'adxThreshYellowMultiplier': 0.7,
    'numCandlesForSlopeProjection':2,
    'adxSlopeThres': 0.06,
    'maThresh': 1,
    'obvOscThresh': 0.25,
    'obvOscThreshYellowMultiplier': 0.7,
    'obvOscSlopeThresh': 0.3,
    'includeOptions': False,
    'plot': True,
    'superLen': 200,
    'maLen': 20,
    'bandWidth': 2,
    'superBandWidth': 2.5,
    'overrideMultiplier': 1.2
}

# Create variables with names as dict keys and values as dict values
for key, value in cfg.items():
    locals()[key] = value
