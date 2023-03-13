#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 21:00:32 2023

@author: nikhilsama
"""


import pickle

with open("Data/^NSEI/ohlv-12mo-1d.pickle", "rb") as f:
    ohlv = pickle.load(f)