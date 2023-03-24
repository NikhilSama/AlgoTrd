#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 09:08:04 2023

@author: nikhilsama
"""
import kite_init as k
import logging

logging.info("EXITING ALL POSITIONS CALLED MANUALLY")

kite = k.initKite()
x = k.get_positions(kite)
for y in x.values():
      k.exit_given_positions(kite, y['positions'])
