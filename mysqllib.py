#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 19:32:58 2023

@author: nikhilsama
"""

import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
  user="trading",
  password="trading", auth_plugin='mysql_native_password')

print(mydb)