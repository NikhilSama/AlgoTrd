#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 12:06:54 2023

@author: nikhilsama
"""

#Help File for login to zerodha
import mysql.connector as sqlConnector
from datetime import date,timedelta,timezone,datetime
import pandas as pd
import numpy as np
from kiteconnect import KiteConnect
from sqlalchemy import create_engine
import logging
import pytz
import sys
import os
import utils
import kite_init as ki

# set timezone to IST
ist = pytz.timezone('Asia/Kolkata')
#cfg has all the config parameters make them all globals here
import cfg
globals().update(vars(cfg))

class DBBasic:
    
    con = sqlConnector.connect(host=dbhost, user=dbuser, passwd=dbpass, database=dbname, port="3306", auth_plugin='mysql_native_password')
    conTick = sqlConnector.connect(host=dbhost, user=dbuser, passwd=dbpass, database=dbname, port="3306", auth_plugin='mysql_native_password')
    engine = create_engine("mysql+pymysql://{user}:{pw}@{host}:3306/{db}"
                        .format(user=dbuser, pw=dbpass, host=dbhost, 
                                db=dbname))

    def __init__(self):
        pass
    
    def delAfter(self,frm):
        cursor = self.con.cursor()
        #Delete seems to be hanging the db, so switched to warning instead; delete manually if you see this warning
        dq = f'delete from ohlcv1m where date > \'{frm}\''
        cursor.execute(dq)
        q = f'select * from ohlcv1m where date > \'{frm}\''
        res = pd.read_sql(q, con=self.engine)
        if len(res) > 0:
            print("DATABASE INSCONSISTANT")
            print(q)
            print(dq)
            logging.error("DATABSE INCONSISTANT")

    def next_tick(self,now):
        q = 'select date from ohlcv1m order by date desc limit 1'
        res = pd.read_sql(q, con=self.engine)
        if len(res) > 0:
            frm = res.iloc[0,0]
            self.delAfter(frm)
            frm = frm + timedelta(minutes=1)
            frm = frm.to_pydatetime()
            frm = ist.localize(frm)
        else:
            frm = now - timedelta(days=1)

        return frm
            
    def next_tick_for_t(self,t,now):
        q = f'select date from ohlcv1m where symbol = \'{t}\' ORDER BY date DESC limit 1;'
        res = pd.read_sql(q, con=self.engine)
        if len(res) > 0:
            frm = res.iloc[0,0]
            frm = frm + timedelta(minutes=1)
            frm = frm.to_pydatetime()
            frm = ist.localize(frm)
        else:
            frm = now - timedelta(days=1)

        return frm

    
    def toDB_row_by_row(self, table, df):
        for i in range(len(df)):
            try:
                df.iloc[i:i+1].to_sql(table, con=self.engine, if_exists = 'append', chunksize = 1000)
                #print('Saved to Database')
            except Exception as e:
                logging.info('toDB_row_by_row Unable to save')
                logging.info(e.args[0])
                
    def toDB(self, table, df):
        #if (len(df) < 5): 
            #HACK - entire insert fails if even one row is a duplicate
            #Avoid this by iterating df and adding one row at a time to db
            #slow but wont miss date.  Makes sense for smaller DF like we get
            # in our time loops
            #self.toDB_row_by_row(table,df)
        try:
            df.to_sql(table, con=self.engine, if_exists = 'append', chunksize = 1000)
            #print('Saved to Database')
        except Exception as e:
            logging.info('toDB Unable to save')
            logging.info(e.args[0])

    def frmDB(self, q):
        try:
            df = pd.read_sql(q, con=self.engine)
            df = df.set_index('date')
            df.rename(columns = {'open' : 'Open', 'close' : 'Adj Close', 'high': 'High', 'low': 'Low', 'volume' : 'Volume'}, inplace=True)
            return df
        except Exception as e:
            logging.info('frmDB Unable to read sql')
            logging.info(e.args[0])
            return 0
    def clearInstruments(self):
        cursor = self.con.cursor()
        q = 'delete from instruments_zerodha'
        cursor.execute(q)
        self.con.commit()
    def get_futures_ticker(self,ticker):
        #get rid of anything after a space in ticker NIFTY 50 => NIFTY
        ticker = ticker.split(' ',1)[0]
        #find the future for this ticker with soonest expiry
        q = f"SELECT tradingsymbol,lot_size,tick_size FROM trading.instruments_zerodha where instrument_type = 'FUT' and underlying_ticker = '{ticker}' AND expiry > '{date.today()}' ORDER BY expiry ASC LIMIT 1"
        
        df = pd.read_sql(q, con=self.engine)
        if len(df) > 0:
            return df.iloc[0,0],df.iloc[0,1],df.iloc[0,2]
        else:
            return -1
    #This function takes a ticker, and returns the underlying option ticker
    #closest to price and soonest expiry
    #If given ticker is an option, then it returns the same ticker along with log and tick_size
    #If given ticker is a future, then it returns the underlying option ticker
    #IF it an put option and the request is call, then it returns the call option instead. 
    def get_option_ticker(self,ticker,price,type,kite,strike=0):
        #get rid of anything after a space in ticker NIFTY 50 => NIFTY
        ticker = ticker.split(' ',1)[0]

        #find the option for this ticker with strike closest to price, and soonest expiry
        tickerType = utils.optionTypeFromTicker(ticker)
        if not tickerType:
            if (utils.tickerIsFuture(ticker)):#Its not an option, check if it is a future
                ticker = utils.getUnderlyingTickerForFuture(ticker)
                price = price -150 #TODO HACK - futures are 150 points off on avg; ideally get the ltp of the future and use that
        if not tickerType:
            if strike == 0:
                q = f"SELECT tradingsymbol,lot_size,tick_size FROM trading.instruments_zerodha where instrument_type = '{type}' and underlying_ticker = '{ticker}' AND expiry >= '{date.today()}' ORDER BY ABS( strike - {price} ) ASC, expiry ASC LIMIT 1"
            else: # strike is provided, find the option with this strike only
                q = f"SELECT tradingsymbol,lot_size,tick_size FROM trading.instruments_zerodha where instrument_type = '{type}' and underlying_ticker = '{ticker}' AND expiry >= '{date.today()}' AND strike = {strike} ORDER BY ABS( strike - {price} ) ASC, expiry ASC LIMIT 1"
        else:
            #ticker is alreadu an option; just return the same ticker, with lot_size and tick_size
            q = f"SELECT tradingsymbol,lot_size,tick_size FROM trading.instruments_zerodha where tradingsymbol = '{ticker}' AND expiry >= '{date.today()}' ORDER BY ABS( strike - {price} ) ASC, expiry ASC LIMIT 1"    
        # else:#Ticker is PE and request is CE or vice versa
        #     inverseTicker = utils.convertPEtoCEAndViceVersa(ticker)
        #     q = f"SELECT tradingsymbol,lot_size,tick_size FROM trading.instruments_zerodha where instrument_type = '{type}' and tradingsymbol = '{inverseTicker}' AND expiry > '{date.today()}' ORDER BY ABS( strike - {price} ) ASC, expiry ASC LIMIT 1"    
        df = pd.read_sql(q, con=self.engine)
        if len(df) > 0:
            return df.iloc[0,0],df.iloc[0,1],df.iloc[0,2]
        else:
            return -1
    
    def resample(self,df,interval):
        if (interval == '1min' or interval == '1m'):
            return df
        else:
            return df.resample(interval).agg({'Open': 'first', 
                                 'High': 'max', 
                                 'Low': 'min', 
                                 'Adj Close': 'last',
                                 'Volume': 'sum'}).dropna()


    def get_ticker_data(self,ticker,interval,start,end):
        tbl = 'ohlcv1m'#+interval
        start_time = start.strftime('%Y-%m-%d %H:%M:%S')
        end_time = end.strftime('%Y-%m-%d %H:%M:%S')
        query = f"SELECT * FROM trading.{tbl} where symbol = '{ticker}' and date BETWEEN '{start_time}' and '{end_time}'"
        #print(query)
        df = self.frmDB(query)
        df = self.resample(df,interval)
        return df
        
    def GetTempToken(self):
        cursor = self.con.cursor()
        cursor.execute('Select SQL_NO_CACHE Token, tstamp from TempToken ORDER BY ID DESC LIMIT 1;')
        for row in cursor:
            return row
    def setTokenInCache(self,s,t):
        path = "Data/ticker_cache/instr/"+s+".txt"
        # Open file for writing
        with open(path, 'w') as f:
            # Write string to file
            f.write(str(t))
    def getTokenFromCache(self,s):
        path = "Data/ticker_cache/instr/"+s+".txt"
        if os.path.exists(path):
            with open(path, 'r') as f:
                return np.int64(f.read())
        else:
            return False
        
    def get_instrument_token(self, symbol):
        #from sqlalchemy import create_engine
        #engine = create_engine("mysql+pymysql://{user}:{pw}@localhost:3306/{db}"
         #                   .format(user="trading", pw="trading", db="trading"))
        token = self.getTokenFromCache(symbol);
        if token:
            return token

        query = f"SELECT instrument_token FROM instruments_zerodha where tradingsymbol = '{symbol}'"
        df = pd.read_sql(query, con=self.engine)

#        query = "SELECT instrument_token FROM trading.instruments_zerodha where tradingsymbol = '{symbol}'"
 #       df = pd.read_sql(query, con=engine)
        if len(df) > 0:
            self.setTokenInCache(symbol,df.iloc[0,0])
            return df.iloc[0,0]
        else:
            return -1
        
    def GetAccessToken(self):
        cursor = self.con.cursor()
        query = "Select SQL_NO_CACHE Token from  AccessToken where Date(tstamp) = '{}';".format(date.today())
        cursor.execute(query)
        for row in cursor:
            if row is None:
                return ''
            else:
                return row
            
    def SaveAccessToken(self, Token):
        q1 = "INSERT INTO kimalgo.accesstoken(Token, UniqueID) Values('{}', 1)"  
        q2 = " ON DUPLICATE KEY UPDATE Token = '{}', tstamp=CURRENT_TIMESTAMP();"
        q1 = q1.format(Token)
        q2 = q2.format(Token)
        query = q1 + q2        
        cur = self.con.cursor()
        cur.execute(query)
        self.con.commit()
        
    def InitiateZerodha(self):
        api_key = "ctyl079egk5jwfai"
        apisecret = "skrapb33nfgsrnivz9ms20w0x6odhr3t"
        kite = KiteConnect(api_key=api_key)
        AccessToken = ''
        accessTokenDB = self.GetAccessToken()
        if accessTokenDB is None:
            tempToken = self.GetTempToken()
            if tempToken is None:
                print('tempToken is None:')
                print('Login to API and rerun the program')
                return False, None
            ttoken = tempToken[0]
            print('ttoken:', ttoken)
            try:
                data = kite.generate_session(ttoken, api_secret=apisecret)
                AccessToken = data["access_token"]
                self.SaveAccessToken(AccessToken)
                kite.set_access_token(AccessToken)
                print('Ready to trade')
                return True, kite
            except Exception as e:
                print(e)
                return False, None
        else:
            AccessToken = accessTokenDB[0]
            kite.set_access_token(AccessToken)
            print('Ready to trade')
            return True, kite

