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
from kiteconnect import KiteConnect
from sqlalchemy import create_engine
import logging
import pytz
import sys

# set timezone to IST
ist = pytz.timezone('Asia/Kolkata')

class DBBasic:
    
    con = sqlConnector.connect(host="localhost", user="trading", passwd="trading", database="trading", port="3306", auth_plugin='mysql_native_password')
    conTick = sqlConnector.connect(host="localhost", user="trading", passwd="trading", database="trading", port="3306", auth_plugin='mysql_native_password')
    engine = create_engine("mysql+pymysql://{user}:{pw}@localhost:3306/{db}"
                        .format(user="trading", pw="trading", db="trading"))

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

    def get_option_ticker(self,ticker,price,type,strike=0):
        #get rid of anything after a space in ticker NIFTY 50 => NIFTY
        ticker = ticker.split(' ',1)[0]
        #find the option for this ticker with strike closest to price, and soonest expiry
        q = f"SELECT tradingsymbol,lot_size,tick_size FROM trading.instruments_zerodha where instrument_type = '{type}' and underlying_ticker = '{ticker}' AND expiry > '{date.today()}' ORDER BY ABS( strike - {price} ) ASC, expiry ASC LIMIT 1"
        
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
    def get_instrument_token(self, symbol):
        from sqlalchemy import create_engine
        #engine = create_engine("mysql+pymysql://{user}:{pw}@localhost:3306/{db}"
         #                   .format(user="trading", pw="trading", db="trading"))

        query = f"SELECT instrument_token FROM trading.instruments_zerodha where tradingsymbol = '{symbol}'"
        df = pd.read_sql(query, con=self.engine)

#        query = "SELECT instrument_token FROM trading.instruments_zerodha where tradingsymbol = '{symbol}'"
 #       df = pd.read_sql(query, con=engine)
        if len(df) > 0:
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

