#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 11:28:46 2023

@author: nikhilsama
"""

from kiteconnect import KiteConnect
from kiteconnect import KiteTicker
import datetime
import os
import requests
import time
import logging
import pytz
import re
import pandas as pd
import numpy as np
import utils
import math 
# import time_loop as tl

#cfg has all the config parameters make them all globals here
import cfg
globals().update(vars(cfg))

# set timezone to IST
ist = pytz.timezone('Asia/Kolkata')

api_key = "ctyl079egk5jwfai"
apisecret = "skrapb33nfgsrnivz9ms20w0x6odhr3t"
zacceccess_file = "Data/zerodha_kite_accesstoken.txt"
tradelog = f"Data/trades/{datetime.datetime.now().strftime('%d-%m-%y')}.trades"
tradelogcsv = f"Data/trades/{datetime.datetime.now().strftime('%d-%m-%y')}-trades.csv"


def getNewAccessToken(kite): 
    
    r = requests.get(kite.login_url())
    print(r.url)
    
    ttoken = input('ENTER ACCESS TOKEN')
    AccessToken = kite.generate_session(ttoken, api_secret=apisecret)["access_token"]
    print(AccessToken)
    
    with open(zacceccess_file, "w") as f:
        f.write(AccessToken)
    return AccessToken


def getAccessToken(kite):
    if 'zerodha_access_token' in cfgDict:
        if (zerodha_access_token):
            return zerodha_access_token
    
    if os.path.isfile(zacceccess_file):
        #logging.debug('Access Token File exists')
        # file modification
        hours_since_creation = (time.time() - os.path.getmtime(zacceccess_file))/(60*60)
        if (hours_since_creation > 8):
            #Stale File
            logging.debug("AccessToken File is Stale.  Regenerating")
            logging.debug(hours_since_creation)
            AccessToken = getNewAccessToken(kite)
        else:
            #logging.debug("AccessToken File is Valid.")
            with open(zacceccess_file, "r") as f:
                AccessToken = f.read()
    else:
        print("AccessToken File does not exist")

        AccessToken = getNewAccessToken(kite)
    return AccessToken

def initKite():
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(getAccessToken(kite))
    return kite

def initKws(access_token):
    kws = KiteTicker(api_key, access_token)
    return kws 

def initKiteTicker():
    kite = initKite()
    kws = initKws(getAccessToken(kite))
    return kite,kws

def logtrade(s='',ticker=0,position=0,q=0,p=0,ltp=0,lot_size=0,tick_size=0,e='NSE'):
    time = datetime.datetime.now(ist).strftime("%I:%M:%S %p")
    p = round(p,2)
        
    if s == '':
        s = f'{time} -> {position} {q} {e}:{ticker} @ {p} (ltp:{ltp}, lot:{lot_size}, tick:{tick_size})'
    else:
        s = f'{time} -> {s}' 
    with open(tradelog, "a") as f:
        f.write(s)
        f.write('\n')

    if(ticker):
        trade = pd.DataFrame(np.array([time,ticker,position,q,p,e]).reshape(1,6), 
            columns=['time', 'ticker', 'position', 'quantity', 'price','exchange'])
        trade.to_csv(tradelogcsv, mode='a', header=not os.path.exists(tradelogcsv))
    
def getQ (lot_size,ltp,betsize, qToExit=0):
    if (lot_size > 1):
        q = max(round((betsize/ltp)/lot_size),1)*lot_size
        return q+qToExit if qToExit else q
        return 10*(lot_size+qToExit) if qToExit else (10*lot_size)
    else:
        return max(1,round(betsize/ltp))+qToExit if qToExit else max(1,round(betsize/ltp))

def getP (ltp,tick_size,delta):
    p = ltp * delta #Almost market order go market or at max 10% above market
    p = round(p/tick_size)*tick_size
    logging.debug(f"getP returning p: {p} for ltp {ltp} and tick: {tick_size} and delta:{delta}")
    return p

def startOfTick():
    logtrade('WAKE***********************************')

def endOfTick():
    logtrade('SLEEP***********************************')

def is_not_tradable(t):
    if (t == 'NIFTY 50'):
        return True
    else:
        return False 
def getExchange(kite,exchange):
    if (exchange == 'NSE'):
        exch = kite.EXCHANGE_NSE
    elif (exchange == 'BSE'):
        exch = kite.EXCHANGE_BSE
    elif (exchange == 'NFO'):
        exch = kite.EXCHANGE_NFO
    else:
        logging.error(f"Unknown exchange {exchange}")
        print(f"Unknown exchange {exchange}")
        exch = -1
    return exch

def getDelta(exchange,tx_type):
    if exchange == 'NFO':
        if tx_type == 'BUY':
            return 1.01
        elif tx_type == 'SELL':
            return 0.99
        else:
            logging.error(f"Unknown tx_type {tx_type}")
  
    elif exchange == 'BSE' or exchange == 'NSE':
        if tx_type == 'BUY':
            return 1.001
        elif tx_type == 'SELL':
            return 0.999
        else:
            logging.error(f"Unknown tx_type {tx_type}")
    else:
        logging.error(f"Unknown exchange {exchange}")

def getTxType(kite,tx_type):
    if tx_type == 'BUY':
        kite_tx_type = kite.TRANSACTION_TYPE_BUY
    elif tx_type == 'SELL':
        kite_tx_type = kite.TRANSACTION_TYPE_SELL
    else:
        logging.error(f"Unknown tx_type {tx_type}")
    return kite_tx_type
def get_ltp(kite,t,exchange):
    try: 
        ltp = kite.ltp([f"{exchange}:{t}"])
    except:
        logging.error(f"Error getting ltp for {exchange}:{t}. Skipping")
        return 0
    if f"{exchange}:{t}" in ltp.keys():
        ltp = ltp[f"{exchange}:{t}"]['last_price']
    else:
        logging.warning(f"No ltp found for {exchange}:{t}. {ltp} Skipping")
        return 0

    return ltp
def getOrderVariety(kite,q,lot_size) : 
    if q > (lot_size*cfgMaxLotsForTrade):
        v = kite.VARIETY_ICEBERG
        iceberg_legs = math.ceil(q/(lot_size*cfgMaxLotsForTrade))
        iceberg_q = q/iceberg_legs
        iceberg_q = (lot_size*max(round(iceberg_q/lot_size),1)) 
        q = iceberg_q * iceberg_legs 
    else:
        v = kite.VARIETY_REGULAR
        iceberg_legs = 0
        iceberg_q = 0
    return (v,iceberg_legs,iceberg_q, q)

def convertSLTriggerToPrice(sltrigger,slTxType,tick_size):
    slTriggerMult = 1.01 if slTxType == 'BUY' else 0.99
    return round(sltrigger*slTriggerMult/tick_size)*tick_size
def convertSLTriggerToTrigger(sltrigger,tick_size):
    return round(sltrigger/tick_size)*tick_size

def exec_sl(kite,t,exchange,slTxType,sltrigger, 
            lot_size=1,tick_size=0.05, slqt = 0, slvariety = None,sliceberg_legs=None,
            sliceberg_quantity=None, ltp=0, betsize=bet_size):
    slqt = getQ(lot_size,ltp,betsize, 0) if slqt == 0 else slqt

    if slvariety == None:
        (slvariety,sliceberg_legs,sliceberg_quantity,slqt) = \
            getOrderVariety(kite,slqt,lot_size)
    slprice = convertSLTriggerToPrice(sltrigger,slTxType,tick_size) 
    sltrigger = convertSLTriggerToTrigger(sltrigger,tick_size)
    exch = getExchange(kite,exchange)

    try:
        # logging.info(f"Placing SL order for {t} {slTxType}  trigger:{sltrigger} bet:{betsize} qt:{slqt} legs:{sliceberg_legs} iceberg_q:{sliceberg_quantity}")
        sl_order_id = kite.place_order(variety=slvariety,tradingsymbol=t,
                                    exchange=exch,
                                    transaction_type=getTxType(kite,slTxType),
                                    quantity=slqt,
                                    order_type=kite.ORDER_TYPE_SL,
                                    product=kite.PRODUCT_MIS,
                                    iceberg_legs=sliceberg_legs, iceberg_quantity=sliceberg_quantity,
                                    validity=kite.VALIDITY_TTL, validity_ttl = 1,
                                    trigger_price=sltrigger,
                                    price = slprice, tag="StopLoss")
    except Exception as e:
        print(f'{exchange} {slTxType} Failed for {t}')
        print(e.args[0])
        print(f"SL exec {t} {exchange} {slTxType} Lot:{lot_size} tick:{tick_size} q:{slqt} ltp:{ltp}  betsize:{betsize} {slprice} ")
        return -1

    return sl_order_id
def exec(kite,t,exchange,tx_type,lot_size=1,tick_size=0.05
         ,q=0,ltp=0,sl=0,qToExit=0,betsize=bet_size,p=0,tag=None, 
         order_type=None,sltrigger=0,slTxType=None,slqt=None):
    order_type = kite.ORDER_TYPE_LIMIT if order_type is None else order_type
    #print(f"exec {t} {exchange} {tx_type} Lot:{lot_size} tick:{tick_size} q:{q} ltp:{ltp} {sl} toExit:{qToExit} betsize:{betsize} p:{p} tag:{tag}")
    if is_not_tradable(t):
        logging.info(f"{t} is not a tradable instrument.  {exchange} {tx_type} not executed")
        return
    if (ltp == 0):
        #Get ltp if not provided
        ltp = kite.ltp([f"{exchange}:{t}"])
        if f"{exchange}:{t}" in ltp.keys():
            ltp = ltp[f"{exchange}:{t}"]['last_price']
        else:
            logging.warning(f"No ltp found for {exchange}:{t}. {ltp} Skipping")
            return
    # if (lot_size==1) and (exchange=="NFO") and utils.isOption(t):
    #     #Option order and we were not provided lot size info 
    #     #get the lot size from db
    #     db = DBBasic() 
    #     optionTicker,lot_size,tick_size = db.get_option_ticker(t, None, None,None) #if t is an option other arguments are not looked at

    qToExit = abs(qToExit) #Make sure qToExit is positive, for short positions it may be negative
    q = getQ(lot_size,ltp,betsize, qToExit) if q == 0 else q

    (variety,iceberg_legs,iceberg_quantity,q) = getOrderVariety(kite,q,lot_size)
    
    delta = getDelta(exchange,tx_type)
    
    p = getP(ltp,tick_size,delta) if p == 0 else getP(p,tick_size,1)
    #print(f"exec {t} {exchange} {tx_type} legs:{iceberg_legs} qt: {iceberg_quantity} Lot:{lot_size} tick:{tick_size} q:{q} ltp:{ltp} {sl} toExit:{qToExit} betsize:{betsize} {p} {tag}")
    exch = getExchange(kite,exchange)

    kite_tx_type = getTxType(kite,tx_type)

    logtrade('',t,tx_type,q,p,ltp,lot_size,tick_size,exchange)
        
    # logtrade(f'{tx_type} {t} Q:{q} P:{p} LTP:{ltp} Tick:{tick_size}',
    #          t,tx_type,q,p)

    #1 Min TTL LIMIT ORDER priced really as a market order (10% above market)
    try:
        order_id = kite.place_order(variety=variety,tradingsymbol=t,
                     exchange=exch,
                     transaction_type=kite_tx_type,
                     quantity=q,
                     order_type=kite.ORDER_TYPE_LIMIT,
                     product=kite.PRODUCT_MIS,
                     iceberg_legs=iceberg_legs, iceberg_quantity=iceberg_quantity,
                     validity=kite.VALIDITY_TTL, price = p, validity_ttl = 1,
                     tag=tag)
        #SL to accompany 
        ### UNTESTED CODE -- need to get teh right tx type etc in there 
        if (sl):
            if slTxType is None:
                slTxType = 'BUY' if tx_type == 'SELL' else 'SELL'
            if slqt is None:
                slqt = q
                slvariety = variety
                sliceberg_legs = iceberg_legs
                sliceberg_quantity = iceberg_quantity
            else:
                slvariety,sliceberg_legs,sliceberg_quantity = (None,None,None)
            sl_order_id = exec_sl(kite,t,exchange,slTxType,sltrigger,lot_size=lot_size,
                                  tick_size = tick_size, slqt = slqt,slvariety = slvariety,
                                  sliceberg_legs=sliceberg_legs,sliceberg_quantity=sliceberg_quantity)
        else:
            sl_order_id = 0
    except Exception as e:
        print(f'{exchange} {tx_type} Failed for {t}')
        print(e.args[0])
        print(f"exec {t} {exchange} {tx_type} Lot:{lot_size} tick:{tick_size} q:{q} ltp:{ltp} {sl} toExit:{qToExit} betsize:{betsize} {p} {tag}")
        return -1
    return order_id,sl_order_id

def nse_buy (kite,t,lot_size=1,tick_size=0.05,q=0,ltp=0,sl=0,exchange='NSE',qToExit=0,betsize=bet_size,tag=None):
    return exec(kite,t,exchange,'BUY',lot_size,tick_size,q,ltp,sl,qToExit,betsize,tag=tag)

def nse_sell (kite,t,lot_size=1,tick_size=0.05,q=0,ltp=0,sl=0,exchange='NSE',qToExit=0,betsize=bet_size,tag=None):
    return exec(kite,t,exchange,'SELL',lot_size,tick_size,q,ltp,sl,qToExit,betsize,tag=tag)

def nfo_buy (kite,t,lot_size=1,tick_size=0.5, q=0,ltp=0,sl=0,qToExit=0,betsize=bet_size, tag=None):
    return exec(kite,t,'NFO', 'BUY', lot_size,tick_size,q,ltp,sl,qToExit,betsize,tag=tag)
    
def nfo_sell (kite, t, lot_size=1, tick_size=0.5, q=0,ltp=0,sl=0, qToExit=0,betsize=bet_size, tag=None):
    return exec(kite,t,'NFO', 'SELL', lot_size,tick_size,q,ltp,sl,qToExit,betsize,tag=tag)

def gettFromOption (string):
    return utils.optionUnderlyingFromTicker(string)
    if 'NIFTY' in string:
        return 'NIFTY23APRFUT' #hack, dont return the index, return the fugure
    #index is not tradable and does not have volume information, this better
    
    # Define regex pattern to match all leading alphabets
    pattern = r'^[a-zA-Z]+'
    
    # Search for the pattern in the string
    match = re.search(pattern, string)
    
    if match:
        # Extract the matched string
        return match.group(0)
    else:
        # Return an empty string if no match found
        return ''

def isPutOption (t,exch):
    if(exch != 'NFO'):
        return False
    if (utils.optionTypeFromTicker(t) == 'PE'):
        return True

    return False

def isOption(exch,t):
    type = False
    if (exch == 'NFO'):
        type = utils.optionTypeFromTicker(t)
    return True if type else False    

def isFutureOrOption(exch):
    if (exch == 'NFO'):
        return True
    else:
        return False    
    
def long_or_short(position):
    if position['quantity'] == 0:
        return 0
    
    if isPutOption(position['tradingsymbol'],position['exchange']):
        if (position['quantity'] > 0):
            return -1
        else:
            return 1        
    else: #Equity, or Call Option
        if (position['quantity'] > 0):
            return 1
        else:
            return -1        
    
def exit_positions (kite,t='day',lot_size=1,tick_size=0.5):
    try:
        positions = kite.positions()
    except Exception as e:
         print(f'Exit Positions T: {t} FAILED.  Unable to Exit')
         print(e.args[0])
         return -1
       
    logtrade(f'{datetime.datetime.now(ist).strftime("%I:%M %p")}-> EXIT {t} lot_size: {lot_size} tick_size{tick_size}')

    for position in positions['day']:
        symb = position['tradingsymbol']
        q = position['quantity']
        prod = position['product']
        
        if q != 0 and prod == 'MIS' and (t == 'day' 
                or t == gettFromOption(symb)) :
            ltp = position['last_price']
            if (q>0):
                #Long position, need to sell to exit
                if (position['exchange'] == 'NFO'):
                    nfo_sell(kite,symb,lot_size,tick_size, q,ltp)
                elif (position['exchange'] == 'NSE' or position['exchange'] == 'BSE'):
                    nse_sell(kite,symb,lot_size,tick_size, q,ltp,position['exchange'])
                else:
                    logging.error("Unhandled Exchange type in positions")
                    print("Unhandled Exchange type in positions")
            else:
                #Short position, need to buy to exit
                if (position['exchange'] == 'NFO'):
                    nfo_buy(kite,symb,lot_size,tick_size, -1*q,ltp)
                elif (position['exchange'] == 'NSE'or position['exchange'] == 'BSE'):
                    nse_buy(kite,symb,lot_size,tick_size, -1*q,ltp, ltp,position['exchange'])
                else:
                    logging.error("Unhandled Exchange type in positions")
                    print("Unhandled Exchange type in positions")

def exit_given_position(kite,p):
    logging.info(f"EXIT: {p['tradingsymbol']}")
    exch = p['exchange']
    t = p['tradingsymbol']
    lot_size = p['lot_size']
    tick_size = p['tick_size']
    #ltp = p['last_price'] # SEems to be very inacurate and laggy, let exec get a updated value
    
    if exch not in ['NSE','BSE','NFO']:
        logging.error(f"Unhandled Exchange type ({p['exchange']}) in positions")
        print(f"Unhandled Exchange type ({p['exchange']})in positions")
        return -1

    txType = 'BUY' if p['quantity'] < 0 else 'SELL'
    q = abs(p['quantity'])

    return exec(kite,t,exch,txType,q=q,lot_size=lot_size,tick_size=tick_size)
    
    if p['exchange'] == 'NFO':
        if p['quantity'] > 0:
            nfo_sell(kite,p['tradingsymbol'], q=p['quantity'],ltp=p['last_price'])
        else:
            nfo_buy(kite,p['tradingsymbol'], q=-1*p['quantity'],ltp=p['last_price'])
    elif p['exchange'] == 'NSE' or p['exchange'] == 'BSE':
        if p['quantity'] > 0:
            nse_sell(kite,p['tradingsymbol'], q=p['quantity'],ltp=p['last_price'],
                     exchange=p['exchange'])
        else:
            nse_buy(kite,p['tradingsymbol'], q=-1*p['quantity'],ltp=p['last_price'],
                    exchange = p['exchange'])
    else:
        logging.error(f"Unhandled Exchange type ({p['exchange']}) in positions")
        print(f"Unhandled Exchange type ({p['exchange']})in positions")

def exit_given_positions(kite,positions,desiredPos=0):
    for position in positions:
        if long_or_short(position) != desiredPos:
            exit_given_position(kite, position)

def exitNFOPositionsONLY(kite,positions):
    for position in positions:
        if position['exchange'] == 'NFO':
            exit_given_position(kite, position)

def get_positions (kite):
    p = {}
    try:
        positions = kite.positions()
    except Exception as e:
        print(f'Get Positions FAILED.')
        print(e.args[0])
        return p
       
    for position in positions['day']:
        
        if position['quantity'] != 0 and position['product'] == 'MIS' :
            pos = {}
            pos['tradingsymbol'] = position['tradingsymbol']
            pos['quantity'] = position['quantity']
            pos['last_price'] = position['last_price']
            pos['exchange'] = position['exchange']
            pos['long_or_short'] = long_or_short(position)
            if isOption(position['exchange'],position['tradingsymbol']):
                t = gettFromOption(position['tradingsymbol'])
            else:
                t = position['tradingsymbol']
                
            if not t in p:
                p[t] = {}
                p[t]['net_position'] = pos['long_or_short']
                p[t]['positions'] = []
            else:
                #check to ensure old positions are consistant
                for other_positions in p[t]['positions']:
                    if other_positions['long_or_short'] != p[t]['net_position'] or \
                    other_positions['long_or_short'] != pos['long_or_short']:
                        p[t]['net_position'] = 'inconsistant'
                        logging.error(f"Positions for {t} inconsistant")
                        logging.error(position)
                        logging.error(other_positions)
                        logging.error(pos)
            p[t]['positions'].append(pos)
            
    return p

# kite = initKite()
# x = get_positions(kite)
# for y in x.values():
#      exit_given_positions(kite, y['positions'])

# import signal
# import sys


#def sigterm_handler(_signo, _stack_frame):
#     # Raises SystemExit(0):
#     print("Called")
#     sys.exit(0)
# signal.signal(signal.SIGTERM, sigterm_handler)

## NFO EXIT ALL MIS POSITIONS
#kite,kws = initKiteTicker()
#nfo_exit(kite,'day',200,1)