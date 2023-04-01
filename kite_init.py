#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 11:28:46 2023

@author: nikhilsama
"""

from kiteconnect import KiteConnect
from kiteconnect import KiteTicker
from datetime import datetime
import os
import requests
import time
import logging
import pytz
import re
import pandas as pd
import numpy as np
#cfg has all the config parameters make them all globals here
import cfg
globals().update(vars(cfg))

# set timezone to IST
ist = pytz.timezone('Asia/Kolkata')

api_key = "ctyl079egk5jwfai"
apisecret = "skrapb33nfgsrnivz9ms20w0x6odhr3t"
zacceccess_file = "Data/zerodha_kite_accesstoken.txt"
tradelog = f"Data/trades/{datetime.now().strftime('%d-%m-%y')}.trades"
tradelogcsv = f"Data/trades/{datetime.now().strftime('%d-%m-%y')}-trades.csv"

    
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
    if 'zerodha_access_token' in cfg:
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

def logtrade(s='',ticker=0,position=0,q=0,p=0,e='NSE'):
    time = datetime.now(ist).strftime("%I:%M:%S %p")
    p = round(p,2)
        
    if s == '':
        s = f'{time} -> {position} {q} {e}:{ticker} @ {p}'
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
        return lot_size+qToExit if qToExit else lot_size
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
            return 1.03
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

def exec(kite,t,exchange,tx_type,lot_size=1,tick_size=0.05,q=0,ltp=0,sl=0,qToExit=0):
    if is_not_tradable(t):
        logging.info(f"{t} is not a tradable instrument.  {exchange} {tx_type} not executed")
        return

    if (ltp ==0):
        #Get ltp if not provided
        ltp = kite.ltp([f"{exchange}:{t}"])[f"{exchange}:{t}"]['last_price']

    if q==0:
        q = getQ(lot_size,ltp,bet_size, qToExit)
            
    delta = getDelta(exchange,tx_type)
    
    p = getP(ltp,tick_size,delta)

    exch = getExchange(kite,exchange)

    kite_tx_type = getTxType(kite,tx_type)

    logtrade('',t,tx_type,q,p,exchange)
        
    # logtrade(f'{tx_type} {t} Q:{q} P:{p} LTP:{ltp} Tick:{tick_size}',
    #          t,tx_type,q,p)

    #1 Min TTL LIMIT ORDER priced really as a market order (10% above market)
    try:
        order_id = kite.place_order(variety=kite.VARIETY_REGULAR,tradingsymbol=t,
                     exchange=exch,
                     transaction_type=kite_tx_type,
                     quantity=q,
                     order_type=kite.ORDER_TYPE_LIMIT,
                     product=kite.PRODUCT_MIS,
                     validity=kite.VALIDITY_TTL, price = p, validity_ttl = 1)
        #SL to accompany 
        ### UNTESTED CODE -- need to get teh right tx type etc in there 
        if (sl):
            order_id = kite.place_order(variety=kite.VARIETY_REGULAR,tradingsymbol=t,
                                        exchange=exch,
                                        transaction_type=kite.TRANSACTION_TYPE_SELL,
                                        quantity=q,
                                        order_type=kite.ORDER_TYPE_SL,
                                        product=kite.PRODUCT_MIS,
                                        trigger_price=getP(ltp,tick_size,0.9), 
                                        price = getP(ltp,tick_size,.85))
    except Exception as e:
        print(f'{exchange} {tx_type} Failed for {t}')
        print(e.args[0])
        return -1
    return order_id

def nse_buy (kite,t,lot_size=1,tick_size=0.05,q=0,ltp=0,sl=0,exchange='NSE',qToExit=0):
    return exec(kite,t,exchange,'BUY',lot_size,tick_size,q,ltp,sl,qToExit)
    
    # if is_not_tradable(t):
    #     logging.info(f"{t} is not a tradable instrument.  NSE Buy not executed")
    #     return

    # if (ltp ==0):
    #     #Get ltp if not provided
    #     ltp = kite.ltp([f"NSE:{t}"])[f"NSE:{t}"]['last_price']

    # if q==0:
    #     q = getQ(1,ltp,bet_size)
        

    # p = getP(ltp,tick_size,1.001)

    # exch = getExchange(kite,exchange)
    
    # logtrade('',t,'BUY',q,p,exchange)

    # # logtrade(f'BUY {t} Q:{q} P:{p} LTP:{ltp} Tick:{tick_size}',
    # #          t,'BUY',q,p)

    # #1 Min TTL LIMIT ORDER priced really as a market order (10% above market)
    # try:
    #     order_id = kite.place_order(variety=kite.VARIETY_REGULAR,tradingsymbol=t,
    #                  exchange=exch,
    #                  transaction_type=kite.TRANSACTION_TYPE_BUY,
    #                  quantity=q,
    #                  order_type=kite.ORDER_TYPE_LIMIT,
    #                  product=kite.PRODUCT_MIS,
    #                  validity=kite.VALIDITY_TTL, price = p, validity_ttl = 1)
    #     #SL to accompany
    #     if (sl):
    #         order_id = kite.place_order(variety=kite.VARIETY_REGULAR,tradingsymbol=t,
    #                                     exchange=exch,
    #                                     transaction_type=kite.TRANSACTION_TYPE_SELL,
    #                                     quantity=q,
    #                                     order_type=kite.ORDER_TYPE_SL,
    #                                     product=kite.PRODUCT_MIS,
    #                                     trigger_price=getP(ltp,tick_size,0.9), 
    #                                     price = getP(ltp,tick_size,.85))
    # except Exception as e:
    #     print(f'NSE Buy Failed for {t}')
    #     print(e.args[0])
    #     return -1
    # return order_id


def nse_sell (kite,t,lot_size=1,tick_size=0.05,q=0,ltp=0,sl=0,exchange='NSE',qToExit=0):
    return exec(kite,t,exchange,'SELL',lot_size,tick_size,q,ltp,sl,qToExit)
    # if is_not_tradable(t):
    #     logging.info(f"{t} is not a tradable instrument.  NSE Sell not executed")
    #     return

    # if (ltp ==0):
    #     #Get ltp if not provided
    #     ltp = kite.ltp([f"NSE:{t}"])[f"NSE:{t}"]['last_price']

    # if q==0:
    #     q = getQ(1,ltp,bet_size)

    # p = getP(ltp,tick_size,0.999)

    # exch = getExchange(kite,exchange)
    # logtrade('',t,'SELL',q,p,exchange)
    # # logtrade(f'{exchange} SELL {t} Q:{q} P:{p} LTP:{ltp} Tick:{tick_size}',
    # #          t,'SELL',q,p)

    # #1 Min TTL LIMIT ORDER priced really as a market order (10% above market)
    # try:
    #     order_id = kite.place_order(variety=kite.VARIETY_REGULAR,tradingsymbol=t,
    #                  exchange=exch,
    #                  transaction_type=kite.TRANSACTION_TYPE_SELL,
    #                  quantity=q,
    #                  order_type=kite.ORDER_TYPE_LIMIT,
    #                  product=kite.PRODUCT_MIS,
    #                  validity=kite.VALIDITY_TTL, price = p, validity_ttl = 1)
    #     #SL to accompany
    #     if (sl):
    #         order_id = kite.place_order(variety=kite.VARIETY_REGULAR,tradingsymbol=t,
    #                                     exchange=kite.EXCHANGE_NFO,
    #                                     transaction_type=kite.TRANSACTION_TYPE_BUY,
    #                                     quantity=q,
    #                                     order_type=kite.ORDER_TYPE_SL,
    #                                     product=kite.PRODUCT_MIS,
    #                                     trigger_price=getP(ltp,tick_size,0.9), 
    #                                     price = getP(ltp,tick_size,.85))
    # except Exception as e:
    #     print(f'NSE Buy Failed for {t}')
    #     print(e.args[0])
    #     return -1
    # return order_id

# def nse_exit (kite,t,lot_size=1,tick_size=0.05):
#     positions = kite.holdings()

#     logtrade(f"{datetime.now(ist)}NSE EXIT {t} lot_size: {lot_size} tick_size{tick_size}")

#     for position in positions:
#         #print(position)
#         symb = position['tradingsymbol']
#         if t == gettFromOption(symb):
#             #print ("has it")
#             q = position['quantity']
#             ltp = position['last_price']
#             if (q>0):
#                 #Long position, need to sell to exit
#                 nse_sell(kite,symb,lot_size,tick_size, q,ltp)
#             else:
#                 #Short position, need to buy to exit
#                 nse_buy(kite,symb,lot_size,tick_size, -1*q,ltp)
                
            

def nfo_buy (kite,t,lot_size=1,tick_size=0.5, q=1,ltp=0,sl=1,qToExit=0):
    return exec(kite,t,'NFO', 'BUY', lot_size,tick_size,q,ltp,sl,qToExit)

    if (ltp ==0):
        #Get ltp if not provided
        ltp = kite.ltp([f"NFO:{t}"])[f"NFO:{t}"]['last_price']

    if q==0:
        q = getQ(lot_size,ltp,bet_size)

    p = getP(ltp,tick_size,1.05)

    logtrade(f"{datetime.now(ist)}NFO BUY {t} Q:{q} P:{p} LTP:: {ltp} Lot:{lot_size} Tick:{tick_size}")

    #1 Min TTL LIMIT ORDER priced really as a market order (10% above market)
    try:
        order_id = kite.place_order(variety=kite.VARIETY_REGULAR,tradingsymbol=t,
                     exchange=kite.EXCHANGE_NFO,
                     transaction_type=kite.TRANSACTION_TYPE_BUY,
                     quantity=q,
                     order_type=kite.ORDER_TYPE_LIMIT,
                     product=kite.PRODUCT_MIS,
                     validity=kite.VALIDITY_TTL, price = p, validity_ttl = 1)
        #SL to accompany
        # if (sl):
        #     order_id = kite.place_order(variety=kite.VARIETY_REGULAR,tradingsymbol=t,
        #                                 exchange=kite.EXCHANGE_NFO,
        #                                 transaction_type=kite.TRANSACTION_TYPE_SELL,
        #                                 quantity=q,
        #                                 order_type=kite.ORDER_TYPE_SL,
        #                                 product=kite.PRODUCT_MIS,
        #                                 trigger_price=getP(ltp,tick_size,0.9), 
        #                                 price = getP(ltp,tick_size,.85))
    except Exception as e:
        print(f'NFO Buy Failed for {t}')
        print(e.args[0])
        return -1
    return order_id

    
def nfo_sell (kite, t, lot_size=1, tick_size=0.5, q=0,ltp=0,sl=1, qToExit=0):
    return exec(kite,t,'NFO', 'SELL', lot_size,tick_size,q,ltp,sl,qToExit)
    if (ltp ==0):
        #Get ltp if not provided
        ltp = kite.ltp([f"NFO:{t}"])[f"NFO:{t}"]['last_price']
    #ltp = kite.ltp([f"NFO:{t}"])
    if q==0:
        q = getQ(lot_size,ltp,bet_size)

    p = getP(ltp,tick_size,0.99)
    logtrade(f"{datetime.now(ist)}NFO SELL {t} Q:{q} P:{p} LTP:: {ltp} Lot:{lot_size} Tick:{tick_size}")

 
    #1 Min TTL LIMIT ORDER priced really as a market order (10% below market)
    try:
        order_id = kite.place_order(variety=kite.VARIETY_REGULAR,tradingsymbol=t,
                     exchange=kite.EXCHANGE_NFO,
                     transaction_type=kite.TRANSACTION_TYPE_SELL,
                     quantity=q,
                     order_type=kite.ORDER_TYPE_LIMIT,
                     product=kite.PRODUCT_MIS,
                     validity=kite.VALIDITY_TTL, price = p, validity_ttl = 1)
        #SL to accompany
        # if (sl):
            # order_id = kite.place_order(variety=kite.VARIETY_REGULAR,tradingsymbol=t,
            #                             exchange=kite.EXCHANGE_NFO,
            #                             transaction_type=kite.TRANSACTION_TYPE_BUY,
            #                             quantity=q,
            #                             order_type=kite.ORDER_TYPE_SL,
            #                             product=kite.PRODUCT_MIS,
            #                             trigger_price=getP(ltp,tick_size,1.1), 
            #                             price = getP(ltp,tick_size,1.15))

    except Exception as e:
        print(f'NFO Sell Failed for {t}')
        print(e.args[0])
        return -1
    return order_id

def gettFromOption (string):
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
    
    # Define regex pattern to match all leading alphabets
    pattern = re.compile(t + r"\d+[a-zA-Z]+(\w{2})")
    
    # Search for the pattern in the string
    match = pattern.search(t)
    
    if match:
        last_two_chars = match.group(1)
        print(last_two_chars)
        if (last_two_chars == 'PE'):
            return True
    return False

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
       
    logtrade(f'{datetime.now(ist).strftime("%I:%M %p")}-> EXIT {t} lot_size: {lot_size} tick_size{tick_size}')

    for position in positions['day']:
        #print(position)
        symb = position['tradingsymbol']
        q = position['quantity']
        prod = position['product']
        
        if q != 0 and prod == 'MIS' and(t == 'day' or t == gettFromOption(symb)) :
            #print ("has it")
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

def exit_given_positions(kite,positions):
    for position in positions:
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
            if (isFutureOrOption(position['exchange'])):
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