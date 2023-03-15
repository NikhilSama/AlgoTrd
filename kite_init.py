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


# set timezone to IST
ist = pytz.timezone('Asia/Kolkata')

api_key = "ctyl079egk5jwfai"
apisecret = "skrapb33nfgsrnivz9ms20w0x6odhr3t"
zacceccess_file = "Data/zerodha_kite_accesstoken.txt"
tradelog = f"Data/trades/{datetime.now().strftime('%d-%m-%y')}.trades"
bet_size = 1000
    
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

def initKiteTicker():
    kite = initKite()
    kws = KiteTicker(api_key, getAccessToken(kite))
    return kite,kws

def logtrade(s):
    with open(tradelog, "a") as f:
        f.write(s)
        f.write('\n')

def getQ (lot_size,ltp,betsize):
    if (lot_size > 1):
        return lot_size
    else:
        return max(1,round(betsize/ltp))
    #    return max(lot_size,round(betsize/ltp))

def getP (ltp,tick_size,delta):
    p = ltp * delta #Almost market order go market or at max 10% above market
    p = round(p/tick_size)*tick_size
    return p

def startOfTick():
    logtrade(f"*********************WAKE -- {datetime.now(ist)}N***********************************")

def endOfTick():
    logtrade(f"*********************SLEEP -- {datetime.now(ist)}N***********************************")

def is_not_tradable(t):
    if (t == 'NIFTY 50'):
        return True
    else:
        return False 

def nse_buy (kite,t,lot_size=1,tick_size=0.05,q=0,ltp=0,sl=0):
    if is_not_tradable(t):
        logging.info(f"{t} is not a tradable instrument.  NSE Buy not executed")
        return

    if (ltp ==0):
        #Get ltp if not provided
        ltp = kite.ltp([f"NSE:{t}"])[f"NSE:{t}"]['last_price']

    if q==0:
        q = getQ(1,ltp,bet_size)

    p = getP(ltp,tick_size,1.001)

    logtrade(f"{datetime.now(ist)}NSE BUY {t} Q:{q} P:{p} LTP:{ltp} Tick:{tick_size}")

    #1 Min TTL LIMIT ORDER priced really as a market order (10% above market)
    try:
        order_id = kite.place_order(variety=kite.VARIETY_REGULAR,tradingsymbol=t,
                     exchange=kite.EXCHANGE_NSE,
                     transaction_type=kite.TRANSACTION_TYPE_BUY,
                     quantity=q,
                     order_type=kite.ORDER_TYPE_LIMIT,
                     product=kite.PRODUCT_MIS,
                     validity=kite.VALIDITY_TTL, price = p, validity_ttl = 1)
        #SL to accompany
        if (sl):
            order_id = kite.place_order(variety=kite.VARIETY_REGULAR,tradingsymbol=t,
                                        exchange=kite.EXCHANGE_NFO,
                                        transaction_type=kite.TRANSACTION_TYPE_SELL,
                                        quantity=q,
                                        order_type=kite.ORDER_TYPE_SL,
                                        product=kite.PRODUCT_MIS,
                                        trigger_price=getP(ltp,tick_size,0.9), 
                                        price = getP(ltp,tick_size,.85))
    except Exception as e:
        print(f'NSE Buy Failed for {t}')
        print(e.args[0])
        return -1
    return order_id


def nse_sell (kite,t,lot_size=1,tick_size=0.05,q=0,ltp=0,sl=0):
    if is_not_tradable(t):
        logging.info(f"{t} is not a tradable instrument.  NSE Sell not executed")
        return

    if (ltp ==0):
        #Get ltp if not provided
        ltp = kite.ltp([f"NSE:{t}"])[f"NSE:{t}"]['last_price']

    if q==0:
        q = getQ(1,ltp,bet_size)

    p = getP(ltp,tick_size,0.999)

    logtrade(f"{datetime.now(ist)}NSE SELL {t} Q:{q} P:{p} LTP:{ltp} Tick:{tick_size}")

    #1 Min TTL LIMIT ORDER priced really as a market order (10% above market)
    try:
        order_id = kite.place_order(variety=kite.VARIETY_REGULAR,tradingsymbol=t,
                     exchange=kite.EXCHANGE_NSE,
                     transaction_type=kite.TRANSACTION_TYPE_SELL,
                     quantity=q,
                     order_type=kite.ORDER_TYPE_LIMIT,
                     product=kite.PRODUCT_MIS,
                     validity=kite.VALIDITY_TTL, price = p, validity_ttl = 1)
        #SL to accompany
        if (sl):
            order_id = kite.place_order(variety=kite.VARIETY_REGULAR,tradingsymbol=t,
                                        exchange=kite.EXCHANGE_NFO,
                                        transaction_type=kite.TRANSACTION_TYPE_BUY,
                                        quantity=q,
                                        order_type=kite.ORDER_TYPE_SL,
                                        product=kite.PRODUCT_MIS,
                                        trigger_price=getP(ltp,tick_size,0.9), 
                                        price = getP(ltp,tick_size,.85))
    except Exception as e:
        print(f'NSE Buy Failed for {t}')
        print(e.args[0])
        return -1
    return order_id

# def nse_exit (kite,t,lot_size=1,tick_size=0.05):
#     positions = kite.holdings()

#     logtrade(f"{datetime.now(ist)}NSE EXIT {t} lot_size: {lot_size} tick_size{tick_size}")

#     for position in positions:
#         #print(position)
#         symb = position['tradingsymbol']
#         if t == gettFromOpt(symb):
#             #print ("has it")
#             q = position['quantity']
#             ltp = position['last_price']
#             if (q>0):
#                 #Long position, need to sell to exit
#                 nse_sell(kite,symb,lot_size,tick_size, q,ltp)
#             else:
#                 #Short position, need to buy to exit
#                 nse_buy(kite,symb,lot_size,tick_size, -1*q,ltp)
                
            

def nfo_buy (kite,t,lot_size,tick_size, q=1,ltp=0,sl=1):
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

    
def nfo_sell (kite, t, lot_size, tick_size, q=0,ltp=0,sl=1):
    
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

def gettFromOpt (string):
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

def exit_positions (kite,t='day',lot_size=1,tick_size=0.5):
    try:
        positions = kite.positions()
    except Exception as e:
         print(f'Exit Positions T: {t} FAILED.  Unable to Exit')
         print(e.args[0])
         return -1
       
    logtrade(f"{datetime.now(ist)}NFO EXIT {t} lot_size: {lot_size} tick_size{tick_size}")

    for position in positions['day']:
        #print(position)
        symb = position['tradingsymbol']
        q = position['quantity']
        prod = position['product']
        
        if q != 0 and prod == 'MIS' and(t == 'day' or t == gettFromOpt(symb)) :
            #print ("has it")
            ltp = position['last_price']
            if (q>0):
                #Long position, need to sell to exit
                if (position['exchange'] == 'NFO'):
                    nfo_sell(kite,symb,lot_size,tick_size, q,ltp)
                elif (position['exchange'] == 'NSE'):
                    nse_sell(kite,symb,lot_size,tick_size, q,ltp)
                else:
                    logging.error("Unhandled Exchange type in positions")
                    print("Unhandled Exchange type in positions")
            else:
                #Short position, need to buy to exit
                if (position['exchange'] == 'NFO'):
                    nfo_buy(kite,symb,lot_size,tick_size, -1*q,ltp)
                elif (position['exchange'] == 'NSE'):
                    nse_buy(kite,symb,lot_size,tick_size, -1*q,ltp)
                else:
                    logging.error("Unhandled Exchange type in positions")
                    print("Unhandled Exchange type in positions")


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