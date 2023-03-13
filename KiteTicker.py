#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 14:14:12 2023

@author: nikhilsama
"""

#!python
import logging
import kite_init as ki 
from time import time, ctime, sleep

#logging.basicConfig(level=logging.DEBUG)

# Initialise
kite, kws = ki.initKiteTicker()
buy_order_id,sell_order_id = 0,0
tradingsymbol = "MINDSPACE-RR"

def on_ticks(ws, ticks):
    # Callback to receive ticks.
    logging.debug("Ticks: {}".format(ticks))
    bid = ticks[0]['depth']['buy'][0]['price']
    ask = ticks[0]['depth']['sell'][0]['price']
    
    print(ctime(time()),">>>Bid=",bid," ASK=",ask, " Last Trade: ", ticks[0]["last_trade_time"])
    
    global buy_order_id #Apparently, we must use keyword global to tell python we are using the global variable order_id, and not to create a new local instance just because we are assigning a value to order id within this function
    global sell_order_id #Apparently, we must use keyword global to tell python we are using the global variable order_id, and not to create a new local instance just because we are assigning a value to order id within this function

    captialToDeploy = 100000
    qt = int(round(captialToDeploy/ask,0))
    
    if (buy_order_id and sell_order_id):
        #cancel
        kite.cancel_order(variety=kite.VARIETY_REGULAR,
                                     order_id=buy_order_id)
        kite.cancel_order(variety=kite.VARIETY_REGULAR,
                                     order_id=buy_order_id)

    #Place buy order at the ask
    buy_order_id = kite.place_order(variety=kite.VARIETY_REGULAR,tradingsymbol=tradingsymbol,
                                exchange=kite.EXCHANGE_NSE,
                                transaction_type=kite.TRANSACTION_TYPE_BUY,
                                quantity=qt,
                                order_type=kite.ORDER_TYPE_LIMIT,
                                product=kite.PRODUCT_CNC,
                                validity=kite.VALIDITY_DAY, price = ask)
    #Place sell order at the bid
    sell_order_id = kite.place_order(variety=kite.VARIETY_REGULAR,tradingsymbol=tradingsymbol,
                                exchange=kite.EXCHANGE_NSE,
                                transaction_type=kite.TRANSACTION_TYPE_SELL,
                                quantity=qt,
                                order_type=kite.ORDER_TYPE_LIMIT,
                                product=kite.PRODUCT_CNC,
                                validity=kite.VALIDITY_DAY, price = bid)
    print("SLEEP START", ctime(time()))
    sleep(10)
    print ("SLEEP END", ctime(time()))


def on_connect(ws, response):
    # Callback on successful connect.
    # Subscribe to a list of instrument_tokens (MINDSPACE and ACC here).
    ws.subscribe([5710849])

    # Set MINDSPACE to tick in `full` mode.
    ws.set_mode(ws.MODE_FULL, [5710849])

def on_close(ws, code, reason):
    # On connection close stop the event loop.
    # Reconnection will not happen after executing `ws.stop()`
    ws.stop()

# Assign the callbacks.
kws.on_ticks = on_ticks
kws.on_connect = on_connect
kws.on_close = on_close

# Infinite loop on the main thread. Nothing after this will run.
# You have to use the pre-defined callbacks to manage subscriptions.
kws.connect()