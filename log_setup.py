#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 09:47:01 2023

@author: nikhilsama
"""
import logging
from datetime import datetime

## LOG SETUP 
# log_format = "%(asctime)s::%(levelname)s::%(name)s::"\
#              "%(filename)s::%(lineno)d::%(message)s"

log_format = "%(asctime)s::%(message)s"

fname = datetime.now().strftime('%d-%m-%y')

#DateTime may have logged some messages already, and a default logger would 
# have been created.  If we dont remove this, then our basicConfig call will 
# be ignored as the root logger can only be configured once at startup

root = logging.getLogger()
if root.handlers:
    for handler in root.handlers:
        root.removeHandler(handler)
#####

 
logging.basicConfig(filename=f'Data/logs/{fname}.log', level='INFO', 
                    format=log_format, datefmt='%I:%M:%S %p')

print(f"Logging to Data/logs/{fname}.log")
logging.debug("LOG SETUP")
logging.info("This is an informational message")
#logging.warning("Careful! Something does not look right")
#logging.error("You have encountered an error")
#logging.critical("You are in trouble")
## END