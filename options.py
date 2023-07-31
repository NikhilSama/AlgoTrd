import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import matplotlib.dates as mdates


def putSpotPayoff (spot, strike, premium):
    return max(strike-spot,0) - premium
def callSpotPayoff (spot, strike, premium):
    return max(spot-strike,0) - premium
def putPayoff (spot, strike, premium):
    return np.where(spot < strike, strike-spot, 0) - premium
def callPayoff (spot, strike, premium):
    return np.where(spot > strike, spot-strike, 0) - premium

def plot(x,y):
    plt.plot(x,y)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.show()
    
def defaultPlot(spot_price,strick_price,premium,type='call'):
    sT = np.arange(0.98*spot_price, 1.03*spot_price)
    if type == "call":
        payoff = callPayoff(sT,strick_price,premium)
    elif type == "put":
        payoff = putPayoff(sT,strick_price,premium)
    elif type == "straddle":
        payoff = callPayoff(sT,strick_price,premium) + putPayoff(sT,strick_price,premium) 
    plot(sT,payoff)
    
defaultPlot(10000,10000,20,'straddle')
    

