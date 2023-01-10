from py5paisa import FivePaisaClient
import json
import pandas as pd
from py5paisa.order import Order, OrderType, Exchange
cred={
    "APP_NAME":"",
    "APP_SOURCE":"",
    "USER_ID":"",
    "PASSWORD":"",
    "USER_KEY":"",
    "ENCRYPTION_KEY":""
    }

client = FivePaisaClient(email="", passwd="", dob="",cred=cred)
client.login()
import requests
from __future__ import print_function
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import math
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib import style
import plotly
import plotly.express as px
import plotly.graph_objects as go
from nsepython import *
from nsepython import *
import pandas as pd
from scipy.stats import norm
from scipy import optimize
from math import sqrt, log
import numpy as np
import time
import os
import glob
import warnings
from jugaad_data.nse import index_csv, index_df
from datetime import date

warnings.simplefilter("ignore")

N_prime = norm.pdf
N = norm.cdf

###############################################################################################

# Variables

date = 'Today'
folder_path = 'your/folder/path/here/'

expiry = '2023-02-23 14:30:00'
expiry_for_strike = '2023-02-23 15:30:00'
expiry_for_ul = '23-Feb-2023'
n_days = 4 # Number of trading days till expiry / till position is closed

num_simulations = 10000

###############################################################################################

# Underlying Volatility
underlying = [
    'NIFTY'
]
vol_df = pd.DataFrame()
for underlying in underlying:
    path = folder_path + 'sample_input_vol.csv'
    df = pd.read_csv(path)
    current_vol = df.tail(1)
    vol_df = pd.concat([vol_df, current_vol], axis=0)

underlying = [
    'NIFTY'
]
    
vol_df = vol_df.reset_index()
vol_df['symbol'] = ''
for i in range(0, len(vol_df)):
    vol_df['symbol'][i] = underlying[i]


vol_df = vol_df[['symbol', 'vol_c2c_20d']]

vol_df['minus_500bp'] = - 0.05
vol_df['minus_300bp'] = - 0.03
vol_df['minus_100bp'] = - 0.01
vol_df['no_change'] = 0
vol_df['plus_100bp'] = 0.01
vol_df['plus_300bp'] = 0.03
vol_df['plus_500bp'] = 0.05
vol_df['plus_700bp'] = 0.07
vol_df['plus_1000bp'] = 0.1

def fetch_prices(strike, underlying, expiry):

    path = '/Users/vamsikrishnasb/My Drive/Financial Analysis/5paisa/scripmaster-csv-format.csv'
    df = pd.read_csv(path)
    option_type1 = 'PE'
    leg_type1 = 'short'
    option_type2 = 'CE'
    leg_type2 = 'short'

    data = [
        [underlying, expiry, strike, option_type1, leg_type1],
        [underlying, expiry, strike, option_type2, leg_type2]
    ]

    spread = pd.DataFrame(data, columns=['underlying', 'expiry', 'strike', 'otm_option_type', 'leg_type'])
    spread['scrip_code'] = ''

    # Scrip Codes function
    for strike, option_type, i in zip(spread['strike'], spread['otm_option_type'], range(0, len(spread['strike']))):
        def scrip_code(underlying, expiry, strike, option_type):
            scrip_code = df[(
                df['ExchType'] == 'D') 
                & (df['Root'] == underlying) 
                & (df['Expiry'] == expiry)
                & (df['StrikeRate'] == strike)
                & (df['CpType'] == option_type)
            ][['Scripcode']]
            return scrip_code['Scripcode'].max()
        scrip_code = scrip_code(underlying, expiry, strike, option_type)
        spread['scrip_code'][i] = scrip_code


    spread['scrip_code'] = pd.to_numeric(spread['scrip_code'], errors = 'coerce')
    spread

    scrip1 = spread['scrip_code'][0].astype(str)
    scrip2 = spread['scrip_code'][1].astype(str)

    spread_scrips=[
        {"Exchange":"N","ExchangeType":"D","ScripCode":scrip1},
        {"Exchange":"N","ExchangeType":"D","ScripCode":scrip2}
    ]

    spread_prices = client.fetch_market_depth(spread_scrips)
    price1 = spread_prices['Data'][0]['LastTradedPrice']
    price2 = spread_prices['Data'][1]['LastTradedPrice']

    return price1, price2

# Long Straddle

underlying = 'NIFTY'

pnl_df = pd.DataFrame(columns=['symbol', 'expiry', 'ul_price', 'strike', 'vol', 'mean', 'std', 'min', 'max', 'pc_profit'])
edge_df = pd.DataFrame(columns=['symbol', 'expiry', 'ul_price', 'strike', 'vol', 'mean', 'std', 'min', 'max', 'pc_positive'])
last_price = nse_quote_ltp(underlying, expiry_for_ul, "Fut")

# Finding ATM strike
path = folder_path + 'sample_input_options.csv'
df = pd.read_csv(path)

call = df[
    (df['expiry'] == expiry_for_strike)
    & (df['moneyness'] < 0)
]
atm_call_strike = call['strike'].min()

put = df[
    (df['expiry'] == expiry_for_strike)
    & (df['moneyness'] > 0)
]
atm_put_strike = put['strike'].max()

if ((atm_call_strike - last_price) <= (last_price - atm_put_strike)):
    strike = atm_call_strike
else:
    strike = atm_put_strike
strike = 18000
# num_days = n_days # Number of trading days till expiry
price1, price2 = fetch_prices(strike, underlying, expiry)

straddle_df = df[
    (df['strike'] == strike)
    & (df['expiry'] == expiry_for_strike)
]
straddle_df['price1'] = price1
straddle_df['price2'] = price2
r = straddle_df['rf_rate'].iloc[-1]
t = (straddle_df['days_to_expiry'].iloc[-1] - n_days) / 365.0
# num_days = math.floor(straddle_df['days_to_expiry'].iloc[-1] - n_days)
num_days = n_days
K = straddle_df['strike'].iloc[-1]
# print(r, t, K)
max_loss = - (price1 + price2)
breakeven_price1 = strike - (price1 + price2)
breakeven_price2 = strike + (price1 + price2)

# Underlying simulation

vol_list = vol_df[vol_df['symbol'] == underlying]
vol_list = vol_list.transpose()
vol_list = vol_list.reset_index()
vol_list = vol_list.rename(columns={'index': 'vol_list', i: 'vol'})
vol_list = vol_list.drop(0)
vol_list = vol_list.reset_index()
vol_list = vol_list.drop('index', 1)
# current_vol = vol_list['vol'][0]
current_vol = 0.12
vol_list = vol_list.drop(0)
vol_list = vol_list.reset_index()
vol_list = vol_list.drop('index', 1)

for vol in vol_list['vol']:

    daily_vol = ((current_vol + vol) / 16)

    simulation_df = pd.DataFrame()

    for x in range(num_simulations):
        count = 0

        price_series = []

        price = last_price * (1 + np.random.normal(0, daily_vol))
        price_series.append(price)

        for y in range(num_days):
            if count == (num_days - 1):
                break
            price = price_series[count] * (1 + np.random.normal(0, daily_vol))
            price_series.append(price)
            count += 1

        simulation_df[x] = price_series

    final_price_df = simulation_df.tail(1).transpose()

    final_price_df = final_price_df.rename(columns={num_days - 1: 'final_price'})
    
    F = final_price_df['final_price']
    F_mean = final_price_df['final_price'].mean()
    d1_mean = (np.log(F_mean / K) + ((current_vol + vol) ** 2 / 2) * t) / ((current_vol + vol) * np.sqrt(t))
    d2_mean = d1_mean - (current_vol + vol) * np.sqrt(t)
    final_price_df['d1_theo'] = (np.log(F / K) + ((current_vol + vol) ** 2 / 2) * t) / ((current_vol + vol) * np.sqrt(t))
    final_price_df['d2_theo'] = final_price_df['d1_theo'] - (current_vol + vol) * np.sqrt(t)
    d1 = final_price_df['d1_theo']
    d2 = final_price_df['d2_theo']
    final_price_df['pnl'] = (price1 + price2) - ((F * N(d1) -  N(d2) * K) * np.exp(-r * t) + (- F * N(-d1) +  N(-d2) * K) * np.exp(-r * t))
    final_price_df['straddle'] = ((F * N(d1) -  N(d2) * K) * np.exp(-r * t) + (- F * N(-d1) +  N(-d2) * K) * np.exp(-r * t))
    final_price_df['edge'] = (final_price_df['straddle'].mean()) - ((F_mean * N(d1_mean) -  N(d2_mean) * K) * np.exp(-r * t) + (- F_mean * N(-d1_mean) +  N(-d2_mean) * K) * np.exp(-r * t))
    print(vol, price1 + price2, final_price_df['straddle'].mean(), ((F_mean * N(d1_mean) -  N(d2_mean) * K) * np.exp(-r * t) + (- F_mean * N(-d1_mean) +  N(-d2_mean) * K) * np.exp(-r * t)))
    data = [
        [
            underlying,
            expiry,
            last_price,
            strike,
            vol,
            final_price_df['pnl'].mean(),
            final_price_df['pnl'].std(),
            final_price_df['pnl'].min(),
            final_price_df['pnl'].max(),
            (final_price_df[final_price_df.pnl > 0].count()['pnl'] / num_simulations * 100)
        ]
    ]
    temp = pd.DataFrame(data, columns=['symbol', 'expiry', 'ul_price', 'strike', 'vol', 'mean', 'std', 'min', 'max', 'pc_profit'])
    pnl_df = pd.concat([pnl_df, temp], axis=0)
    
    data = [
        [
            underlying,
            expiry,
            last_price,
            strike,
            vol,
            final_price_df['edge'].mean(),
            final_price_df['edge'].std(),
            final_price_df['edge'].min(),
            final_price_df['edge'].max(),
            (final_price_df[final_price_df.edge > 0].count()['edge'] / num_simulations * 100)
        ]
    ]
    temp = pd.DataFrame(data, columns=['symbol', 'expiry', 'ul_price', 'strike', 'vol', 'mean', 'std', 'min', 'max', 'pc_positive'])
    edge_df = pd.concat([edge_df, temp], axis=0)
    
    print("Returns Distribution", pnl_df)