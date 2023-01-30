import os
from datetime import date, timedelta, datetime, time
from statsmodels.tsa.stattools import ccf
from matplotlib.gridspec import GridSpec
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import stats, pearsonr, spearmanr
import requests
import re

'''
   Here the data features correlation with the price spread was analysed.
'''


def analyze_features(price_data, feature_data):
    val_start = datetime(2019, 1, 1, 00, 00, 00)
    delta_day = timedelta(days=1)
    forecast = feature_data.iloc[:, :4].loc[val_start:]
    cap = feature_data.iloc[:, 4:-5]
    flow = feature_data.iloc[:, -5:]
    for c in forecast.columns:
        print(' \n \n {}'.format(c))
        p_dayahead, p_intraday, p_spread = pearsonr(price_data['Dayahead SE3'].loc[val_start:], forecast[c]), pearsonr(
            price_data['Intraday SE3'].loc[val_start:], forecast[c]), pearsonr(price_data['Spread'].loc[val_start:], forecast[c])

        print('   Dayahead    Intraday    Spread  ')
        print('Pearson:     {}          {}          {}      '.format(
            p_dayahead, p_intraday, p_spread))

    for c in cap.columns:
        print(' \n \n {}'.format(c))
        s_dayahead, s_intraday, s_spread = spearmanr(price_data['Dayahead SE3'], cap[c]), spearmanr(
            price_data['Intraday SE3'], cap[c]), spearmanr(price_data['Spread'], cap[c])

        print('   Dayahead    Intraday    Spread  ')
        print('Spearman:     {}          {}          {}      '.format(
            s_dayahead, s_intraday, s_spread))

    l = len(price_data['Dayahead SE3'])
    lookback = 24*30
    sl = 2 / np.sqrt(l)
    colors = sns.color_palette("tab10")
    for i, c in enumerate(flow.columns):
        print(c)
        cc_dayahead = ccf(price_data['Dayahead SE3'], flow[c])
        cc_intraday = ccf(price_data['Intraday SE3'], flow[c])
        cc_spread = ccf(price_data['Spread'], flow[c])

        plt.plot(np.divide(list(range(lookback)), 24),
                 cc_dayahead[:lookback], label='Dayahead', color=colors[0], alpha=0.45)
        plt.plot(np.divide(list(range(lookback)), 24),
                 cc_intraday[:lookback], label='Intraday', color=colors[1], alpha=0.45)
        plt.plot(np.divide(list(range(lookback)), 24),
                 cc_spread[:lookback], label='Spread', color=colors[2], alpha=0.45)

        plt.hlines(sl, xmin=0, xmax=lookback/24,
                   color=colors[3], label='Confidence Interval')
        plt.hlines(-sl, xmin=0, xmax=lookback/24, color=colors[3])

        plt.xlabel('Lag in days')
        plt.ylabel('Cross correlation')

        plt.legend()
        plt.show()

    l = len(price_data['Dayahead SE3'])
    lookback = 24*30
    sl = 2 / np.sqrt(l)
    colors = sns.color_palette("tab10")

    cc_dayahead = ccf(price_data['Spread'], price_data['Dayahead SE3'])
    cc_intraday = ccf(price_data['Spread'], price_data['Intraday SE3'])

    plt.plot(np.divide(list(range(lookback)), 24),
             cc_dayahead[:lookback], label='Dayahead', color=colors[0], alpha=0.45)
    plt.plot(np.divide(list(range(lookback)), 24),
             cc_intraday[:lookback], label='Intraday', color=colors[1], alpha=0.45)

    plt.hlines(sl, xmin=0, xmax=lookback/24,
               color=colors[3], label='Confidence Interval')
    plt.hlines(-sl, xmin=0, xmax=lookback/24, color=colors[3])

    plt.xlabel('Lag in days')
    plt.ylabel('Cross correlation')

    plt.legend()
    plt.show()


price_data = pd.read_csv('../data/price_data.csv', encoding="ISO-8859-1",
                         sep=',', decimal='.', index_col='Delivery', parse_dates=['Delivery'])

nuclear = pd.read_csv('../data/production_data/nuclear.csv', encoding="ISO-8859-1",
                      sep=',', decimal='.', index_col='date', parse_dates=['date'])
solar = pd.read_csv('../data/production_data/solar.csv', encoding="ISO-8859-1",
                    sep=',', decimal='.', index_col='date', parse_dates=['date'])
wp = pd.read_csv('../data/production_data/wp.csv', encoding="ISO-8859-1",
                 sep=',', decimal='.', index_col='date', parse_dates=['date'])
cons = pd.read_csv('../data/consumption_data/cons.csv', encoding="ISO-8859-1",
                   sep=',', decimal='.', index_col=0, parse_dates=True)
cap = pd.read_csv('../data/elspot_dayahead/cap.csv', encoding="ISO-8859-1",
                  sep=',', decimal='.', index_col=0, parse_dates=True)
flow = pd.read_csv('../data/elspot_dayahead/flow.csv', encoding="ISO-8859-1",
                   sep=',', decimal='.', index_col=0, parse_dates=True)

price_data['Spread'] = np.subtract(
    price_data['Dayahead SE3'], price_data['Intraday SE3'])

feature_data = pd.concat(
    [nuclear, solar, wp, cons, cap, flow], ignore_index=False, axis=1)
feature_data.fillna(method='ffill', inplace=True)

analyze_features(price_data, feature_data)
