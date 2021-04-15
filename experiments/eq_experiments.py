import os
from datetime import date, timedelta, datetime, time
from energyquantified import EnergyQuantified
from energyquantified.time import Frequency
from sklearn.metrics import mean_absolute_error
from energyquantified.metadata import Aggregation, Filter
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

with open('../.env', 'r') as f:
   vars_dict = dict(
      tuple(line.split('='))
      for line in f.readlines() if not line.startswith('#')
   )

eq = EnergyQuantified(api_key=vars_dict['EG_API_KEY'])
curves = eq.metadata.curves(q="se nuclear forecast")
forecasts = eq.instances.load(
   curves[0]
)
for f in forecasts:
   print(f)
def get_wind_se():
   wind_prod = eq.instances.relative(
      'SE Wind Power Production MWh/h 15min Forecast',
      begin=datetime(2019, 1, 1, 0, 0, 0),
      end=datetime(2020, 1, 1, 0, 0, 0),
      days_ahead=1,
      tag='ecsr',
      before_time_of_day=time(12, 0),
      frequency=Frequency.PT1H,
      aggregation=Aggregation.AVERAGE,
   )
   forecasted_wp = wind_prod.to_dataframe()

   wind_prod = eq.timeseries.load(
      'SE Wind Power Production MWh/h H Actual',
      begin='2015-01-01',
      end='2019-01-01',
   )
   actual_wp = wind_prod.to_dataframe()

   forecasted_wp.fillna(method='ffill', inplace=True)
   forecasted_wp.rename(columns={'SE Wind Power Production MWh/h 15min Forecast':'wp'}, inplace=True)
   actual_wp.fillna(method='ffill', inplace=True)
   actual_wp.rename(columns={'SE Wind Power Production MWh/h H Actual':'wp'}, inplace=True)

   wp_se = pd.concat([actual_wp, forecasted_wp], axis=0, ignore_index=False)
   return wp_se  

def get_nuclear_se():
   nuclear_prod = eq.instances.relative(
      curves[0],
      begin=datetime(2020, 3, 1, 0, 0, 0),
      end=datetime(2021, 1, 1, 0, 0, 0),
      days_ahead=1,
      tag='',
      before_time_of_day=time(12, 0),
      frequency=Frequency.PT1H,
      aggregation=Aggregation.AVERAGE,
   )
   forecasted_nuclear = nuclear_prod.to_dataframe()
   print(nuclear_prod)
   print(forecasted_nuclear)

   nuclear_prod = eq.timeseries.load(
      'SE Nuclear Production MWh/h H Actual',
      begin='2015-01-01',
      end='2019-01-01',
   )
   actual_nuclear = nuclear_prod.to_dataframe()

   forecasted_nuclear.fillna(method='ffill', inplace=True)
   forecasted_nuclear.rename(columns={'SE Nuclear Production MWh/h 15min Forecast':'se nuclear'}, inplace=True)
   actual_nuclear.fillna(method='ffill', inplace=True)
   actual_nuclear.rename(columns={'SE Nuclear Production MWh/h H Actual':'se nuclear'}, inplace=True)

   nuclear_se = pd.concat([actual_nuclear, forecasted_nuclear], axis=0, ignore_index=False)
   return nuclear_se

def get_solar_se():
   solar_prod = eq.instances.relative(
      'SE Solar Photovoltaic Production MWh/h 15min Forecast',
      begin=datetime(2019, 1, 1, 0, 0, 0),
      end=datetime(2020, 1, 1, 0, 0, 0),
      days_ahead=1,
      tag='ecsr',
      before_time_of_day=time(12, 0),
      frequency=Frequency.PT1H,
      aggregation=Aggregation.AVERAGE,
   )
   forecasted_solar = solar_prod.to_dataframe()

   solar_prod = eq.timeseries.load(
      'SE Solar Photovoltaic Production MWh/h H Actual',
      begin='2015-01-01',
      end='2019-01-01',
   )
   actual_solar = solar_prod.to_dataframe()

   forecasted_solar.fillna(method='ffill', inplace=True)
   forecasted_solar.rename(columns={'SE Solar Photovoltaic Production MWh/h 15min Forecast':'se solar'}, inplace=True)
   actual_solar.fillna(method='ffill', inplace=True)
   actual_solar.rename(columns={'SE Solar Photovoltaic Production MWh/h H Actual':'se solar'}, inplace=True)

   solar_se = pd.concat([actual_solar, forecasted_solar], axis=0, ignore_index=False)
   return solar_se

def get_consumption_se():
   cons = eq.instances.relative(
      'SE Consumption MWh/h 15min Forecast',
      begin=datetime(2019, 1, 1, 0, 0, 0),
      end=datetime(2020, 1, 1, 0, 0, 0),
      days_ahead=1,
      tag='ecsr',
      before_time_of_day=time(12, 0),
      frequency=Frequency.PT1H,
      aggregation=Aggregation.AVERAGE,
   )
   forecasted_cons = cons.to_dataframe()

   cons = eq.timeseries.load(
      'SE Consumption MWh/h H Actual',
      begin='2015-01-01',
      end='2019-01-01',
   )
   actual_cons = cons.to_dataframe()

   forecasted_cons.fillna(method='ffill', inplace=True)
   forecasted_cons.rename(columns={'SE Consumption MWh/h 15min Forecast':'se cons'}, inplace=True)
   actual_cons.fillna(method='ffill', inplace=True)
   actual_cons.rename(columns={'SE Consumption MWh/h H Actual':'se cons'}, inplace=True)

   cons = pd.concat([actual_cons, forecasted_cons], axis=0, ignore_index=False)
   cons.index = cons.index.tz_localize(None)
   cons = cons[~cons.index.duplicated(keep='last')]
   training_stop = datetime(2020, 1, 1, 00, 00, 00)

   cons_max = cons.loc[:training_stop].max()
   cons_min = cons.loc[:training_stop].min()
   
   cons = (cons - cons_min) / (cons_max - cons_min)
   return cons

def get_residual_se():
   res = eq.instances.relative(
      'SE resumption MWh/h 15min Forecast',
      begin=datetime(2019, 1, 1, 0, 0, 0),
      end=datetime(2020, 1, 1, 0, 0, 0),
      days_ahead=1,
      tag='ecsr',
      before_time_of_day=time(12, 0),
      frequency=Frequency.PT1H,
      aggregation=Aggregation.AVERAGE,
   )
   forecasted_res = res.to_dataframe()

   res = eq.timeseries.load(
      'SE resumption MWh/h H Actual',
      begin='2015-01-01',
      end='2019-01-01',
   )
   actual_res = res.to_dataframe()

   forecasted_res.fillna(method='ffill', inplace=True)
   forecasted_res.rename(columns={'SE resumption MWh/h 15min Forecast':'se res'}, inplace=True)
   actual_res.fillna(method='ffill', inplace=True)
   actual_res.rename(columns={'SE resumption MWh/h H Actual':'se res'}, inplace=True)

   res_se = pd.concat([actual_res, forecasted_res], axis=0, ignore_index=False)
   return res_se

#wp_se = get_wind_se()
nuclear_se = get_nuclear_se()
print(nuclear_se)
#solar_se = get_solar_se()
#cons_se = get_consumption_se()

features = pd.concat([wp_se, nuclear_se, solar_se, cons_se], axis=1, ignore_index=False)
features.index = features.index.tz_localize(None)
features = features[~features.index.duplicated(keep='last')]

price_data = pd.read_csv('../data/price_data.csv', encoding = "ISO-8859-1", sep=',', decimal='.', index_col='Delivery', parse_dates=['Delivery'])
price_spread = np.subtract(price_data['Dayahead SE3'], price_data['Intraday SE3'])

df = pd.concat([price_spread, features], ignore_index=False, axis=1)
df.fillna(method='ffill', inplace=True)
print(df)

print(stats.pearsonr(df.iloc[:,0],df.iloc[:,1]))
print(stats.pearsonr(df.iloc[:,0],df.iloc[:,2]))

corrMatrix = df.corr()
print(corrMatrix)
""" reg = LassoCV()
features = df.iloc[:,1:]
reg.fit(features.loc[:date(2019,12,31)].values, np.subtract(price_data['Dayahead SE3'].loc[:date(2019,12,31)],price_data['Intraday SE3'].loc[:date(2019,12,31)]).values)
coef = pd.Series(reg.coef_, index = features.columns)
print(coef) """