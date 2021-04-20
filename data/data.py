import pandas as pd
import numpy as np
import datetime
import holidays
import requests
import re
from sklearn import linear_model
from energyquantified import EnergyQuantified
from energyquantified.time import Frequency
from energyquantified.metadata import Aggregation, Filter
from os import path
from datetime import date, timedelta, datetime, time
#from PyEMD import EMD, CEEMDAN, Visualisation
from scipy.stats import zscore

with open('../.env', 'r') as f:
    vars_dict = dict(
        tuple(line.split('='))
        for line in f.readlines() if not line.startswith('#')
    )
eq = EnergyQuantified(api_key=vars_dict['EG_API_KEY'])

def add_intraday( year ):
    intraday_prices = {
        'Intraday SE3' : [],
        'Intraday SE4' : []
    }

    start_date = datetime.date(year, 1, 1)
    end_date = datetime.date(year+1, 1, 1)
    delta = datetime.timedelta(days=1)

    while start_date < end_date:
        print (start_date.strftime("%Y-%m-%d"))
        SE3_intraday, SE4_intraday = get_intraday_price( start_date )
        
        SE3_hourly_prices = [None]*24
        SE4_hourly_prices = [None]*24
        for d in SE3_intraday:
            if d['Columns'][4]['Name'] == '':
                continue
            else:
                try:
                    i = int(str(d['Name']).split('-')[-1].split()[0]) - 1
                    SE3_hourly_prices[i] = float( d['Columns'][4]['Value'].replace(',','.') )
                except:
                    print('Nicht')

        for d in SE4_intraday:
            if d['Columns'][4]['Name'] == '':
                continue
            else:
                try:
                    i = int(str(d['Name']).split('-')[-1].split()[0]) - 1
                    SE4_hourly_prices[i] = float( d['Columns'][4]['Value'].replace(',','.') )
                except:
                    print('Nicht')

        intraday_prices['Intraday SE3'] = intraday_prices['Intraday SE3'] + SE3_hourly_prices
        intraday_prices['Intraday SE4'] = intraday_prices['Intraday SE4'] + SE4_hourly_prices
        
        start_date += delta

    return intraday_prices

def arrange_time_relations(df):
    start_date = datetime.datetime(2015, 1, 1, 12, 00, 00)
    end_date = datetime.datetime(2021, 1, 1, 12, 00, 00)
    
    delta_day = datetime.timedelta(days=1)
    delta_twelve_hours = datetime.timedelta(hours=12)
    delta_eleven_hours = datetime.timedelta(hours=11)
    delta_twenty_three_hours = datetime.timedelta(hours=23)

    data = {
        'Date' : [],
        'Dayahead SE3' : [],
        'Dayahead SE4' : [],
        'Intraday SE3' : [],
        'Intraday SE4' : []
    }

    while start_date < end_date:
        data['Date'].append(start_date)

        todays_data = df.loc[
                (start_date-delta_twelve_hours)
                : 
                (start_date+delta_eleven_hours)
            ]
        data['Dayahead SE3'].append(
            todays_data['Dayahead SE3'].to_numpy()
        )
        data['Dayahead SE4'].append(
            todays_data['Dayahead SE4'].to_numpy()
        )

        last_twenty_four_hours = df.loc[
            (start_date-delta_twenty_three_hours)
            :
            (start_date)
        ]

        data['Intraday SE3'].append(
            last_twenty_four_hours['Intraday SE3'].to_numpy()
        )
        data['Intraday SE4'].append(
            last_twenty_four_hours['Intraday SE4'].to_numpy()
        )

        start_date += delta_day
    return data

def arrange_price_data():
    for year in range(2015, 2021):
        if path.exists('price_yearly/price_data_{}.csv'.format(year)):
            continue
        else:
            df = load_dayahead_prices( year )
            df = df.drop_duplicates(subset=['Delivery'],keep='last')
        
            intraday = add_intraday( year )
            idf = pd.DataFrame.from_dict(intraday)
            idf.to_csv('intraday_price_data_{}.csv'.format(year), index=False)

            df['Intraday SE3'] = intraday['Intraday SE3']
            df['Intraday SE4'] = intraday['Intraday SE4']
            
            df.to_csv('price_yearly/price_data_{}.csv'.format(year), index=False)

def get_cap_data():
    # Elspot flow data
    cap = load_market_data('elspot_dayahead/elspot-capacities-se', 2015, 'cap ')

    for y in range(2016, 2021):
        cap2 = load_market_data('elspot_dayahead/elspot-capacities-se', y, 'cap ')
        cap = pd.concat([cap, cap2], axis=0, ignore_index=False)

    cap.dropna(axis=1, inplace=True)
    cap = cap[~cap.index.duplicated(keep='last')]
    training_stop = datetime(2020, 1, 1, 00, 00, 00)

    cap_max = cap.loc[:training_stop].max()
    cap_min = cap.loc[:training_stop].min()
    
    cap = (cap - cap_min) / (cap_max - cap_min)
    return cap

def get_flow_data():
    # Elspot flow data
    flow = load_market_data('elspot_dayahead/elspot-flow-se', 2015, 'flow ')

    for y in range(2016, 2021):
        flow2 = load_market_data('elspot_dayahead/elspot-flow-se', y, 'flow ')
        flow = pd.concat([flow, flow2], axis=0, ignore_index=False)

    flow.dropna(axis=1, inplace=True)
    flow = flow[~flow.index.duplicated(keep='last')]
    training_stop = datetime(2020, 1, 1, 00, 00, 00)

    flow_max = flow.loc[:training_stop].max()
    flow_min = flow.loc[:training_stop].min()
    
    flow = (flow - flow_min) / (flow_max - flow_min)
    return flow

def seasonal_data():
    start_date = datetime(2015, 1, 1, 00, 00, 00)
    end_date = datetime(2021, 1, 1, 00, 00, 00)

    se_holidays = holidays.CountryHoliday('SE')
    delta_day = timedelta(days=1)

    data = {
        'date_time'     : [],
        'holiday'       : [],
        'week'         : [],
        'day of week'   : []
    }

    while start_date < end_date:
        data['date_time'].append(start_date)

        is_holiday = 1 if start_date in se_holidays else 0
        data['holiday'].append(is_holiday)
        data['week'].append(start_date.isocalendar()[1] - 1)
        data['day of week'].append(start_date.weekday())
        
        start_date += delta_day

    df = pd.DataFrame.from_dict(data)
    df.set_index('date_time', inplace=True)
    return df

def load_dayahead_prices( year ):
    # Load dayahead prices downloaded from Nordpool's website
    if year == 2015:
        df = pd.read_csv('elspot_dayahead/elspot-prices_{}_hourly_eur.csv'.format(year), sep=';', decimal=",", header=2, encoding='iso 8859-1', usecols=[0, 1, 5, 6])
    else:
        df = pd.read_csv('elspot_dayahead/elspot-prices_{}_hourly_eur.csv'.format(year), sep=';', decimal=",", header=2, usecols=[0, 1, 5, 6])

    df.rename(columns={"Unnamed: 0": "Date"}, inplace=True)
    df.rename(columns={"SE3": "Dayahead SE3"}, inplace=True)
    df.rename(columns={"SE4": "Dayahead SE4"}, inplace=True)

    delivery = []
    for index, row in df.iterrows():
        d_str = row['Date'] + ' ' + row['Hours'].split()[0]
        delivery.append(
            datetime.datetime.strptime(d_str, '%d-%m-%Y %H')
        )

    df.insert(0, "Delivery", delivery, True)
    df.drop(columns=['Date', 'Hours'], inplace=True)

    return df

def get_intraday_price( date ):
    day, month, year = date.strftime('%d'), date.strftime('%m'), date.strftime('%Y')
    URL = "https://www.nordpoolgroup.com/api/marketdata/page/194?currency=,EUR,EUR,EUR&endDate={}-{}-{}&entityName=SE3".format(day, month, year)
    try:
        SE3_data = requests.get(url = URL).json()['data']['Rows'][:24]
    except:
        SE3_data = []

    if(len(SE3_data) < 24):
        print('SE3')
        print(len(SE3_data))

    URL = "https://www.nordpoolgroup.com/api/marketdata/page/194?currency=,EUR,EUR,EUR&endDate={}-{}-{}&entityName=SE4".format(day, month, year)
    try:
        SE4_data = requests.get(url = URL).json()['data']['Rows'][:24]
    except:
        SE4_data = []

    if(len(SE4_data) < 24):
        print('SE4')
        print(len(SE4_data))

    return SE3_data, SE4_data

def combine_yearly_price_data():
    df =  pd.read_csv('price_yearly/price_data_2015.csv', sep=',', decimal=".", index_col='Delivery', parse_dates=['Delivery'])
    for year in range(2016, 2021):
        df2 =  pd.read_csv('price_yearly/price_data_{}.csv'.format(year), sep=',', decimal=".", index_col='Delivery', parse_dates=['Delivery'])
        df = df.append(df2, ignore_index = False)


    df.fillna(0, inplace=True)
    df.replace(to_replace=0, method='ffill', inplace=True)
    #print(df.describe())

    df.drop(columns=['Dayahead SE4', 'Intraday SE4'], inplace=True)

    standardize(df)    
    return df

def load_market_data(path, year, prefix):
    df = pd.read_csv('{}_{}_hourly.csv'.format(path, year), sep=';', decimal=",", header=2, encoding='iso 8859-1')
    df.rename(columns={"Unnamed: 0": "Date"}, inplace=True)
    df.fillna(0, inplace=True)

    delivery = []
    for index, row in df.iterrows():
        d_str = row['Date'] + ' ' + row['Hours'].split()[0]
        delivery.append(
            datetime.strptime(d_str, '%d-%m-%Y %H')
        )
    regions = re.compile('> SE3')
    cols = [c for c in df.columns if regions.search(c)]
    df = df[cols]
    df = df.add_prefix(prefix)

    df.index = delivery
    return df

def standardize(df):
    training_stop = datetime(2020, 1, 1, 00, 00, 00)
    for column in df.columns:
        mean = df[column].loc[:training_stop].mean()
        std = df[column].loc[:training_stop].std()

        df['{} (s)'.format(column)] = (df[column] - mean) / std
    return df

def normalize(df):
    for column in df.columns:
        minc = df[column].min()
        maxc = df[column].max()

        df['{} (n)'.format(column)] = (df[column] - minc) / (maxc - minc)
    return df

def create_imf(df):
    ceemdan = CEEMDAN()

    for column in ['Dayahead SE3', 'Intraday SE3']:
        print("Creating IMFS for {}".format(column))
        ceemdan.ceemdan(df[column].to_numpy(), max_imf=5)
        imfs, res = ceemdan.get_imfs_and_residue()

        for n, imf in enumerate(imfs):
            df['{} (imf {})'.format(column, n+1)] = imf
        df['{} (res)'.format(column)] = res
    
    return df

def get_eq_data(region, variable, tag, path_name, shortname):
    if not path.exists(path_name):
        f = eq.instances.relative(
        '{} {} 15min Forecast'.format(region, variable),
        begin=datetime(2019, 1, 1, 0, 0, 0),
        end=datetime(2021, 1, 1, 0, 0, 0),
        days_ahead=1,
        tag=tag,
        before_time_of_day=time(12, 0),
        frequency=Frequency.PT1H,
        aggregation=Aggregation.AVERAGE,
        )
        forecasted = f.to_dataframe()

        a = eq.timeseries.load(
            '{} {} H Actual'.format(region, variable),
            begin='2015-01-01',
            end='2019-01-01',
        )
        actual = a.to_dataframe()

        forecasted.fillna(method='ffill', inplace=True)
        forecasted.rename(columns={'{} {} 15min Forecast'.format(region, variable):'{} {}'.format(shortname, region)}, inplace=True)
        actual.fillna(method='ffill', inplace=True)
        actual.rename(columns={'{} {} H Actual'.format(region, variable):'{} {}'.format(shortname, region)}, inplace=True)

        data = pd.concat([actual, forecasted], axis=0, ignore_index=False)
        data.index = data.index.tz_localize(None)
        data = data[~data.index.duplicated(keep='last')]
        data.to_csv(path_name, index=True)
    else:
        cons  = pd.read_csv(path_name, encoding = "ISO-8859-1", sep=',', decimal='.', index_col=0, parse_dates=True)

    training_stop = datetime(2020, 1, 1, 00, 00, 00)
    data_max = data.loc[:training_stop].max()
    data_min = data.loc[:training_stop].min()
    
    data = (data - data_min) / (data_max - data_min)
    return data

def combine_production_data():
    wp      = get_eq_data('SE', 'Wind Power Production MWh/h', 'ecsr', 'production_data/wp.csv', 'wp')
    solar   = get_eq_data('SE', 'Solar Photovoltaic Production MWh/h', 'ecsr', 'production_data/solar.csv', 'solar')
    nuclear = get_eq_data('SE', 'Nuclear Production MWh/h', '', 'production_data/nuclear.csv', 'nuclear')
    return pd.concat([wp, solar, nuclear], axis=1, ignore_index=True)

""" def load_wind_production(region):
    if not path.exists('production_data/wp.csv'):
        wind_prod = eq.instances.relative(
            '{} Wind Power Production MWh/h 15min Forecast'.format(region),
            begin=datetime(2019, 1, 1, 0, 0, 0),
            end=datetime(2021, 1, 1, 0, 0, 0),
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
        forecasted_wp.rename(columns={'{} Wind Power Production MWh/h 15min Forecast'.format(region):'wp {}'.format(region)}, inplace=True)
        actual_wp.fillna(method='ffill', inplace=True)
        actual_wp.rename(columns={'{} Wind Power Production MWh/h H Actual'.format(region):'wp {}'.format(region)}, inplace=True)

        wp = pd.concat([actual_wp, forecasted_wp], axis=0, ignore_index=False)
        wp.index = wp.index.tz_localize(None)
        wp = wp[~wp.index.duplicated(keep='last')]
        wp.to_csv('production_data/wp.csv', index=True)
    else:
        wp  = pd.read_csv('production_data/wp.csv', encoding = "ISO-8859-1", sep=',', decimal='.', index_col=0, parse_dates=True)
    
    training_stop = datetime(2020, 1, 1, 00, 00, 00)
    wp_max = wp.loc[:training_stop].max()
    wp_min = wp.loc[:training_stop].min()
    
    wp = (wp - wp_min) / (wp_max - wp_min)
    return wp """

""" def load_solar_production(region):
    if not path.exists('production_data/solar.csv'):
        solar_prod = eq.instances.relative(
            '{} Solar Photovoltaic Production MWh/h 15min Forecast'.format(region),
            begin=datetime(2019, 1, 1, 0, 0, 0),
            end=datetime(2021, 1, 1, 0, 0, 0),
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
        forecasted_solar.rename(columns={'{} Solar Photovoltaic Production MWh/h 15min Forecast'.format(region):'solar {}'.format(region)}, inplace=True)
        actual_solar.fillna(method='ffill', inplace=True)
        actual_solar.rename(columns={'{} Solar Photovoltaic Production MWh/h H Actual'.format(region):'solar {}'.format(region)}, inplace=True)

        solar = pd.concat([actual_solar, forecasted_solar], axis=0, ignore_index=False)
        solar.index = solar.index.tz_localize(None)
        solar = solar[~solar.index.duplicated(keep='last')]
        solar.to_csv('production_data/solar.csv', index=True)
    else:
        solar  = pd.read_csv('production_data/solar.csv', encoding = "ISO-8859-1", sep=',', decimal='.', index_col=0, parse_dates=True)

    training_stop = datetime(2020, 1, 1, 00, 00, 00)
    solar_max = solar.loc[:training_stop].max()
    solar_min = solar.loc[:training_stop].min()
    
    solar = (solar - solar_min) / (solar_max - solar_min)
    return solar """

""" 
    if not path.exists('production_data/nuclear.csv'):
        nuclear_prod = eq.instances.relative(
            '{} Nuclear Production MWh/h 15min Forecast'.format(region),
            begin=datetime(2019, 1, 1, 0, 0, 0),
            end=datetime(2021, 1, 1, 0, 0, 0),
            days_ahead=1,
            tag='',
            before_time_of_day=time(12, 0),
            frequency=Frequency.PT1H,
            aggregation=Aggregation.AVERAGE,
        )
        forecasted_nuclear = nuclear_prod.to_dataframe()

        nuclear_prod = eq.timeseries.load(
            '{} Nuclear Production MWh/h H Actual'.format(region),
            begin='2015-01-01',
            end='2019-01-01',
        )
        actual_nuclear = nuclear_prod.to_dataframe()

        forecasted_nuclear.fillna(method='ffill', inplace=True)
        forecasted_nuclear.rename(columns={'{} Nuclear Production MWh/h 15min Forecast'.format(region):'nuclear {}'.format(region)}, inplace=True)
        actual_nuclear.fillna(method='ffill', inplace=True)
        actual_nuclear.rename(columns={'{} Nuclear Production MWh/h H Actual'.format(region):'nuclear {}'.format(region)}, inplace=True)

        nuclear = pd.concat([actual_nuclear, forecasted_nuclear], axis=0, ignore_index=False)
        nuclear.index = nuclear.index.tz_localize(None)
        nuclear = nuclear[~nuclear.index.duplicated(keep='last')]
        nuclear.to_csv('production_data/nuclear.csv', index=True)
    else:
        nuclear  = pd.read_csv('production_data/nuclear.csv', encoding = "ISO-8859-1", sep=',', decimal='.', index_col=0, parse_dates=True)

    training_stop = datetime(2020, 1, 1, 00, 00, 00)

    nuclear_max = nuclear.loc[:training_stop].max()
    nuclear_min = nuclear.loc[:training_stop].min()
    
    nuclear = (nuclear - nuclear_min) / (nuclear_max - nuclear_min)
    return nuclear """

def get_cons_data(region):
    if not path.exists('consumption_data/cons.csv'):
        cons = eq.instances.relative(
        '{} Consumption MWh/h 15min Forecast'.format(region),
        begin=datetime(2019, 1, 1, 0, 0, 0),
        end=datetime(2021, 1, 1, 0, 0, 0),
        days_ahead=1,
        tag='ecsr',
        before_time_of_day=time(12, 0),
        frequency=Frequency.PT1H,
        aggregation=Aggregation.AVERAGE,
        )
        forecasted_cons = cons.to_dataframe()

        cons = eq.timeseries.load(
            '{} Consumption MWh/h H Actual'.format(region),
            begin='2015-01-01',
            end='2019-01-01',
        )
        actual_cons = cons.to_dataframe()

        forecasted_cons.fillna(method='ffill', inplace=True)
        forecasted_cons.rename(columns={'{} Consumption MWh/h 15min Forecast'.format(region):'cons {}'.format(region)}, inplace=True)
        actual_cons.fillna(method='ffill', inplace=True)
        actual_cons.rename(columns={'{} Consumption MWh/h H Actual'.format(region):'cons {}'.format(region)}, inplace=True)

        cons = pd.concat([actual_cons, forecasted_cons], axis=0, ignore_index=False)
        cons.index = cons.index.tz_localize(None)
        cons = cons[~cons.index.duplicated(keep='last')]
        cons.to_csv('consumption_data/cons.csv', index=True)
    else:
        cons  = pd.read_csv('consumption_data/cons.csv', encoding = "ISO-8859-1", sep=',', decimal='.', index_col=0, parse_dates=True)

    training_stop = datetime(2020, 1, 1, 00, 00, 00)
    cons_max = cons.loc[:training_stop].max()
    cons_min = cons.loc[:training_stop].min()
    
    cons = (cons - cons_min) / (cons_max - cons_min)
    return cons

def analyze_features(price_data, feature_data):
    reg = linear_model.LassoCV()
    features = feature_data.iloc[:,:6]
    price_spread = price_data['Dayahead SE3'] - price_data['Intraday SE3']
    
    x = features.loc[:date(2019,12,31)]
    y = price_spread.loc[:date(2019,12,31)]
    
    reg.fit(x.values, y.values)
    coef = pd.Series(reg.coef_, index = features.columns)
    print(coef)
    print(pd.concat([y,x], ignore_index=False, axis=1).corr())

def combine_weather_data():
    temp = get_eq_data('SE', 'Consumption Temperature °C', 'ecsr', 'weather_data/temp.csv', 'temp')
    return pd.concat([temp], axis=1, ignore_index=True)

if __name__ == "__main__":
    if not path.exists('feature_data.csv'):
        # Create seasonal data
        seasonal_data = seasonal_data()

        # Get flow data
        flow_data = get_flow_data()
        # Get forecasted capacity
        capacity_data = get_cap_data()
        # Get forecasted production data
        production_data = combine_production_data()
        # Get forecasted consumption data
        consumption_data = get_cons_data('SE')

        feature_data = pd.concat([production_data, consumption_data, capacity_data, seasonal_data, flow_data], ignore_index=False, axis=1)
        feature_data.rename(columns={ 
            feature_data.columns[0] : 'wp se', 
            feature_data.columns[1] : 'solar se',
            feature_data.columns[2] : 'nuclear se', 
            feature_data.columns[3] : 'cons se',
        }, inplace=True)
        feature_data.fillna(method='ffill', inplace=True)
        feature_data.to_csv('feature_data.csv', index=True)
    else:
        feature_data    = pd.read_csv('feature_data.csv', encoding = "ISO-8859-1", sep=',', decimal='.', index_col=0, parse_dates=True)
    
    if not path.exists('price_data.csv'):
        arrange_price_data()
        price_data = combine_yearly_price_data()
        price_data.to_csv('price_data.csv', index=True)
    else:
        price_data = pd.read_csv('price_data.csv', encoding = "ISO-8859-1", sep=',', decimal='.', index_col='Delivery', parse_dates=['Delivery'])

    analyze_features(price_data, feature_data)
    