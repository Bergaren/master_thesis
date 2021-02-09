import datetime
import requests
import random
import time
import pandas as pd
import os.path
from os import path

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


def arrange_data():
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

def combine_yearly_data():
    price_2015 = pd.read_csv('price_yearly/price_data_2015.csv', sep=',', decimal=".", index_col='Delivery', parse_dates=['Delivery'])
    price_2016 = pd.read_csv('price_yearly/price_data_2016.csv', sep=',', decimal=".", index_col='Delivery', parse_dates=['Delivery'])
    price_2017 = pd.read_csv('price_yearly/price_data_2017.csv', sep=',', decimal=".", index_col='Delivery', parse_dates=['Delivery'])
    price_2018 = pd.read_csv('price_yearly/price_data_2018.csv', sep=',', decimal=".", index_col='Delivery', parse_dates=['Delivery'])
    price_2019 = pd.read_csv('price_yearly/price_data_2019.csv', sep=',', decimal=".", index_col='Delivery', parse_dates=['Delivery'])
    price_2020 = pd.read_csv('price_yearly/price_data_2020.csv', sep=',', decimal=".", index_col='Delivery', parse_dates=['Delivery'])

    df = price_2015
    for df2 in [price_2016, price_2017, price_2018, price_2019, price_2020]:
        df = df.append(df2, ignore_index = False)

    return df

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

if __name__ == "__main__":
    arrange_data()
    price_data = combine_yearly_data()
    price_data.to_csv('price_data.csv', index=True)
