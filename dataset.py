import torch
import torch.nn as nn
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import datetime
from PyEMD import EMD, Visualisation
from torch.utils.data import TensorDataset, DataLoader
from kmodes.KPrototypes import KPrototypes
import holidays

def create_datasets(file_path, model_name, seasonality = False, EMD = False):
    df = pd.read_csv(file_path, encoding = "ISO-8859-1", sep=',', decimal='.', index_col='Delivery', parse_dates=['Delivery'])
    
    df.fillna(0, inplace=True)
    #df.dropna(inplace=True)
    #df.fillna(method='ffill', inplace=True)
    df.replace(to_replace=0, method='ffill',inplace=True)

    if model_name == "MLP":
        X, Y = flat_data(df, 18, seasonality)
    elif model_name == "RNN":
        if EMD:
            X, Y = seq_emd_data(df, 6)
        else:
            X, Y = sequential_data(df, 6, seasonality)

    if EMD:
        training_generators = []
        validate_generators = []
        frac = round(len(X['imf 1'])*0.9)

        for key in X.keys():
            if key == 'all':
                dataset_train = TensorDataset(X[key][:frac], Y[:frac])
                dataset_val = TensorDataset(X[key][frac:], Y[frac:])

                training_generator = DataLoader(dataset_train, shuffle = True, batch_size = 32)
                validate_generator = DataLoader(dataset_val, shuffle = False, batch_size = 32)

                training_generators.append( training_generator )
                validate_generators.append( validate_generator )
            else:
                dataset_train = TensorDataset(X[key][:frac], Y[:frac])
                dataset_val = TensorDataset(X[key][frac:], Y[frac:])

                training_generator = DataLoader(dataset_train, shuffle = True, batch_size = 32)
                validate_generator = DataLoader(dataset_val, shuffle = False, batch_size = 32)

                training_generators.append( training_generator )
                validate_generators.append( validate_generator )

        return training_generators, validate_generators
    else:
        frac = round(len(X)*0.9)
        dataset_train = TensorDataset(X[:frac], Y[:frac])
        dataset_val = TensorDataset(X[frac:], Y[frac:])

        training_generator = DataLoader(dataset_train, shuffle = True, batch_size = 32)
        validate_generator = DataLoader(dataset_val, shuffle = False, batch_size = 32)

        return training_generator, validate_generator

def flat_data(df, lookback, seasonality):
    start_date = datetime.datetime(2015, 1, 2, 00, 00, 00)
    end_date = datetime.datetime(2021, 1, 1, 00, 00, 00)

    delta_day = datetime.timedelta(days=1)
    delta_twenty_three_hours = datetime.timedelta(hours=23)
    delta_lookback = datetime.timedelta(hours=lookback)
    delta_one_hour = datetime.timedelta(hours=1)

    se_holidays = holidays.CountryHoliday('SE')

    X = []
    Y = []

    while start_date < end_date:
        todays_data = df.loc[
            (start_date)
            : 
            (start_date+delta_twenty_three_hours)
        ]
        Y.append(
            todays_data['Dayahead SE3'].tolist() + 
            todays_data['Dayahead SE4'].tolist() + 
            todays_data['Intraday SE3'].tolist() + 
            todays_data['Intraday SE4'].tolist()
        )

        backward_data = df.loc[
            (start_date-delta_lookback)
            : 
            (start_date - delta_one_hour)
        ]
        

        X.append(
            backward_data['Dayahead SE3 (s)'].tolist() + 
            backward_data['Dayahead SE4 (s)'].tolist() + 
            backward_data['Intraday SE3 (s)'][:lookback-11].tolist() + 
            backward_data['Intraday SE4 (s)'][:lookback-11].tolist()   
        )

        if seasonality:
            is_holiday = 1 if start_date in se_holidays else 0
            month = [0]*12
            month[start_date.month-1] = 1
            day_of_week = [0]*7
            day_of_week[start_date.weekday()] = 1

            X[-1] += ([is_holiday] + month + day_of_week )

        start_date += delta_day

    X = torch.Tensor(X)
    Y = torch.Tensor(Y)

    return X, Y

def seq_emd_data(df, lookback):
    start_date = datetime.datetime(2015, 1, lookback+1, 00, 00, 00)
    end_date = datetime.datetime(2021, 1, 1, 00, 00, 00)

    delta_day = datetime.timedelta(days=1)
    delta_lookback = datetime.timedelta(days=lookback)
    delta_twenty_three_hours = datetime.timedelta(hours=23)
    delta_one_hour = datetime.timedelta(hours=1)

    se_holidays = holidays.CountryHoliday('SE')

    X = {
        'imf 1' : [],
        'imf 2' : [],
        'imf 3' : [],
        'imf 4' : [],
        'imf 5' : [],
        'imf 6' : [],
        'all'   : []
    }
    Y = []

    while start_date < end_date:
        todays_data = df.loc[
            (start_date)
            : 
            (start_date+delta_twenty_three_hours)
        ]
        Y.append(
            todays_data['Dayahead SE3'].tolist() + 
            todays_data['Dayahead SE4'].tolist() + 
            todays_data['Intraday SE3'].tolist() + 
            todays_data['Intraday SE4'].tolist()
        )

        backward_data = df.loc[
                (start_date-delta_lookback)
                : 
                (start_date-delta_one_hour)
            ]

        backward_data['Intraday SE3 (s)'].iloc[-11:] = 0
        backward_data['Intraday SE4 (s)'].iloc[-11:] = 0

        for key in X.keys():
            if key != 'all':
                x = np.vstack((
                    backward_data['Dayahead SE3 ({})'.format(key)],
                    backward_data['Dayahead SE4 ({})'.format(key)],
                    backward_data['Intraday SE3 ({})'.format(key)],
                    backward_data['Intraday SE4 ({})'.format(key)]
                ))
                X[key].append(x)
            else:
                x = np.vstack((
                    backward_data['Dayahead SE3 (imf 1)'],
                    backward_data['Dayahead SE4 (imf 1)'],
                    backward_data['Intraday SE3 (imf 1)'],
                    backward_data['Intraday SE4 (imf 1)'],

                    backward_data['Dayahead SE3 (imf 2)'],
                    backward_data['Dayahead SE4 (imf 2)'],
                    backward_data['Intraday SE3 (imf 2)'],
                    backward_data['Intraday SE4 (imf 2)'],

                    backward_data['Dayahead SE3 (imf 3)'],
                    backward_data['Dayahead SE4 (imf 3)'],
                    backward_data['Intraday SE3 (imf 3)'],
                    backward_data['Intraday SE4 (imf 3)'],

                    backward_data['Dayahead SE3 (imf 4)'],
                    backward_data['Dayahead SE4 (imf 4)'],
                    backward_data['Intraday SE3 (imf 4)'],
                    backward_data['Intraday SE4 (imf 4)'],

                    backward_data['Dayahead SE3 (imf 5)'],
                    backward_data['Dayahead SE4 (imf 5)'],
                    backward_data['Intraday SE3 (imf 5)'],
                    backward_data['Intraday SE4 (imf 5)'],

                    backward_data['Dayahead SE3 (imf 6)'],
                    backward_data['Dayahead SE4 (imf 6)'],
                    backward_data['Intraday SE3 (imf 6)'],
                    backward_data['Intraday SE4 (imf 6)'],
                ))
                X[key].append(x)

        start_date += delta_day
    
    for key in X.keys():
        X[key] = torch.transpose(torch.FloatTensor(X[key]), 1, 2)
    Y = torch.FloatTensor(Y)

    return X, Y
    
def sequential_data(df, lookback, seasonality):
    start_date = datetime.datetime(2015, 1, lookback+1, 00, 00, 00)
    end_date = datetime.datetime(2021, 1, 1, 00, 00, 00)

    delta_day = datetime.timedelta(days=1)
    delta_lookback = datetime.timedelta(days=lookback)
    delta_twenty_three_hours = datetime.timedelta(hours=23)
    delta_one_hour = datetime.timedelta(hours=1)

    se_holidays = holidays.CountryHoliday('SE')

    X = []
    Y = []

    while start_date < end_date:
        todays_data = df.loc[
            (start_date)
            : 
            (start_date+delta_twenty_three_hours)
        ]
        Y.append(
            todays_data['Dayahead SE3'].tolist() + 
            todays_data['Dayahead SE4'].tolist() + 
            todays_data['Intraday SE3'].tolist() + 
            todays_data['Intraday SE4'].tolist()
        )

        backward_data = df.loc[
                (start_date-delta_lookback)
                : 
                (start_date-delta_one_hour)
            ]

        

        backward_data['Intraday SE3 (s)'].iloc[-11:] = 0
        backward_data['Intraday SE4 (s)'].iloc[-11:] = 0

        x = np.vstack((
            backward_data['Dayahead SE3 (s)'],
            backward_data['Dayahead SE4 (s)'],
            backward_data['Intraday SE3 (s)'],
            backward_data['Intraday SE4 (s)']
            
        ))
        if seasonality:
            is_holiday = 1 if start_date in se_holidays else 0
            month = [0]*12
            month[start_date.month-1] = 1
            day_of_week = [0]*7
            day_of_week[start_date.weekday()] = 1
            x = np.vstack((
                x, 
                [is_holiday]*lookback*24,
                np.transpose(np.array([month]*lookback*24)),
                np.transpose(np.array([day_of_week]*lookback*24))
            ))

        X.append(x)
        start_date += delta_day
    
    X = torch.transpose(torch.FloatTensor(X), 1, 2)
    Y = torch.FloatTensor(Y)
    
    return X, Y

class Dataset():
    def __init__(self, price_file_path = 'data/price_data.csv', model_name="RNN", seasonality = True, lookback=2, cluster = True):
        self.price_file_path        = price_file_path
        self.model_name             = model_name
        self.seasonality   = seasonality
        self.lookback               = lookback
        self.price_data             = self.load_price_data()
        
        if model_name == "RNN":
            self.X, self.Y = self.sequential_data()
        if cluster:
            self.cluster()
        
        self.train_set, self.validate_set = self.create_generators()

    def load_price_data(self):
        df = pd.read_csv(self.price_file_path, encoding = "ISO-8859-1", sep=',', decimal='.', index_col='Delivery', parse_dates=['Delivery'])
        df.fillna(0, inplace=True)
        df.replace(to_replace=0, method='ffill',inplace=True)
        return df

    def sequential_data(self):
        start_date = datetime.datetime(2015, 1, self.lookback+1, 00, 00, 00)
        end_date = datetime.datetime(2021, 1, 1, 00, 00, 00)

        delta_day = datetime.timedelta(days=1)
        delta_lookback = datetime.timedelta(days=self.lookback)
        delta_twenty_three_hours = datetime.timedelta(hours=23)
        delta_one_hour = datetime.timedelta(hours=1)

        se_holidays = holidays.CountryHoliday('SE')

        X = []
        Y = []

        while start_date < end_date:
            todays_data = self.price_data.loc[
                (start_date)
                : 
                (start_date+delta_twenty_three_hours)
            ]
            Y.append(
                todays_data['Dayahead SE3'].tolist() + 
                todays_data['Dayahead SE4'].tolist() + 
                todays_data['Intraday SE3'].tolist() + 
                todays_data['Intraday SE4'].tolist()
            )

            backward_data = self.price_data.loc[
                    (start_date-delta_lookback)
                    : 
                    (start_date-delta_one_hour)
                ]

            

            backward_data['Intraday SE3 (s)'].iloc[-11:] = 0
            backward_data['Intraday SE4 (s)'].iloc[-11:] = 0

            x = np.vstack((
                backward_data['Dayahead SE3 (s)'],
                backward_data['Dayahead SE4 (s)'],
                backward_data['Intraday SE3 (s)'],
                backward_data['Intraday SE4 (s)']
                
            ))
            if self.seasonality:
                is_holiday = 1 if start_date in se_holidays else 0
                month = [0]*12
                month[start_date.month-1] = 1
                day_of_week = [0]*7
                day_of_week[start_date.weekday()] = 1
                x = np.vstack((
                    x, 
                    [is_holiday]*self.lookback*24,
                    np.transpose(np.array([month]*self.lookback*24)),
                    np.transpose(np.array([day_of_week]*self.lookback*24))
                ))

            X.append(x)
            start_date += delta_day
        
        X = torch.transpose(torch.FloatTensor(X), 1, 2)
        Y = torch.FloatTensor(Y)
        
        return X, Y

    def cluster(self):
        feature_data = pd.read_csv('data/feature_data.csv', encoding = "ISO-8859-1", sep=',', decimal='.', index_col='date_time', parse_dates=['date_time'])
        def create_clusters():
            clusters = KPrototypes().fit_predict(feature_data.to_numpy(), categorical=[0, 1 ,2])
            pass


    def split_into_clusters(self):
        pass

    def create_generators(self):
        frac = round(len(self.X)*0.9)
        dataset_train = TensorDataset(self.X[:frac], self.Y[:frac])
        dataset_val = TensorDataset(self.X[frac:], self.Y[frac:])

        training_generator = DataLoader(dataset_train, shuffle = True, batch_size = 32)
        validate_generator = DataLoader(dataset_val, shuffle = False, batch_size = 32)

        return training_generator, validate_generator