import torch
import torch.nn as nn
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import datetime
from PyEMD import EMD, Visualisation
from torch.utils.data import TensorDataset, DataLoader
from kmodes.kprototypes import KPrototypes
from sklearn.model_selection import train_test_split
import holidays

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

class Dataset():
    def __init__(self, price_file_path = 'data/price_data.csv', model_name="RNN", embedded_features = False, lookback=2, k=1):
        print("Loading dataset...")
        self.price_file_path        = price_file_path
        self.model_name             = model_name
        self.embedded_features      = embedded_features
        self.lookback               = lookback
        self.k                      = k
        self.price_data             = self.load_price_data()
        self.feature_data           = self.load_feature_data()
        self.date_list              = []
        
        if self.k > 1:
            self.create_cluster()
            
        if model_name == "RNN":
            self.X, self.Y = self.sequential_data() 
        elif model_name == "MLP":
            self.X, self.Y = self.flat_data()

        self.train_set, self.validate_set = self.create_generators()

    def load_price_data(self):
        df = pd.read_csv(self.price_file_path, encoding = "ISO-8859-1", sep=',', decimal='.', index_col='Delivery', parse_dates=['Delivery'])
        df.fillna(0, inplace=True)
        df.replace(to_replace=0, method='ffill',inplace=True)
        return df
    
    def load_feature_data(self):
        df = pd.read_csv('data/feature_data.csv', encoding = "ISO-8859-1", sep=',', decimal='.', index_col='date_time', parse_dates=['date_time'])
        df.fillna(method='ffill', inplace=True)
        return df

    def flat_data(self):
        start_date = datetime.datetime(2015, 1, 2, 00, 00, 00)
        end_date = datetime.datetime(2021, 1, 1, 00, 00, 00)

        delta_day = datetime.timedelta(days=1)
        delta_twenty_three_hours = datetime.timedelta(hours=23)
        delta_lookback = datetime.timedelta(hours=self.lookback)
        delta_one_hour = datetime.timedelta(hours=1)

        X = {}
        Y = {}
        for i in range(self.k):
            X[i] = []
            Y[i] = []

        while start_date < end_date:            
            cluster = self.feature_data.loc[start_date]['cluster'] if self.k > 1 else 0

            todays_data = self.price_data.loc[start_date : start_date+delta_twenty_three_hours]
            y = []

            Y[cluster].append(
                todays_data['Dayahead SE3'].tolist() + 
                todays_data['Dayahead SE4'].tolist() + 
                todays_data['Intraday SE3'].tolist() + 
                todays_data['Intraday SE4'].tolist()
            )

            backward_data = self.price_data.loc[start_date-delta_lookback : start_date - delta_one_hour]
            X[cluster].append(
                backward_data['Dayahead SE3 (s)'].tolist() + 
                backward_data['Dayahead SE4 (s)'].tolist() + 
                backward_data['Intraday SE3 (s)'][:self.lookback-11].tolist() + 
                backward_data['Intraday SE4 (s)'][:self.lookback-11].tolist()   
            )
            
            if self.embedded_features:
                todays_f_data = self.feature_data.loc[start_date]
                month         = self.create_one_hot( todays_f_data['month'], 12 )
                day_of_week   = self.create_one_hot( todays_f_data['day of week'], 7)
                weather       = todays_f_data.iloc[3:].tolist()

                f_data = [todays_f_data['holiday']] + month + day_of_week + weather
                X[cluster][-1] += f_data

            start_date += delta_day

        for key in X.keys():
            X[key] = torch.Tensor(X[key])
            Y[key] = torch.Tensor(Y[key])

        return X, Y
    
    def create_one_hot(self, i, size):
        array = [0]*size
        array[int(i)] = 1
        return array

    def sequential_data(self):
        start_date = datetime.datetime(2015, 1, self.lookback+1, 00, 00, 00)
        end_date = datetime.datetime(2021, 1, 1, 00, 00, 00)

        delta_day = datetime.timedelta(days=1)
        delta_lookback = datetime.timedelta(days=self.lookback)
        delta_twenty_three_hours = datetime.timedelta(hours=23)
        delta_one_hour = datetime.timedelta(hours=1)

        X = {}
        Y = {}
        for i in range(self.k):
            X[i] = []
            Y[i] = []

        while start_date < end_date:
            cluster = self.feature_data.loc[start_date]['cluster'] if self.k > 1 else 0

            todays_data = self.price_data.loc[start_date : start_date+delta_twenty_three_hours]
            Y[cluster].append(
                todays_data['Dayahead SE3'].tolist() + 
                todays_data['Dayahead SE4'].tolist() + 
                todays_data['Intraday SE3'].tolist() + 
                todays_data['Intraday SE4'].tolist()
            )

            backward_data = self.price_data.loc[start_date-delta_lookback : start_date-delta_one_hour]
            backward_data['Intraday SE3 (s)'].iloc[-11:] = 0
            backward_data['Intraday SE4 (s)'].iloc[-11:] = 0

            x = np.vstack((
                backward_data['Dayahead SE3 (s)'],
                backward_data['Dayahead SE4 (s)'],
                backward_data['Intraday SE3 (s)'],
                backward_data['Intraday SE4 (s)']
            ))

            if self.embedded_features:
                backward_f_data = self.feature_data.loc[start_date-delta_lookback : start_date-delta_one_hour]

                f_data = pd.DataFrame(np.repeat(backward_f_data.values,24,axis=0))
                f_data.columns = backward_f_data.columns

                month = np.stack(f_data['month'].apply(lambda m: self.create_one_hot(m, 12)), axis=1)
                day_of_week = np.stack(f_data['day of week'].apply(lambda d: self.create_one_hot(d, 7)), axis=1)

                x = np.vstack((
                    x,
                    f_data['holiday'],
                    month,
                    day_of_week,
                    np.transpose(f_data.iloc[:,3:])
                ))
                
            X[cluster].append(x)

            start_date += delta_day
        
        for key in X.keys():
            X[key] = torch.transpose(torch.FloatTensor(X[key]), 1, 2)
            Y[key] = torch.FloatTensor(Y[key])

        return X, Y

    def create_cluster(self):
        clusters = KPrototypes(n_clusters = self.k, verbose=1).fit_predict(self.feature_data.to_numpy(), categorical=[0, 1 ,2])
        self.feature_data['cluster'] = clusters

    def create_generators(self):
        training_generators = []
        validate_generators = []
        
        for k in self.X.keys():
            X_train, X_test, y_train, y_test = train_test_split(self.X[k], self.Y[k], test_size=0.1, shuffle=True)
            dataset_train = TensorDataset(X_train, y_train)
            dataset_val = TensorDataset(X_test, y_test)

            training_generator = DataLoader(dataset_train, shuffle = True, batch_size = 32)
            validate_generator = DataLoader(dataset_val, shuffle = False, batch_size = 32)

            training_generators.append(training_generator)
            validate_generators.append(validate_generator)

        return training_generators, validate_generators