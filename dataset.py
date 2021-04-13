import torch
import torch.nn as nn
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import datetime
from torch.utils.data import TensorDataset, DataLoader
from kmodes.kprototypes import KPrototypes
from sklearn.model_selection import train_test_split
import holidays

class Dataset():
    def __init__(self, price_file_path = 'data/price_data_2.csv', model_name="RNN", embedded_features = False, mode_decomp = False, lookback=2, k=1):
        print("Loading dataset...")
        self.price_file_path        = price_file_path
        self.model_name             = model_name
        self.embedded_features      = embedded_features
        self.lookback               = lookback
        self.k                      = k
        self.mode_decomp            = mode_decomp
        self.price_data             = self.load_price_data()
        self.feature_data           = self.load_feature_data()

        if self.k > 1:
            self.create_cluster()
            
        if model_name == "LSTM" or model_name == "GRU":
            self.X, self.Y = self.sequential_data() 
        elif model_name == "MLP":
            self.X, self.Y = self.flat_data()

        self.train_set, self.validate_set, self.test_set = self.split_and_create_generators()

    def load_price_data(self):
        df = pd.read_csv(self.price_file_path, encoding = "ISO-8859-1", sep=',', decimal='.', index_col='Delivery', parse_dates=['Delivery'])
        return df
    
    def load_feature_data(self):
        df = pd.read_csv('data/feature_data.csv', encoding = "ISO-8859-1", sep=',', decimal='.', index_col=0, parse_dates=True)
        return df

    def flat_data(self):
        start_date = datetime.datetime(2015, 1, 2, 00, 00, 00)
        end_date = datetime.datetime(2021, 1, 1, 00, 00, 00)

        valid_start = datetime.datetime(2019, 1, 1)
        test_start = datetime.datetime(2020, 1, 1)

        delta_day = datetime.timedelta(days=1)
        delta_twenty_three_hours = datetime.timedelta(hours=23)
        delta_lookback = datetime.timedelta(hours=self.lookback)
        delta_one_hour = datetime.timedelta(hours=1)

        X = {
            'train' : [[] for _ in range(self.k)],
            'valid' : [[] for _ in range(self.k)],
            'test'  : [[] for _ in range(self.k)]
        }
        Y = {
            'train' : [[] for _ in range(self.k)],
            'valid' : [[] for _ in range(self.k)],
            'test'  : [[] for _ in range(self.k)]
        }
        split = 'train'

        while start_date < end_date:
            if start_date >= valid_start:
                split = 'valid'
            if start_date >= test_start:
                split = 'test'

            cluster = int(self.feature_data.loc[start_date]['cluster']) if self.k > 1 else 0

            todays_data = self.price_data.loc[start_date : start_date+delta_twenty_three_hours]
            
            Y[split][cluster].append( np.subtract(todays_data['Dayahead SE3'], todays_data['Intraday SE3']) )

            backward_data = self.price_data.loc[start_date-delta_lookback : start_date - delta_one_hour]
            
            x = ( backward_data['Dayahead SE3 (s)'].tolist() + backward_data['Intraday SE3 (s)'][:self.lookback-11].tolist() )
            
            if self.mode_decomp:
                for n in range(1,7):
                    x += (
                        backward_data['Dayahead SE3 (imf {})'.format(n)].tolist() +
                        backward_data['Intraday SE3 (imf {})'.format(n)][:self.lookback-11].tolist()
                    )
            
            if self.embedded_features:
                todays_f_data   = self.feature_data.loc[start_date : start_date+delta_twenty_three_hours]
                cap             = todays_f_data.iloc[:,:-3].values.flatten().tolist()

                todays_f_data = self.feature_data.loc[start_date]
                month         = self.create_one_hot( todays_f_data['month'], 12 )
                day_of_week   = self.create_one_hot( todays_f_data['day of week'], 7)
                seasonal      = [todays_f_data['holiday']] + month + day_of_week

                x += cap + seasonal
                
            X[split][cluster].append(x)

            start_date += delta_day

        for split in X.keys():
            for k in range(self.k):
                X[split][k] = torch.Tensor(X[split][k])
                Y[split][k] = torch.Tensor(Y[split][k])
        
        return X, Y
    
    def create_one_hot(self, i, size):
        array = [0]*size
        array[int(i)] = 1
        return array

    def sequential_data(self):
        start_date = datetime.datetime(2015, 1, self.lookback+1, 00, 00, 00)
        end_date = datetime.datetime(2021, 1, 1, 00, 00, 00)

        valid_start = datetime.datetime(2019, 1, 1)
        test_start = datetime.datetime(2020, 1, 1)

        delta_day = datetime.timedelta(days=1)
        delta_lookback = datetime.timedelta(days=self.lookback)
        delta_twenty_three_hours = datetime.timedelta(hours=23)
        delta_one_hour = datetime.timedelta(hours=1)

        X = {
            'train' : [[] for _ in range(self.k)],
            'valid' : [[] for _ in range(self.k)],
            'test'  : [[] for _ in range(self.k)]
        }
        Y = {
            'train' : [[] for _ in range(self.k)],
            'valid' : [[] for _ in range(self.k)],
            'test'  : [[] for _ in range(self.k)]
        }
        split = 'train'

        while start_date < end_date:
            if start_date >= valid_start:
                split = 'valid'
            if start_date >= test_start:
                split = 'test'
            
            cluster = self.feature_data.loc[start_date]['cluster'] if self.k > 1 else 0

            todays_data = self.price_data.loc[start_date : start_date+delta_twenty_three_hours]
            Y[split][cluster].append( np.subtract(todays_data['Dayahead SE3'], todays_data['Intraday SE3']) )

            backward_data = self.price_data.loc[start_date-delta_lookback : start_date-delta_one_hour]
            backward_data['Intraday SE3 (s)'].iloc[-11:] = backward_data['Intraday SE3 (s)'].iloc[-12]

            x = np.vstack((backward_data['Dayahead SE3 (s)'], backward_data['Intraday SE3 (s)']))

            if self.mode_decomp:
                for n in range(1,7):
                    backward_data['Intraday SE3 (imf {})'.format(n)].iloc[-11:] = backward_data['Intraday SE3 (imf {})'.format(n)].iloc[-12]
                    backward_data['Intraday SE4 (imf {})'.format(n)].iloc[-11:] = backward_data['Intraday SE4 (imf {})'.format(n)].iloc[-12]

                    x = np.vstack((
                        x,
                        backward_data['Dayahead SE3 (imf {})'.format(n)].tolist(),
                        backward_data['Dayahead SE4 (imf {})'.format(n)].tolist(),
                        backward_data['Intraday SE3 (imf {})'.format(n)].tolist(),
                        backward_data['Intraday SE4 (imf {})'.format(n)].tolist()
                    ))                

            if self.embedded_features:
                f_data   = self.feature_data.loc[start_date : start_date+delta_twenty_three_hours]
                cap      = f_data.iloc[:,:-3].values.flatten()
                cap      = np.tile(cap, (x.shape[1],1))

                todays_f_data = self.feature_data.loc[start_date]
                
                backward_f_data = self.feature_data.loc[start_date-delta_lookback : start_date-delta_one_hour]

                #f_data.columns = backward_f_data.columns

                month = np.stack(f_data['month'].apply(lambda m: self.create_one_hot(m, 12)), axis=1)
                day_of_week = np.stack(f_data['day of week'].apply(lambda d: self.create_one_hot(d, 7)), axis=1)
                x = np.vstack((
                    x,
                    cap.transpose()
                ))
                
            X[split][cluster].append(x)

            start_date += delta_day
        
        for split in X.keys():
            for k in range(self.k):
                X[split][k] = torch.transpose(torch.FloatTensor(X[split][k]), 1, 2)
                Y[split][k] = (torch.FloatTensor(Y[split][k]))

        return X, Y

    def create_cluster(self):
        clusters = KPrototypes(n_clusters = self.k, verbose=1).fit_predict(self.feature_data.to_numpy(), categorical=[0, 1 ,2])
        self.feature_data['cluster'] = clusters

    def split_and_create_generators(self):
        training_generators = []
        validate_generators = []
        test_generators     = []
        
        """ test_start = datetime.datetime(2020, 1, 1)
        test_stop = datetime.datetime(2021, 1, 1)
        test_len = (test_stop - test_start).days """
        
        
        for k in range(self.k):
            #X_train, X_test, y_train, y_test = X[:-test_len], X[-test_len:], Y[:-test_len], Y[-test_len:]
            #X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1)

            dataset_train = TensorDataset(self.X['train'][k], self.Y['train'][k])
            dataset_val = TensorDataset(self.X['valid'][k], self.Y['valid'][k])
            dataset_test = TensorDataset(self.X['test'][k], self.Y['test'][k])

            training_generator = DataLoader(dataset_train, shuffle = True, batch_size = 32)
            val_generator = DataLoader(dataset_val, shuffle = True, batch_size = 32)
            test_generator = DataLoader(dataset_test, shuffle = False, batch_size = 32)

            training_generators.append(training_generator)
            validate_generators.append(val_generator)
            test_generators.append(test_generator)

        return training_generators, validate_generators, test_generators