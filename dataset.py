import holidays
from sklearn.model_selection import train_test_split
from kmodes.kprototypes import KPrototypes
from torch.utils.data import TensorDataset, DataLoader, Dataset
import datetime
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None


class DataLoaderCreator():
    def __init__(self, price_file_path='data/price_data.csv', model_name="RNN", embedded_features=False, lookback=1, regression=True):
        print("Loading dataset...")
        self.price_file_path = price_file_path
        self.model_name = model_name
        self.embedded_features = embedded_features
        self.lookback = lookback
        self.regression = regression
        self.price_data = self.load_price_data()
        self.feature_data = self.load_feature_data()

        if model_name in ['LSTM', 'GRU', 'THH']:
            self.X, self.Y = self.sequential_data()
        elif model_name == "MLP":
            self.X, self.Y = self.flat_data()

        self.train_set, self.validate_set, self.test_set, self.final_train_set, self.final_validate_set = self.split_and_create_generators()

    def load_price_data(self):
        df = pd.read_csv(self.price_file_path, encoding="ISO-8859-1", sep=',',
                         decimal='.', index_col='Delivery', parse_dates=['Delivery'])
        return df

    def load_feature_data(self):
        df = pd.read_csv('data/feature_data.csv', encoding="ISO-8859-1",
                         sep=',', decimal='.', index_col=0, parse_dates=True)
        return df

    def create_one_hot(self, i, size):
        array = [0]*size
        array[int(i)] = 1
        return array

    def flat_data(self):
        delta_lookback = datetime.timedelta(days=self.lookback)
        start_date = datetime.datetime(2015, 1, 1, 00, 00, 00) + delta_lookback
        end_date = datetime.datetime(2021, 1, 1, 00, 00, 00)

        valid_start = datetime.datetime(2019, 1, 1)
        test_start = datetime.datetime(2020, 1, 1)

        delta_day = datetime.timedelta(days=1)
        delta_twenty_three_hours = datetime.timedelta(hours=23)
        delta_one_hour = datetime.timedelta(hours=1)

        X = {
            'train': [],
            'valid': [],
            'test': []
        }
        Y = {
            'train': [],
            'valid': [],
            'test': []
        }
        split = 'train'

        while start_date < end_date:
            if start_date >= valid_start:
                split = 'valid'
            if start_date >= test_start:
                split = 'test'

            todays_data = self.price_data.loc[start_date: start_date +
                                              delta_twenty_three_hours]

            y_spread = np.subtract(
                todays_data['Dayahead SE3'], todays_data['Intraday SE3'])
            if not self.regression:
                y_spread = np.sign(y_spread)
                y_spread = np.where(y_spread == -1, 0, y_spread)

            Y[split].append(y_spread)

            backward_data = self.price_data.loc[start_date -
                                                delta_lookback: start_date - delta_one_hour]

            x = (backward_data['Dayahead SE3 (s)'].tolist(
            ) + backward_data['Intraday SE3 (s)'][:self.lookback-11].tolist())

            if self.embedded_features:
                entire_day = self.feature_data.loc[start_date: start_date +
                                                   delta_twenty_three_hours]
                one_point = self.feature_data.loc[start_date]
                backward_f_data = self.feature_data.loc[start_date -
                                                        delta_lookback: start_date - delta_one_hour]

                prod = entire_day[['wp se', 'solar se',
                                   'nuclear se']].values.flatten().tolist()
                cons = entire_day[['cons se']].values.flatten().tolist()
                cap = entire_day[['cap SE2 > SE3', 'cap SE4 > SE3', 'cap NO1 > SE3',
                                  'cap DK1 > SE3', 'cap FI > SE3']].values.flatten().tolist()

                year = [start_date.year-2015]
                week = self.create_one_hot(one_point['week'], 53)
                day_of_week = self.create_one_hot(one_point['day of week'], 7)
                seasonal = [one_point['holiday']] + year + week + day_of_week

                flow = backward_f_data[['flow SE2 > SE3', 'flow SE4 > SE3', 'flow NO1 > SE3',
                                        'flow DK1 > SE3', 'flow FI > SE3']].values.flatten().tolist()

                x += prod + cons + cap + seasonal + flow

            X[split].append(x)

            start_date += delta_day

        for split in X.keys():
            X[split] = torch.Tensor(X[split])
            Y[split] = torch.Tensor(Y[split])

        return X, Y

    def sequential_data(self):
        delta_lookback = datetime.timedelta(days=self.lookback)
        start_date = datetime.datetime(2015, 1, 1, 00, 00, 00) + delta_lookback
        end_date = datetime.datetime(2021, 1, 1, 00, 00, 00)

        valid_start = datetime.datetime(2019, 1, 1)
        test_start = datetime.datetime(2020, 1, 1)

        delta_day = datetime.timedelta(days=1)
        delta_twenty_three_hours = datetime.timedelta(hours=23)
        delta_one_hour = datetime.timedelta(hours=1)

        X = {
            'train': [],
            'valid': [],
            'test': []
        }
        Y = {
            'train': [],
            'valid': [],
            'test': []
        }
        split = 'train'

        while start_date < end_date:
            if start_date >= valid_start:
                split = 'valid'
            if start_date >= test_start:
                split = 'test'

            todays_data = self.price_data.loc[start_date: start_date +
                                              delta_twenty_three_hours]

            y_spread = np.subtract(
                todays_data['Dayahead SE3'], todays_data['Intraday SE3'])
            if not self.regression:
                y_spread = np.sign(y_spread)
                y_spread = np.where(y_spread == -1, 0, y_spread)

            Y[split].append(y_spread)

            backward_data = self.price_data.loc[start_date -
                                                delta_lookback: start_date-delta_one_hour]
            backward_data['Intraday SE3 (s)'].iloc[-11:
                                                   ] = backward_data['Intraday SE3 (s)'].iloc[-12]

            x = np.vstack(
                (backward_data['Dayahead SE3 (s)'], backward_data['Intraday SE3 (s)']))

            if self.embedded_features:
                entire_day = self.feature_data.loc[start_date: start_date +
                                                   delta_twenty_three_hours]
                one_point = self.feature_data.loc[start_date]
                backward_f_data = self.feature_data.loc[start_date -
                                                        delta_lookback: start_date - delta_one_hour]

                prod = entire_day[['wp se', 'solar se',
                                   'nuclear se']].values.flatten()
                prod = np.tile(prod, (x.shape[1], 1))
                cons = entire_day[['cons se']].values.flatten()
                cons = np.tile(cons, (x.shape[1], 1))
                cap = entire_day[['cap SE2 > SE3', 'cap SE4 > SE3', 'cap NO1 > SE3',
                                  'cap DK1 > SE3', 'cap FI > SE3']].values.flatten()
                cap = np.tile(cap, (x.shape[1], 1))

                year = [start_date.year-2015]
                year = np.tile(year, (x.shape[1], 1))
                week = self.create_one_hot(one_point['week'], 53)
                week = np.tile(week, (x.shape[1], 1))
                day_of_week = self.create_one_hot(one_point['day of week'], 7)
                day_of_week = np.tile(day_of_week, (x.shape[1], 1))
                holiday = [one_point['holiday']]
                holiday = np.tile(holiday, (x.shape[1], 1))

                flow = np.vstack((backward_f_data['flow SE2 > SE3'], backward_f_data['flow SE4 > SE3'],
                                 backward_f_data['flow NO1 > SE3'], backward_f_data['flow DK1 > SE3'], backward_f_data['flow FI > SE3']))

                x = np.vstack((
                    x,
                    prod.transpose(),
                    cons.transpose(),
                    cap.transpose(),
                    year.T,
                    week.T,
                    day_of_week.T,
                    holiday.T,
                    flow
                ))

            X[split].append(x)

            start_date += delta_day

        for split in X.keys():
            X[split] = torch.transpose(torch.FloatTensor(X[split]), 1, 2)
            Y[split] = (torch.FloatTensor(Y[split]))

        return X, Y

    def split_and_create_generators(self):
        X = torch.cat((self.X['train'], self.X['valid']))
        y = torch.cat((self.Y['train'], self.Y['valid']))
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.1, shuffle=True)

        dataset_train = TensorDataset(self.X['train'], self.Y['train'])
        dataset_val = TensorDataset(self.X['valid'], self.Y['valid'])
        dataset_test = TensorDataset(self.X['test'], self.Y['test'])
        dataset_final_train = TensorDataset(X_train, y_train)
        dataset_final_val = TensorDataset(X_val, y_val)

        training_generator = DataLoader(
            dataset_train, shuffle=True, batch_size=32)
        val_generator = DataLoader(dataset_val, shuffle=False, batch_size=32)

        final_training_generator = DataLoader(
            dataset_final_train, shuffle=True, batch_size=32)
        final_val_generator = DataLoader(
            dataset_final_val, shuffle=False, batch_size=32)

        test_generator = DataLoader(dataset_test, shuffle=False, batch_size=32)

        return training_generator, val_generator, test_generator, final_training_generator, final_val_generator
