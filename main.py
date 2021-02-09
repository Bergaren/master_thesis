import torch
import torch.nn as nn
from torch.autograd import Variable
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import time
import datetime
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import holidays
from models import RNN, MLP

def train(model, dataset, lr = 10**-3, epochs = 50):
    print('Training...')
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    losses = []
    for epoch in tqdm(range(epochs), desc='Epoch'):
        y_pred = []
        y_true = []
        epoch_loss = []

        for i, batch in enumerate(dataset):
            optimizer.zero_grad()

            X, target = batch
            output = model(X)
            
            if type(model).__name__ == "RNN":
                output = output[:,-1]

            loss = criterion(output, target)
            loss.backward()
            epoch_loss.append(loss.item())

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
        
        losses.append( sum(epoch_loss) / len(epoch_loss) )

    plt.plot(losses)
    plt.title('Loss')
    plt.ylabel('MSE')
    plt.xlabel('Epoch')
    plt.show()

    return model

def validate(model, dataset):
    model.eval()

    criterion = torch.nn.L1Loss()

    y_pred = []
    y_true = []
    losses = []
    for i, batch in enumerate(dataset):
        X, target = batch
        output = model(X)    
        if type(model).__name__ == "RNN":
            output = output[:,-1]

        loss = criterion(output, target)
        losses.append(loss.item())

        y_pred.extend(output.flatten().tolist())
        y_true.extend(target.flatten().tolist())

    plt.plot(y_pred[:25], label='Prediction')
    plt.plot(y_true[:25], label='Actual')
    plt.title('Predicted vs actual price')
    plt.ylabel('€/MWh')
    plt.xlabel('Hours')
    plt.legend()

    print("Average price: {:.2f} €/MWh".format( np.mean(y_true) ))
    print("STD:           {:.2f}".format( np.std(y_true) ))
    print("L1:            {:.2f}".format( np.mean(losses) ))

    plt.show()

def load_dataset( model_name ):
    print('Loading data...')
    
    df = pd.read_csv('data/price_data.csv', encoding = "ISO-8859-1", sep=',', decimal='.', index_col='Delivery', parse_dates=['Delivery'])
    
    df = standard_score(df)
    df.fillna(0, inplace=True)
    if model_name == "MLP":
        X, Y = flat_data(df, 18)
    elif model_name == "RNN":
        X, Y = sequential_data(df, 2)
        print(X.shape)
        X = torch.transpose(X, 1, 2)

    dataset_train = TensorDataset(X, Y)        
    training_generator = DataLoader(dataset_train, shuffle = True, batch_size = 32)
    validate_generator = DataLoader(dataset_train, shuffle = False, batch_size = 32)

    return training_generator, validate_generator

def flat_data(df, lookback):
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
        is_holiday = 1 if start_date in se_holidays else 0
        month = [0]*12
        month[start_date.month-1] = 1
        day_of_week = [0]*7
        day_of_week[start_date.weekday()] = 1

        X.append(
            backward_data['Dayahead SE3 (s)'].tolist() + 
            backward_data['Dayahead SE4 (s)'].tolist() + 
            backward_data['Intraday SE3 (s)'][:lookback-11].tolist() + 
            backward_data['Intraday SE4 (s)'][:lookback-11].tolist() +
            [is_holiday] +
            month + 
            day_of_week
        )
        start_date += delta_day

    return torch.Tensor(X), torch.Tensor(Y)

def sequential_data(df, lookback):
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

        is_holiday = 1 if start_date in se_holidays else 0
        month = [0]*12
        month[start_date.month-1] = 1
        day_of_week = [0]*7
        day_of_week[start_date.weekday()] = 1

        backward_data['Intraday SE3 (s)'].iloc[-11:] = 0
        backward_data['Intraday SE4 (s)'].iloc[-11:] = 0

        x = np.vstack((
            backward_data['Dayahead SE3 (s)'],
            backward_data['Dayahead SE4 (s)'],
            backward_data['Intraday SE3 (s)'],
            backward_data['Intraday SE4 (s)'],
            [is_holiday]*lookback*24,
            np.transpose(np.array([month]*lookback*24)),
            np.transpose(np.array([day_of_week]*lookback*24))
        ))
        X.append(x)
        start_date += delta_day

    return torch.FloatTensor(X), torch.FloatTensor(Y)

def standard_score(df):
    for column in ['Dayahead SE3', 'Dayahead SE4', 'Intraday SE3', 'Intraday SE4']:
        std = df[column].std()
        mean = df[column].mean()

        df['{} (s)'.format(column)] = (df[column] - mean) / std
    return df

if __name__ == "__main__":
    print("(1) - MLP, (2) - LSTM")
    m_choice = int(input('Choose model: '))
    if (m_choice == 1):
        model = MLP(input_size=70, output_size=24*4, hidden_size=64)
    elif (m_choice == 2):
        model = RNN(input_size=24, num_layers=1, output_size=24*4, hidden_size=64)

    train_set, validate_set = load_dataset( model_name = type(model).__name__ )    

    model = train(model, train_set, epochs=50)
    validate(model, validate_set)