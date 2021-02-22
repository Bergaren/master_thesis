import torch
import torch.nn as nn
from torch.autograd import Variable
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import time
import numpy as np
import sys
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import holidays
from models import RNN, MLP, EnsembleModel
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from matplotlib.animation import FuncAnimation
from math import sqrt
from dataset import Dataset

device = "cuda" if torch.cuda.is_available() else "cpu"

def train(model, dataset, lr = 10**-2, epochs = 50):
    print('\n Training... \n')
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    losses = []
    for epoch in tqdm(range(epochs), desc='EPOCH'):
        y_pred = []
        y_true = []
        epoch_loss = []

        for i, batch in enumerate(tqdm(dataset,desc='Batch')):
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

    """ plt.plot(losses)
    plt.title('Loss')
    plt.ylabel('MSE')
    plt.xlabel('Epoch')
    plt.show() """

    return model

def validate(model, dataset):
    print("\n Validating... \n")
    model.eval()

    criterion = torch.nn.L1Loss()

    y_pred = {
        'Dayahead SE3' : [],
        'Dayahead SE4' : [],
        'Intraday SE3' : [],
        'Intraday SE4' : []
    }
    y_true = {
        'Dayahead SE3' : [],
        'Dayahead SE4' : [],
        'Intraday SE3' : [],
        'Intraday SE4' : []
    }
    losses = []
    for i, batch in enumerate(tqdm(dataset,desc='Batch')):
        X, target = batch

        output = model(X)    
        if type(model).__name__ == "RNN":
            output = output[:,-1]

        loss = criterion(output, target)
        losses.append(loss.item())

        i = 1
        for column in y_pred.keys():
            y_pred[column].extend(output[:,24*(i-1):24*i].flatten().tolist())
            y_true[column].extend(target[:,24*(i-1):24*i].flatten().tolist())
            i += 1

    print("L1:        {:.2f}".format( np.mean(losses) ))
    
    d = { 
        'Market': y_pred.keys(), 
        'Average Actual Price': [np.mean(v) for v in y_true.values()],
        'Std (actual price)': [np.std(v) for v in y_true.values()],
        'Average Predicted Price': [np.mean(v) for v in y_pred.values()],
        'Std (pred price)': [np.std(v) for v in y_pred.values()],
        'L1' : [mean_absolute_error(y_true[k],y_pred[k]) for k in y_true.keys()],
        'L1 (%)' : [mean_absolute_percentage_error(y_true[k],y_pred[k]) for k in y_true.keys()]
    }
    print(pd.DataFrame(data=d))

def test(models, datasets):
    print("\n Testing... \n")
    criterion = torch.nn.L1Loss()
    y_pred = {
        'Dayahead SE3' : [],
        'Dayahead SE4' : [],
        'Intraday SE3' : [],
        'Intraday SE4' : []
    }
    y_true = {
        'Dayahead SE3' : [],
        'Dayahead SE4' : [],
        'Intraday SE3' : [],
        'Intraday SE4' : []
    }
    losses = []
    for i, model in enumerate(models):
        for i, batch in enumerate(tqdm(datasets[i],desc='Batch')):
            X, target = batch

            output = model(X)    
            if type(model).__name__ == "RNN":
                output = output[:,-1]

            loss = criterion(output, target)
            losses.append(loss.item())

            i = 1
            for column in y_pred.keys():
                y_pred[column].extend(output[:,24*(i-1):24*i].flatten().tolist())
                y_true[column].extend(target[:,24*(i-1):24*i].flatten().tolist())
                i += 1
    
    print("L1:        {:.2f}".format( np.mean(losses) ))
    
    d = { 
        'Market': y_pred.keys(), 
        'Average Actual Price': [np.mean(v) for v in y_true.values()],
        'Std (actual price)': [np.std(v) for v in y_true.values()],
        'Average Predicted Price': [np.mean(v) for v in y_pred.values()],
        'Std (pred price)': [np.std(v) for v in y_pred.values()],
        'L1' : [mean_absolute_error(y_true[k],y_pred[k]) for k in y_true.keys()],
        'L1 (%)' : [mean_absolute_percentage_error(y_true[k],y_pred[k]) for k in y_true.keys()]
    }
    print(pd.DataFrame(data=d))
    fig1, axs = plt.subplots(2, 1)
    colors = sns.color_palette("tab10")

    delta_pred = np.subtract(y_pred['Dayahead SE4'], y_pred['Intraday SE4'])
    delta_true = np.subtract(y_true['Dayahead SE4'], y_true['Intraday SE4'])

    d_std = np.std(delta_pred)
    d_mean = np.mean(delta_pred)

    trade = []
    for s_pred, s_true in zip(delta_pred, delta_true):
        if np.abs(s_pred) > (1*d_std + np.abs(d_mean)):
            trade.append( np.sign(s_pred) == np.sign(s_true) )

    all_trade_acc = np.sum(np.sign(delta_pred) == np.sign(delta_true)) / len(delta_pred)
    trade_rule_acc = np.sum(trade) / len(trade)

    print('\n             Spread Acc    N')
    print('All:         {:.3f}        {}'.format(all_trade_acc, len(delta_pred)))
    print('>std:        {:.3f}        {}'.format(trade_rule_acc, len(trade)))

if __name__ == "__main__":
    args = sys.argv

    if len(args) != 2:
        arg = int(input(
            "Choice a model please."
            "Argument:    Price:    Feature Repr:     Model:  \n"
            "(0)        - Raw       None              MLP     \n"
            "(1)        - Raw       None              RNN     \n"
            "(2)        - Raw       Emb. Features     MLP     \n"
            "(3)        - Raw       Emb. Features     RNN     \n"
            "(4)        - Raw       Clustering        MLP     \n"
            "(5)        - Raw       Clustering        RNN     \n"
            "(6)        - Raw       Autoencoder       MLP     \n"
            "(7)        - Raw       Autoencoder       RNN     \n"
            "(8)        - VMD       None              MLP     \n"
            "(9)        - VMD       None              RNN     \n"
            "(10)       - VMD       Emb. Features     MLP     \n"
            "(11)       - VMD       Emb. Features     RNN     \n"
            "(12)       - VMD       Clustering        MLP     \n"
            "(13)       - VMD       Clustering        RNN     \n"
        ))
    else:
        arg = int(args[1])

    lookback = 24
    input_size = lookback*2 + (24-11)*2
    k = 1
    models = []
    embedded_features = False
    
    if arg == 0:
        models.append(MLP(input_size=input_size, output_size=24*4, hidden_size=64))
    elif arg == 1:
        lookback = 2
        models.append(RNN(input_size=4, num_layers=1, output_size=24*4, hidden_size=64))
    elif arg in range(2,4):
        seasonal_len = 20
        weather_len = 24
        embedded_features = True
        if arg == 2:
            models.append(MLP(input_size=input_size+seasonal_len+weather_len, output_size=24*4, hidden_size=64))
        elif arg == 3:
            lookback = 2
            models.append(RNN(input_size=4+seasonal_len+weather_len, num_layers=1, output_size=24*4, hidden_size=64))
    elif arg in range(4, 6):
        k = 4
        if arg == 4:
            for i in range(k):
                models.append(MLP(input_size=input_size, output_size=24*4, hidden_size=64))
        elif arg == 5:
            lookback = 2
            for i in range(k):
                models.append(RNN(input_size=4, num_layers=1, output_size=24*4, hidden_size=64))

    Dataset = Dataset(model_name=type(models[0]).__name__, lookback=lookback, k=k, embedded_features=embedded_features)

    for i, model in enumerate(models):
        train(model, Dataset.train_set[i], epochs=20)
        validate(model, Dataset.validate_set[i])

    test(models, Dataset.validate_set)