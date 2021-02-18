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
from math import sqrt
from dataset import create_datasets, Dataset

device = "cuda" if torch.cuda.is_available() else "cpu"

def train(model, dataset, lr = 10**-2, epochs = 50):
    print('\n Training... \n')
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    losses = []
    for epoch in tqdm(range(epochs), desc='Epoch'):
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
            y_pred[column].extend(output[:,:24*i].flatten().tolist())
            i += 1

        i = 1
        for column in y_true.keys():
            y_true[column].extend(target[:,:24*i].flatten().tolist())
            i += 1

    d = 30
    plt.plot(y_pred['Dayahead SE3'][24*d:24*(d+1)+1], label='Pred Dayahead')
    plt.plot(y_true['Dayahead SE3'][24*d:24*(d+1)+1], label='True Dayahead')
    plt.plot(y_pred['Intraday SE3'][24*d:24*(d+1)+1], label='Pred Intraday')
    plt.plot(y_true['Intraday SE3'][24*d:24*(d+1)+1], label='True Intraday')
    plt.title('Predicted vs actual price')
    plt.ylabel('€/MWh')
    plt.xlabel('Hours')
    plt.legend()

    print("All L1:        {:.2f}".format( np.mean(losses) ))
    #print(mean_absolute_error(y_true,y_pred))
    
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
    print("STD:           {:.2f}".format( np.std(y_true['Dayahead SE3']) ))

    """ plt.show() """

if __name__ == "__main__":
    args = sys.argv

    if len(args) != 4:
        raise Exception( (
            "Please specify 2 arguments.\n"
            "Argument 1: (0) - No Seasonality   | (1) - Seasonality\n"
            "Argument 2: (0) - MLP              | (1) - NLP\n"
            "Argument 3: (0) - Raw Price        | (1) - EMD         | (2) - VMD\n"
            ))
        
    n_features = 0
    if int(args[1]) == 0:
        seasonality = False
        print("(0) - No Seasonality")
    elif int(args[1]) == 1:
        n_features += 20
        seasonality = True
        print("(1) - Seasonality")

    if int(args[2]) == 0:
        print("(0) - MLP")
        model = MLP(input_size=50+n_features, output_size=24*4, hidden_size=64)
        model_name = type(model).__name__
    elif int(args[2]) == 1:
        print("(1) - LSTM")
        if int(args[3]) == 0:
            print("(0) - Raw price")
            model = RNN(input_size=4+n_features, num_layers=1, output_size=24*4, hidden_size=64)
            model.to(device)
        elif int(args[3]) == 1:
            print("(1) - EMD")
            models = []
            for i in range(1,5):    
                model = RNN(input_size=4, num_layers=4, output_size=24*4, hidden_size=64)
                model.to(device)
                models.append(model)
            #model = RNN(input_size=4*6, num_layers=1, output_size=24*4, hidden_size=64)
        
        model_name = "RNN"
    
    Dataset = Dataset(seasonality=seasonality)
    #train_set, validate_set = create_datasets(file_path = 'data/price_data.csv', model_name = model_name, seasonality = seasonality, EMD=False )    

    if int(args[3]) == 1:
        for i, model in enumerate(models):
            print('Training model: {}'.format(i+1))
            models[i] = train(model, Dataset.train_set[i], epochs=20)
            validate(model, Dataset.validate_set[i])
        
        ensemble_model = EnsembleModel(models, n_features=4)
        #model = train(model, train_set[-1], epochs=20)
        validate(ensemble_model, Dataset.validate_set[-1])
    else:
        model = train(model, Dataset.train_set, epochs=20)
        validate(model, Dataset.validate_set)
        
        
