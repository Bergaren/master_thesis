import torch
import torch.nn as nn
from torch.autograd import Variable
from torchsummary import summary
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import time
import numpy as np
import sys
import optuna
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import holidays
from models import LSTM, GRU, MLP, EnsembleModel, Autoencoder, HybridModel
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from matplotlib.animation import FuncAnimation
from math import sqrt
from dataset import Dataset
from config import Config

device = "cuda" if torch.cuda.is_available() else "cpu"
config = Config()

def train(model, train_set, val_set, lr = 10**-2, epochs=20, trial=None, plot=False):
    print('\n Training... \n')
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    training_losses = []
    validation_losses = []
    for epoch in tqdm(range(epochs), desc='EPOCH'):
        y_pred = []
        y_true = []
        epoch_loss = []

        for i, batch in enumerate(tqdm(train_set, desc='Batch', disable=False)):
            optimizer.zero_grad()

            X, target = batch
            output = model(X)
            loss = criterion(output, target)
            loss.backward()
            epoch_loss.append(loss.item())

            #torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
        
        training_losses.append( sum(epoch_loss) / len(epoch_loss) )
        mse = validate(model, val_set)
        validation_losses.append(mse)

        if trial:
            trial.report(mse, epoch)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    if plot:
        plt.plot(training_losses, label='Training loss')
        plt.plot(validation_losses, label='Validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('Mean square error')
        plt.legend()
        plt.show()

    return mse

def validate(model, dataset):
    verbose = False
    if verbose:
        print("\n Validating... \n")

    model.eval()
    criterion = torch.nn.MSELoss()

    losses = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataset, desc='Batch', disable=not verbose)):
            X, target = batch

            output = model(X)    

            loss = criterion(output, target)
            losses.append(loss.item())
    
    return np.average(losses)

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

def objective(trial, model, train_set, val_set):
    model.define_model(trial)
    model = model.to(device)

    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    mse = train(model, train_set, val_set, lr=lr, epochs = 30, trial=trial)

    return mse

def hyper_parameter_selection(model, train_set, val_set):
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, model, train_set, val_set), n_trials=100)

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    return trial.params

if __name__ == "__main__":
    models = []

    for i in range(config.n_modes if config.ensemmble else 1):
        if config.model == "MLP":
            model = MLP(input_size=config.input_size, output_size=config.output_size, mode=0)
        elif config.model == "LSTM":
            model = LSTM(input_size=config.input_size, output_size=config.output_size, mode=0)
        elif config.model == "GRU":
            model = GRU(input_size=config.input_size, output_size=config.output_size, mode=0)
        models.append(model)
        
    if config.mode_decomp and config.ensemmble:
        model = EnsembleModel(models, n_features=config.input_size, output_size=config.output_size)
        models = [model]

    Dataset = Dataset(model_name=config.model, lookback=config.lookback, k=config.k, embedded_features=config.embedded_features, mode_decomp=config.mode_decomp)

    for i, model in enumerate(models):
        hyper_params = hyper_parameter_selection(model, Dataset.train_set[i], Dataset.validate_set[i])        
        train(model, Dataset.train_set[i], Dataset.validate_set[i], lr=hyper_params['lr'], epochs=40, plot=True)
    
    test(models, Dataset.test_set)