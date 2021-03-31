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
import optuna
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import holidays
from models import LSTM, GRU, MLP, EnsembleModel, Autoencoder, HybridModel
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, accuracy_score, mean_squared_error, classification_report, confusion_matrix
from matplotlib.animation import FuncAnimation
from math import sqrt
from dataset import Dataset
from config import Config

device = "cuda" if torch.cuda.is_available() else "cpu"
config = Config()

def train(model, train_set, val_set, lr = 10**-2, epochs=30, max_norm=2, weight_decay=0, trial=None, plot=False):
    print('\n Training... \n')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.MSELoss()

    training_losses = []
    validation_losses = []
    for epoch in tqdm(range(epochs), desc='EPOCH'):
        model.train()
        y_pred = []
        y_true = []
        epoch_loss = []

        for i, batch in enumerate(tqdm(train_set, desc='Batch', disable=False)):
            optimizer.zero_grad()

            X, target = batch[0].to(device), batch[1].to(device)
            output = model(X)
            loss = criterion(output, target)
            loss.backward()
            epoch_loss.append(loss.item())

            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
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
            X, target = batch[0].to(device), batch[1].to(device)

            output = model(X)    

            loss = criterion(output, target)
            losses.append(loss.item())
    
    return np.average(losses)

def test(models, datasets):
    print("\n Testing... \n")
    criterion = torch.nn.L1Loss()
    """ y_pred = {
        'Dayahead SE3' : [],
        'Intraday SE3' : []
    }
    y_true = {
        'Dayahead SE3' : [],
        'Intraday SE3' : []
    } """
    
    y_pred = []
    y_true = []
    losses = []
    for i, model in enumerate(models):
        model.eval()

        with torch.no_grad():
            for i, batch in enumerate(tqdm(datasets[i],desc='Batch')):
                X, target = batch[0].to(device), batch[1].to(device)

                output = model(X)    

                loss = criterion(output, target)
                losses.append(loss.item())

                y_pred.extend(output.flatten().tolist())
                y_true.extend(target.flatten().tolist())

                """ for i, column in enumerate(y_pred.keys()):
                    y_pred[column].extend(output[:,24*i:24*(i+1)].flatten().tolist())
                    y_true[column].extend(target[:,24*i:24*(i+1)].flatten().tolist())
 """
    dir_pred = np.sign(y_pred)
    dir_true = np.sign(y_true)
    #delta_pred = np.subtract(y_pred['Dayahead SE3'], y_pred['Intraday SE3'])
    #delta_true = np.subtract(y_true['Dayahead SE3'], y_true['Intraday SE3'])

    #dir_pred = np.sign(delta_pred)
    #dir_true = np.sign(delta_true)

    print(accuracy_score(dir_true, dir_pred))
    """ print("\n ### Price prediction ### \n")
    print("\n # Summary statistics #")
    summary_statistics = {
        'Market' : y_pred.keys(),
        'Market': y_pred.keys(), 
        'Mean True Price': [np.mean(v) for v in y_true.values()],
        'Std (true price)': [np.std(v) for v in y_true.values()],
        'Mean Predicted Price': [np.mean(v) for v in y_pred.values()],
        'Std (pred price)': [np.std(v) for v in y_pred.values()]
    }
    print(pd.DataFrame(data=summary_statistics))

    print("\n # Error #")
    print("MAE:        {:.2f}".format( np.mean(losses) ))
    error = { 
        'Market': y_pred.keys(), 
        'MAE' : [mean_absolute_error(y_true[k],y_pred[k]) for k in y_true.keys()],
        'MSE' : [mean_squared_error(y_true[k],y_pred[k]) for k in y_true.keys()],
    }
    print(pd.DataFrame(data=error))

    print("\n ### Price-spread prediction ###")
    delta_pred = np.subtract(y_pred['Dayahead SE3'], y_pred['Intraday SE3'])
    delta_true = np.subtract(y_true['Dayahead SE3'], y_true['Intraday SE3'])

    dir_pred = np.sign(delta_pred)
    dir_true = np.sign(delta_true)

    print("\n # Summary statistics #")
    summary_statistics = {
        'Mean True Spread': [np.mean(delta_pred) for v in y_true.values()],
        'Std (true price)': [np.std(delta_pred) for v in y_true.values()],
        'Mean Predicted Spread': [np.mean(delta_true) for v in y_pred.values()],
        'Std (pred price)': [np.std(delta_true) for v in y_pred.values()]
    }
    print(pd.DataFrame(data=summary_statistics))

    print("\n # Error #")
    error = {  
        'MAE' : [mean_absolute_error(delta_true, delta_pred)],
        'MSE' : [mean_squared_error(delta_true, delta_pred)],
        'Accuracy' : [accuracy_score(dir_true, dir_pred)]
    }
    print(pd.DataFrame(data=error))
    
    print('\nClassification Report:')
    print(classification_report(dir_true, dir_pred, digits=3, labels=np.unique(dir_pred)))

    print('Confusion matrix:')
    print(confusion_matrix(dir_true, dir_pred, labels=np.unique(dir_pred))) """

    """ fig1, axs = plt.subplots(2, 1)
    colors = sns.color_palette("tab10")

    d_std = np.std(delta_pred)
    d_mean = np.mean(delta_pred)

    trade = []
    for s_pred, s_true in zip(delta_pred, delta_true):
        if np.abs(s_pred) > (1*d_std + np.abs(d_mean)):
            trade.append( np.sign(s_pred) == np.sign(s_true) )

    all_trade_acc = np.sum(np.sign(delta_pred) == np.sign(delta_true)) / len(delta_pred) """
    """ trade_rule_acc = np.sum(trade) / len(trade)

    print('\n             Spread Acc    N')
    print('All:         {:.3f}        {}'.format(all_trade_acc, len(delta_pred)))
    print('>std:        {:.3f}        {}'.format(trade_rule_acc, len(trade))) """

def objective(trial, model, train_set, val_set):
    model.define_model(trial)
    model = model.to(device)

    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    epochs = trial.suggest_int("epochs", 10, 100)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1)
    #max_norm = trial.suggest_int("max_norm", 1, 5)
    mse = train(model, train_set, val_set, lr=lr, epochs=epochs, weight_decay=weight_decay, trial=trial)

    return mse

def hyper_parameter_selection(model, train_set, val_set):
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, model, train_set, val_set), n_trials=200)

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    #optuna.visualization.plot_param_importances(study)

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

    for i in range(config.n_modes if config.ensemble else 1):
        if config.model == "MLP":
            model = MLP(input_size=config.input_size, output_size=config.output_size, mode=i)
            print(config.input_size)
        elif config.model == "LSTM":
            model = LSTM(input_size=config.input_size, output_size=config.output_size, mode=i)
        elif config.model == "GRU":
            model = GRU(input_size=config.input_size, output_size=config.output_size, mode=i)
        models.append(model)
    Dataset = Dataset(model_name=config.model, lookback=config.lookback, k=config.k, embedded_features=config.embedded_features, mode_decomp=config.mode_decomp)

    for i, model in enumerate(models):
        hyper_params = hyper_parameter_selection(model, Dataset.train_set[i], Dataset.validate_set[i])        
        model.set_optimized_model(hyper_params)
        model.to(device)
        train(model, Dataset.train_set[i], Dataset.validate_set[i], lr=hyper_params['lr'], epochs=hyper_params['epochs'], weight_decay=hyper_params['weight_decay'], plot=True)

    if config.mode_decomp and config.ensemble:
        model = EnsembleModel(models, n_features=config.input_size, output_size=config.output_size)
        models = [model]

    test(models, Dataset.validate_set)