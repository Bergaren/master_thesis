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
from models import LSTM, GRU, MLP, EnsembleModel, TwoHeadedHybridModel
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, accuracy_score, mean_squared_error, classification_report, confusion_matrix, ConfusionMatrixDisplay
from matplotlib.animation import FuncAnimation
from math import sqrt
from dataset import DataLoaderCreator
from config import Config

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
config = Config()

def train(model, train_set, val_set, lr = 10**-2, epochs=30, max_norm=2, weight_decay=0, trial=None, plot=False):
    print('\n Training... \n')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.SmoothL1Loss(beta=1)
    criterion2 = nn.BCEWithLogitsLoss()

    training_losses = []
    validation_losses = []
    for epoch in tqdm(range(epochs), desc='EPOCH'):
        model.train()
        y_pred = []
        
        y_true = []
        epoch_loss = []

        for i, batch in enumerate(tqdm(train_set, desc='Batch', disable=True)):
            optimizer.zero_grad()

            X, target = batch[0].to(device), batch[1].to(device)
            output = model(X)
            loss = criterion(output, target)

            #target, output = torch.sign(target), torch.sign(output)
            #target[target==-1], output[output==-1] = 0, 0
            #loss2 = criterion2(target, output)
            
            #loss = loss/torch.norm(loss) + loss2/torch.norm(loss2)

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
    criterion = torch.nn.SmoothL1Loss()

    losses = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataset, desc='Batch', disable=not verbose)):
            X, target = batch[0].to(device), batch[1].to(device)

            output = model(X)    

            loss = criterion(output, target)
            losses.append(loss.item())
    
    return np.average(losses)

def test(models, datasets, extension_set):
    print("\n Testing... \n")
    criterion = torch.nn.L1Loss()
    
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

    dir_pred = np.sign(y_pred)
    dir_true = np.sign(y_true)
    dir_pred[dir_pred==-1], dir_true[dir_true==-1] = 0, 0 

    print("\n ### Direction prediction ###")
    a = accuracy_score(dir_true, dir_pred)
    dir_naive = [1]*len(dir_true)
    a_naive = accuracy_score(dir_true, dir_naive)
    cm = confusion_matrix(dir_true, dir_pred)
    cr = classification_report(dir_true, dir_pred)
    
    print("\n # Accuracy #")
    print(a)

    print('\n # Consfusion Report #')
    print(cm)
    
    print('\n # Classification report #')
    print(cr)

    print("\n ### Price spread prediction ### \n")
    print("\n # Summary statistics #")
    summary_statistics = {
        'Mean True PS': [np.mean(y_true)],
        'Std (true)': [np.std(y_true)],
        'Mean Predicted PS': [np.mean(y_pred)],
        'Std (pred)': [np.std(y_pred)]
    }
    print(pd.DataFrame(data=summary_statistics))

    print("\n # Regression Error #")
    print("MAE:        {:.2f}".format( np.mean(losses) ))

    y_pred_extend = extension_set.dataset[-7:][1].flatten().tolist() + y_true
    y_naive = y_pred_extend[:-7*24]

    error = { 
        'MAE'   : [mean_absolute_error(y_true, y_pred)],
        'sMAPE' : [np.mean(2*np.abs(np.subtract(y_true,y_pred)) / (np.abs(y_true) + np.abs(y_pred)))],
        'rMAE'  : [mean_absolute_error(y_true, y_pred) / mean_absolute_error(y_true, y_naive)]
    }
    print(pd.DataFrame(data=error))

    ps_std = np.std(extension_set.dataset[:][1].flatten().tolist())

    trade = []
    for i in range(len(y_true)):
        if np.abs(y_pred[i]) > ps_std:
            trade.append( dir_pred[i] == dir_true[i] )

    trade_rule_acc = np.sum(trade) / len(trade)
    print(np.sum(trade))
    print(trade_rule_acc)

def feature_filter(params):
    series_f_idxs = []
    static_f_idxs = []

    if config.embedded_features:
        for k in config.f_mapping.keys():
            if k == 'price dayahead':
                if config.model == 'MLP':
                    series_f_idxs += list(config.f_mapping[k][-params['lookback']:])
                    print(len(list(config.f_mapping[k][-params['lookback']:])))
                else:
                    series_f_idxs += list(config.f_mapping[k])
            elif k == 'price intraday':
                if config.model == 'MLP':
                    series_f_idxs += list(config.f_mapping[k][-params['lookback']+11:])
                    print(len(config.f_mapping[k][-params['lookback']+11:]))
                else:
                    series_f_idxs += list(config.f_mapping[k])
            elif k == 'flow' and params[k] == 1:
                if config.model == 'MLP':
                    series_f_idxs += list(config.f_mapping[k][-params['lookback']:])
                else:
                    series_f_idxs += list(config.f_mapping[k])
            elif params[k] == 1:
                static_f_idxs += list(config.f_mapping[k])
    else:
        series_f_idxs += list(config.f_mapping['price'])
    
    return (series_f_idxs, static_f_idxs)

def objective(trial, model, train_set, val_set):
    trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    trial.suggest_int("epochs", 10, 100)
    trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)
    trial.suggest_float("beta", 1e-1,3)
    trial.suggest_int("lookback", config.min_lookback, config.max_lookback)

    # Set optimized model
    if config.embedded_features:
        trial.suggest_int("wp", 0, 1)
        trial.suggest_int("nuclear", 0, 1)
        trial.suggest_int("solar", 0, 1)
        trial.suggest_int("cons", 0, 1)
        trial.suggest_int("cap", 0, 1)
        trial.suggest_int("year", 0, 1)
        trial.suggest_int("week", 0, 1)
        trial.suggest_int("day of week", 0, 1)
        trial.suggest_int("holiday", 0, 1)
        trial.suggest_int("flow", 0, 1)
    
    f_idxs = feature_filter(trial.params)
    model.define_model(trial, f_idxs)
    model = model.to(device)

    mse = train(model, train_set, val_set, lr=trial.params['lr'], epochs=trial.params['epochs'], weight_decay=trial.params['weight_decay'], trial=trial)

    return mse

def hyper_parameter_selection(model, train_set, val_set):
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, model, train_set, val_set), n_trials=50)

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

    #for i in range(config.n_modes if config.ensemble else 1):
    i = 0
    if config.model == "MLP":
        model = MLP(output_size=config.output_size, mode=i)
    elif config.model == "LSTM":
        model = LSTM(output_size=config.output_size, mode=i)
    elif config.model == "GRU":
        model = GRU(output_size=config.output_size, mode=i)
    elif config.model == "THD":
        model = TwoHeadedHybridModel(output_size=config.output_size)
          
    models.append(model)
    Dataset = DataLoaderCreator(model_name=config.model, lookback=config.max_lookback, k=config.k, embedded_features=config.embedded_features, mode_decomp=config.mode_decomp)

    for i, model in enumerate(models):
        hyper_params = hyper_parameter_selection(model, Dataset.train_set[i], Dataset.validate_set[i])        
        model.set_optimized_model(hyper_params, feature_filter(hyper_params))
        model.to(device)
        train(model, Dataset.train_set[i], Dataset.validate_set[i], lr=hyper_params['lr'], epochs=hyper_params['epochs'], weight_decay=hyper_params['weight_decay'], plot=True)

    if config.mode_decomp and config.ensemble:
        model = EnsembleModel(models, n_features=config.input_size, output_size=config.output_size)
        models = [model]

    test(models, Dataset.validate_set, Dataset.train_set[0])