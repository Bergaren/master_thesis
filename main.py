import torch
import copy
import torch.nn as nn
from torch.autograd import Variable
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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
from os import path

device = "cuda" if torch.cuda.is_available() else "cpu"
config = Config()


def position_weight(dataset):
    target = dataset[:][1]
    l = list(target.size())[1]
    pos_weights = []
    for i in range(l):
        pos = sum(target[:, i])
        neg = len(target[:, i]) - pos
        pos_weights.append(neg/pos)
    return torch.tensor(pos_weights).to(device)


def train(model, train_set, val_set, lr=10**-2, epochs=300, weight_decay=0, trial=None, plot_training=None):
    print('\n Training... \n')
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.SmoothL1Loss(beta=1)

    if config.regression:
        criterion = nn.SmoothL1Loss(beta=1)
    else:
        pos_weights = position_weight(train_set.dataset)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    c = 0
    best_run = float('inf')
    checkpoint = model.state_dict()

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

            loss.backward()
            epoch_loss.append(loss.item())

            optimizer.step()

        training_losses.append(np.average(epoch_loss))
        val_loss = validate(model, val_set)
        validation_losses.append(val_loss)

        if trial:
            trial.report(loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        if val_loss > best_run:
            c += 1
        else:
            c = 0
            best_run = val_loss
            checkpoint = copy.deepcopy(model.state_dict())

        if c >= config.patience:
            if trial == None:
                model.load_state_dict(checkpoint)
            break

    if plot_training:
        plt.plot(training_losses, label='Training loss', c=config.colors[0])
        plt.plot(validation_losses, label='Validation loss',
                 c=config.colors[1])
        plt.xlabel('Epoch')
        plt.ylabel('Val Loss')
        plt.plot(np.argmin(validation_losses), min(validation_losses),
                 '*', c=config.colors[2], label='Early stop')
        plt.legend()

        plt.savefig('results/misc/{}_{}.png'.format(plot_training, config.name))
        plt.close()

    return best_run


def validate(model, dataset, verbose=False):
    if verbose:
        print("\n Validating... \n")

    model.eval()

    if config.regression:
        criterion = nn.SmoothL1Loss(beta=1)
    else:
        criterion = nn.BCEWithLogitsLoss()

    losses = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataset, desc='Batch', disable=not verbose)):
            X, target = batch[0].to(device), batch[1].to(device)

            output = model(X)

            loss = criterion(output, target)

            losses.append(loss.item())

    return np.average(losses)


def test(model, dataset, extension_set, test_type):
    print("\n Testing... \n")

    y_pred = []
    y_true = []

    model.eval()
    activation = nn.Sigmoid()

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataset, desc='Batch')):
            X, target = batch[0].to(device), batch[1].to(device)
            output = model(X)

            output = output.flatten()
            if not config.regression:
                output = activation(output)

            y_pred.extend(output.tolist())
            y_true.extend(target.flatten().tolist())

    results = {
        'pred': y_pred,
        'true': y_true
    }
    results = pd.DataFrame.from_dict(results)
    results.to_csv(config.get_results_path(test_type, 'csv'))

    if config.regression:
        dir_pred = np.sign(y_pred)
        dir_true = np.sign(y_true)
        dir_pred[dir_pred == -1], dir_true[dir_true == -1] = 0, 0
    else:
        dir_pred = np.rint(y_pred)
        dir_true = y_true

    with open(config.get_results_path(test_type, 'txt'), 'w') as f:
        print_n_write(f, "\n ### Direction prediction ###")
        a = accuracy_score(dir_true, dir_pred)
        dir_naive = [1]*len(dir_true)
        a_naive = accuracy_score(dir_true, dir_naive)
        cm = confusion_matrix(dir_true, dir_pred)
        cr = classification_report(dir_true, dir_pred)

        print_n_write(f, "\n # Accuracy # \n")
        print_n_write(f, a)

        print_n_write(f, "\n # Accuracy naive # \n")
        print_n_write(f, a_naive)

        print_n_write(f, '\n # Consfusion report # \n')
        print_n_write(f, cm)

        print_n_write(f, '\n # Classification report # \n')
        print_n_write(f, cr)

        if config.regression:
            print_n_write(f, "\n ### Price spread prediction ### \n")
            print_n_write(f, "\n # Summary statistics # \n")
            summary_statistics = {
                'Mean True PS': [np.mean(y_true)],
                'Std (true)': [np.std(y_true)],
                'Mean Predicted PS': [np.mean(y_pred)],
                'Std (pred)': [np.std(y_pred)]
            }
            print_n_write(f, pd.DataFrame(data=summary_statistics))

            print_n_write(f, "\n # Regression error # \n")
            y_pred_extend = extension_set.dataset[-7:][1].flatten(
            ).tolist() + y_true
            y_naive = y_pred_extend[:-7*24]

            error = {
                'MAE': [mean_absolute_error(y_true, y_pred)],
                'sMAPE': [np.mean(2*np.abs(np.subtract(y_true, y_pred)) / (np.abs(y_true) + np.abs(y_pred)))],
                'rMAE': [mean_absolute_error(y_true, y_pred) / mean_absolute_error(y_true, y_naive)]
            }
            print_n_write(f, pd.DataFrame(data=error))

            print_n_write(f, "\n ### Direction strategy ### \n")
            try:
                trade_true = []
                trade_pred = []
                for i in range(len(y_true)):
                    if np.abs(y_pred[i]) >= 1:
                        trade_true.append(dir_true[i])
                        trade_pred.append(dir_pred[i])

                a = accuracy_score(trade_true, trade_pred)
                cm = confusion_matrix(trade_true, trade_pred)
                cr = classification_report(trade_true, trade_pred)

                print_n_write(f, "\n # Accuracy # \n")
                print_n_write(f, a)

                print_n_write(f, '\n # Consfusion Report # \n')
                print_n_write(f, cm)

                print_n_write(f, '\n # Classification report # \n')
                print_n_write(f, cr)
            except:
                print_n_write(f, '\n Test failed \n')


def feature_filter(params):
    '''
        Since the dataset is created before the hyperparameters optimisation a mapping to wanted
        features are created. So each batch do in fact contain all features but the unwanted ones
        are filtered out in the model forward step.
    '''

    series_f_idxs = []
    static_f_idxs = []

    if config.embedded_features:
        for k in config.f_mapping.keys():
            if k == 'price dayahead':
                if config.model == 'MLP':
                    series_f_idxs += list(config.f_mapping[k]
                                          [-params['lookback']*24:])
                else:
                    series_f_idxs += list(config.f_mapping[k])
            elif k == 'price intraday':
                if config.model == 'MLP':
                    series_f_idxs += list(config.f_mapping[k]
                                          [-params['lookback']*24+11:])
                else:
                    series_f_idxs += list(config.f_mapping[k])
            elif k in ['flow 1', 'flow 2', 'flow 3', 'flow 4', 'flow 5'] and params['flow'] == 1:
                if config.model == 'MLP':
                    series_f_idxs += list(config.f_mapping[k]
                                          [-params['lookback']*24:])
                else:
                    series_f_idxs += list(config.f_mapping[k])
            elif k in params.keys() and params[k] == 1:
                static_f_idxs += list(config.f_mapping[k])
    else:
        series_f_idxs += list(config.f_mapping['price dayahead'])
        series_f_idxs += list(config.f_mapping['price intraday'])

    return (series_f_idxs, static_f_idxs)


def objective(trial, model, train_set, val_set):
    # Define the hyperparameters that are outside the model.
    trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)
    trial.suggest_int("lookback", config.min_lookback, config.max_lookback)

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

    hubber_loss = train(model, train_set, val_set,
                        lr=trial.params['lr'], weight_decay=trial.params['weight_decay'], trial=trial)

    return hubber_loss


def hyper_parameter_selection(model, train_set, val_set, mode):
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(
        trial, model, train_set, val_set), n_trials=200)

    pruned_trials = [t for t in study.trials if t.state ==
                     optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state ==
                       optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    with open(config.get_param_path(mode), 'w') as f:
        print_n_write(f, "  Params: ")
        for key, value in trial.params.items():
            print_n_write(f, "    {}: {} \n".format(key, value))

    return trial.params


def print_n_write(f, s):
    print(s)
    f.write(str(s))


def construct_model(Dataset):
    models = []
    params = []
    checkpoints = []

    for i in range(config.n_models):
        if config.model == "MLP":
            model = MLP(output_size=config.output_size, mode=i)
        elif config.model == "LSTM":
            model = LSTM(output_size=config.output_size, mode=i)
        elif config.model == "GRU":
            model = GRU(output_size=config.output_size, mode=i)
        elif config.model == "THH":
            model = TwoHeadedHybridModel(
                output_size=config.output_size, rnn_type=config.rnn_head, mode=i)

        # Find optimal hyper parameters.
        hyper_params = hyper_parameter_selection(
            model, Dataset.train_set, Dataset.validate_set, i)
        model.set_optimized_model(hyper_params, feature_filter(hyper_params))
        model.to(device)

        # Save a copy of the model with its initiated model parameters.
        checkpoint = copy.deepcopy(model.state_dict())
        train(model, Dataset.train_set, Dataset.validate_set,
              lr=hyper_params['lr'], weight_decay=hyper_params['weight_decay'], plot_training='validate')

        if config.ensemble:
            models.append(model)
            params.append(hyper_params)

            checkpoints.append(checkpoint)

    if config.ensemble:
        model = EnsembleModel(models)

    # Validate its performance
    test(model, Dataset.validate_set, Dataset.train_set, 'validation')

    if config.ensemble:
        for i, m in enumerate(model.models):
            m.load_state_dict(checkpoints[i])
            train(m, Dataset.final_train_set, Dataset.final_validate_set,
                  lr=params[i]['lr'], weight_decay=params[i]['weight_decay'], plot_training='test')
    else:
        model.load_state_dict(checkpoint)
        train(model, Dataset.final_train_set, Dataset.final_validate_set,
              lr=hyper_params['lr'], weight_decay=hyper_params['weight_decay'], plot_training='test')
    return model


if __name__ == "__main__":
    Dataset = DataLoaderCreator(model_name=config.model, lookback=config.max_lookback,
                                embedded_features=config.embedded_features, regression=config.regression)

    # Load the model if there already exists a trained one.
    if path.exists(config.get_model_path()):
        model = torch.load(config.get_model_path())
    else:
        # Otherwise construct one and save it.
        model = construct_model(Dataset)
        torch.save(model, config.get_model_path())

    # Run final test.
    test(model, Dataset.test_set, Dataset.validate_set, 'test')
