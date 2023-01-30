import datetime
import pandas as pd
import numpy as np
import holidays
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.discrete.discrete_model import Probit
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, accuracy_score, mean_squared_error, classification_report, confusion_matrix, ConfusionMatrixDisplay
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)


def load_data(lag, rolling_window, start, stop):
    pdf = pd.read_csv('data/price_data.csv', encoding="ISO-8859-1", sep=',',
                      decimal='.', index_col='Delivery', parse_dates=['Delivery'])
    fdf = pd.read_csv('data/feature_data.csv', encoding="ISO-8859-1",
                      sep=',', decimal='.', index_col=0, parse_dates=True)

    se_holidays = holidays.CountryHoliday('SE')

    delta_rolling_window = datetime.timedelta(days=rolling_window)
    delta_lag = datetime.timedelta(days=lag)
    delta_day = datetime.timedelta(days=1)
    delta_hour = datetime.timedelta(hours=1)
    start_date = datetime.datetime(
        start, 1, 1, 00, 00, 00) - delta_lag - delta_rolling_window
    end_date = datetime.datetime(stop, 1, 1, 00, 00, 00)

    endogenous_n_exogenous = {
        'date': [],
        'dayahead': [],
        'intraday': [],
        'spread': [],
        'dir': [],
        'cons': [],
        'wp': [],
        'lagged': [],
        'holiday': [],
        'monday': [],
        'saturday': [],
        'spread_l2': [],
        'spread_l3': [],
        'spread_l4': [],
        'spread_l5': [],
        'spread_l6': [],
        'spread_l7': [],
    }

    while start_date < end_date:
        todays_prices = pdf.loc[start_date]
        todays_features = fdf.loc[start_date]
        yeasterdays_prices = pdf.loc[start_date-delta_day]

        endogenous_n_exogenous['date'].append(start_date)

        endogenous_n_exogenous['dayahead'].append(
            todays_prices['Dayahead SE3'])
        endogenous_n_exogenous['intraday'].append(
            todays_prices['Intraday SE3'])

        spread = np.subtract(
            todays_prices['Dayahead SE3'], todays_prices['Intraday SE3'])
        endogenous_n_exogenous['spread'].append(spread.tolist())

        if start_date >= (datetime.datetime(start, 1, 1, 00, 00, 00)-delta_rolling_window):
            endogenous_n_exogenous['spread_l2'].append(
                endogenous_n_exogenous['spread'][-24*2-1])
            endogenous_n_exogenous['spread_l3'].append(
                endogenous_n_exogenous['spread'][-24*3-1])
            endogenous_n_exogenous['spread_l4'].append(
                endogenous_n_exogenous['spread'][-24*4-1])
            endogenous_n_exogenous['spread_l5'].append(
                endogenous_n_exogenous['spread'][-24*5-1])
            endogenous_n_exogenous['spread_l6'].append(
                endogenous_n_exogenous['spread'][-24*6-1])
            endogenous_n_exogenous['spread_l7'].append(
                endogenous_n_exogenous['spread'][-24*7-1])
        else:
            endogenous_n_exogenous['spread_l2'].append(0)
            endogenous_n_exogenous['spread_l3'].append(0)
            endogenous_n_exogenous['spread_l4'].append(0)
            endogenous_n_exogenous['spread_l5'].append(0)
            endogenous_n_exogenous['spread_l6'].append(0)
            endogenous_n_exogenous['spread_l7'].append(0)

        direction = np.sign(spread)
        direction = np.where(direction == -1, 0, direction)
        endogenous_n_exogenous['dir'].append(direction.tolist())

        endogenous_n_exogenous['cons'].append(todays_features['cons se'])
        endogenous_n_exogenous['wp'].append(todays_features['wp se'])

        endogenous_n_exogenous['lagged'].append(
            yeasterdays_prices['Dayahead SE3'])

        is_holiday = 1 if start_date in se_holidays else 0
        endogenous_n_exogenous['holiday'].append(is_holiday)

        is_monday = 1 if start_date.weekday() == 0 else 0
        endogenous_n_exogenous['monday'].append(is_monday)

        is_saturday = 1 if start_date.weekday() == 5 else 0
        endogenous_n_exogenous['saturday'].append(is_saturday)

        start_date += delta_hour

    data = pd.DataFrame.from_dict(endogenous_n_exogenous)
    data.index = data['date']
    data.drop(['date'], axis=1, inplace=True)
    data.index.freq = data.index.inferred_freq

    return data


def run(data, start, stop, rolling_window, lag, variable, exog_label, model_name):
    delta_lookback = datetime.timedelta(days=rolling_window+max(lag))
    delta_day = datetime.timedelta(days=1)

    y_pred = []
    y_true = []

    while start < stop:
        y = data.loc[start][variable]
        x = data.loc[start-delta_lookback:start-delta_day]

        if model_name == 'probit':
            model = Probit(x[variable], exog=x[exog_label], old_names=True)
            try:
                model_fit = model.fit(methd='nm', maxiter=10000, disp=0)
                exog = data[exog_label].loc[start:start]
                y_hat = model_fit.predict(exog=exog)
            except:
                if y_pred:
                    y_hat = y_pred[-1]
                    print('Filled with previous')
                else:
                    y_hat = 0.0
                    print('Filled with zero')

        else:
            model = AutoReg(x[variable], exog=x[exog_label],
                            lags=lag, old_names=True)
            model_fit = model.fit()
            exog = data[exog_label].loc[start:start]
            y_hat = model_fit.predict(start=start, end=start, exog_oos=exog)

        y_pred.append(y_hat)
        y_true.append(y)

        start += delta_day

    return y_pred, y_true


def run_ar(data, start, stop, rolling_window, lag, y_type, exog, test=False):
    y_pred, y_true = [], []

    for h in range(0, 24):

        validate_start = datetime.datetime(start, 1, 1, h, 0, 0)
        validate_stop = datetime.datetime(stop, 1, 1, 0, 0, 0)

        if y_type == 'seperate':
            y_pred_d, y_true_d = run(data[data.index.hour == h], validate_start, validate_stop, rolling_window, [
                                     1] + lag, 'dayahead', exog, model_name='ar')
            y_pred_i, y_true_i = run(data[data.index.hour == h], validate_start, validate_stop,
                                     rolling_window, lag, 'intraday', exog+['lagged'], model_name='ar')

            pred = np.subtract(y_pred_d, y_pred_i)
            true = np.subtract(y_true_d, y_true_i)
        else:
            pred, true = run(data[data.index.hour == h], validate_start, validate_stop,
                             rolling_window, lag, 'spread', exog+['lagged'], model_name='ar')

        y_pred.append(pred)
        y_true.append(true)

    y_pred = np.array(y_pred).flatten('F')
    y_true = np.array(y_true).flatten('F')

    s = datetime.datetime(start, 1, 1, 0, 0, 0)

    print_ar_results(y_pred, y_true, data['spread'].loc[:s], test)


def run_probit(data, start, stop, rolling_window, lag, exog, test=False):
    y_pred, y_true = [], []

    for l in lag:
        exog.append('spread_l{}'.format(l))

    for h in range(0, 24):
        print('Hour: {}'.format(h))
        validate_start = datetime.datetime(start, 1, 1, h, 0, 0)
        validate_stop = datetime.datetime(stop, 1, 1, 0, 0, 0)

        pred, true = run(data[data.index.hour == h], validate_start,
                         validate_stop, rolling_window, lag, 'dir', exog, model_name='probit')

        y_pred.append(pred)
        y_true.append(true)

    y_pred = np.array(y_pred).flatten('F')
    y_true = np.array(y_true).flatten('F')

    if test:
        y = {
            'pred': y_pred,
            'true': y_true
        }
        pd.DataFrame.from_dict(y).to_csv('results/test_probit.csv')

    s = datetime.datetime(start, 1, 1, 0, 0, 0)
    print_probit_results(y_pred, y_true, test)


def print_ar_results(y_pred, y_true, extension_set, test):
    dir_pred = np.sign(y_pred)
    dir_true = np.sign(y_true)
    dir_pred[dir_pred == -1], dir_true[dir_true == -1] = 0, 0

    with open('results/log_ar.txt', 'a') as f:
        if test:
            print_n_write(f, "\n #### On test data ####")

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
        ext = extension_set.iloc[-7*24:].tolist()
        y_pred_extend = ext + y_true.tolist()
        y_naive = y_pred_extend[:-7*24]

        error = {
            'MAE': [mean_absolute_error(y_true, y_pred)],
            'sMAPE': [np.mean(2*np.abs(np.subtract(y_true, y_pred)) / (np.abs(y_true) + np.abs(y_pred)))],
            'rMAE': [mean_absolute_error(y_true, y_pred) / mean_absolute_error(y_true, y_naive)]
        }
        print_n_write(f, pd.DataFrame(data=error))


def print_probit_results(y_pred, y_true, test):

    for mu in [0.3, 0.4, 0.5]:
        y_dir = np.where(y_pred > mu, 1, 0)
        with open('results/log_probit.txt', 'a') as f:
            if test:
                print_n_write(f, "\n ### On test data ###")

            print_n_write(f, "\nThreshold: {}".format(mu))

            a = accuracy_score(y_true, y_dir)
            dir_naive = [1]*len(y_true)
            a_naive = accuracy_score(y_true, dir_naive)
            cm = confusion_matrix(y_true, y_dir)
            cr = classification_report(y_true, y_dir)

            print_n_write(f, "\n # Accuracy # \n")
            print_n_write(f, a)

            print_n_write(f, "\n # Accuracy naive # \n")
            print_n_write(f, a_naive)

            print_n_write(f, '\n # Consfusion report # \n')
            print_n_write(f, cm)

            print_n_write(f, '\n # Classification report # \n')
            print_n_write(f, cr)


def print_n_write(f, s):
    print(s)
    f.write(str(s))


if __name__ == '__main__':
    max_lag = 7
    max_rolling_window = 365
    test = True

    if not test:
        start = 2019
        stop = 2020
        data = load_data(
            lag=max_lag, rolling_window=max_rolling_window, start=start, stop=stop)

        for y_type in ['seperate', 'spread']:
            for exog in [['holiday', 'monday', 'saturday'], ['holiday', 'monday', 'saturday', 'cons'], ['holiday', 'monday', 'saturday', 'wp'], ['holiday', 'monday', 'saturday', 'cons', 'wp']]:
                for rolling_window in [30, 91, 182, 365]:
                    for lag in [[2], [2, 7], list(range(2, 8))]:

                        with open('results/log_ar.txt', 'a') as f:
                            print_n_write(f, '\n\nVariables: \n\t{} \n\t{} \n\t{} \n\t{}'.format(
                                y_type, exog, rolling_window, lag))

                        run_ar(data, start, stop,
                               rolling_window, lag, y_type, exog)

        for exog in [['holiday', 'monday', 'saturday', 'lagged'], ['holiday', 'monday', 'saturday', 'lagged', 'cons'], ['holiday', 'monday', 'saturday', 'lagged', 'wp'], ['holiday', 'monday', 'saturday', 'lagged', 'cons', 'wp']]:
            for rolling_window in [30, 91, 182, 365]:
                for lag in [[2], [2, 7], list(range(2, 8))]:

                    with open('results/log_probit.txt', 'a') as f:
                        print_n_write(f, '\n\nVariables: \n\t{} \n\t{} \n\t{}'.format(
                            exog, rolling_window, lag))

                    run_probit(data, start, stop,
                               rolling_window, lag, exog.copy())
    else:
        start = 2020
        stop = 2021
        data = load_data(
            lag=max_lag, rolling_window=max_rolling_window, start=start, stop=stop)

        """ for y_type in ['seperate', 'spread']:
         for exog in [['holiday', 'monday', 'saturday'], ['holiday', 'monday', 'saturday', 'cons'], ['holiday', 'monday', 'saturday', 'wp'], ['holiday', 'monday', 'saturday', 'cons','wp']]:
            for rolling_window in [30, 91, 182, 365]:
               for lag in [[2], [2,7], list(range(2,8))]:

                  with open('results/log_ar.txt', 'a') as f:
                     print_n_write(f, '\n\nVariables: \n\t{} \n\t{} \n\t{} \n\t{}'.format(y_type, exog, rolling_window, lag))

                  run_ar(data, start, stop, rolling_window, lag, y_type, exog, test=True) """

        for exog in [['holiday', 'monday', 'saturday', 'lagged', 'wp'], ['holiday', 'monday', 'saturday', 'lagged', 'cons', 'wp']]:
            for rolling_window in [91, 182, 365]:
                for lag in [[2], [2, 7], list(range(2, 8))]:

                    with open('results/log_probit.txt', 'a') as f:
                        print_n_write(f, '\n\nVariables: \n\t{} \n\t{} \n\t{}'.format(
                            exog, rolling_window, lag))

                    run_probit(data, start, stop, rolling_window,
                               lag, exog.copy(), test=True)
