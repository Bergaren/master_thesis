import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, output_size=24, mode=0):
        super(LSTM, self).__init__()
        self.input_size = None
        self.output_size = output_size
        self.mode = mode

        self.f_idxs = None
        self.lstm = None
        self.fc = None
        self.lookback = None

    def define_model(self, trial, f_idxs):
        # Let the TPE algorithm suggest model hyperparameters.
        series_f_idxs, static_f_idxs = f_idxs
        self.f_idxs = series_f_idxs + static_f_idxs
        self.input_size = len(self.f_idxs)
        self.lookback = trial.params['lookback']

        prefix = chr(ord('@')+self.mode+1)
        n_layers = trial.suggest_int("{}_{}_n_layers".format(
            self.__class__.__name__, prefix), 1, 8)
        hidden_size = trial.suggest_int("{}_{}_n_units".format(
            self.__class__.__name__, prefix), 4, 256)
        p = trial.suggest_float("{}_{}_dropout".format(
            self.__class__.__name__, prefix), 0.05, 0.5) if n_layers > 1 else 0

        self.lstm = nn.LSTM(input_size=self.input_size, num_layers=n_layers,
                            hidden_size=hidden_size, batch_first=True, dropout=p)
        self.fc = nn.Linear(hidden_size, self.output_size)

    def set_optimized_model(self, params, f_idxs):
        # Set the hyperparameters that maximised performance on the test set.
        series_f_idxs, static_f_idxs = f_idxs
        self.f_idxs = series_f_idxs + static_f_idxs
        self.input_size = len(self.f_idxs)
        self.lookback = params['lookback']

        prefix = chr(ord('@')+self.mode+1)
        n_layers = params["{}_{}_n_layers".format(
            self.__class__.__name__, prefix)]
        hidden_size = params["{}_{}_n_units".format(
            self.__class__.__name__, prefix)]
        p = params["{}_{}_dropout".format(
            self.__class__.__name__, prefix)] if n_layers > 1 else 0

        self.lstm = nn.LSTM(input_size=self.input_size, num_layers=n_layers,
                            hidden_size=hidden_size, batch_first=True, dropout=p)
        self.fc = nn.Linear(hidden_size, self.output_size)

    def forward(self, x):
        o, _ = self.lstm(x[:, -self.lookback*24:, self.f_idxs])
        x = self.fc(o)
        return x[:, -1]


class GRU(nn.Module):
    def __init__(self, output_size=None, mode=0):
        super(GRU, self).__init__()
        self.input_size = None
        self.output_size = output_size
        self.gru = None
        self.fc = None
        self.mode = mode

        self.lookback = None

    def define_model(self, trial, f_idxs):
        # Let the TPE algorithm suggest model hyperparameters.
        series_f_idxs, static_f_idxs = f_idxs
        self.f_idxs = series_f_idxs + static_f_idxs
        self.input_size = len(self.f_idxs)
        self.lookback = trial.params['lookback']

        prefix = chr(ord('@')+self.mode+1)
        n_layers = trial.suggest_int("{}_{}_n_layers".format(
            self.__class__.__name__, prefix), 1, 8)
        hidden_size = trial.suggest_int("{}_{}_n_units".format(
            self.__class__.__name__, prefix), 4, 256)
        p = trial.suggest_float("{}_{}_dropout".format(
            self.__class__.__name__, prefix), 0.05, 0.5) if n_layers > 1 else 0

        self.gru = nn.GRU(input_size=self.input_size, num_layers=n_layers,
                          hidden_size=hidden_size, batch_first=True, dropout=p)
        self.fc = nn.Linear(hidden_size, self.output_size)

    def set_optimized_model(self, params, f_idxs):
        # Set the hyperparameters that maximised performance on the test set.
        series_f_idxs, static_f_idxs = f_idxs
        self.f_idxs = series_f_idxs + static_f_idxs
        self.input_size = len(self.f_idxs)

        prefix = chr(ord('@')+self.mode+1)
        n_layers = params["{}_{}_n_layers".format(
            self.__class__.__name__, prefix)]
        hidden_size = params["{}_{}_n_units".format(
            self.__class__.__name__, prefix)]
        p = params["{}_{}_dropout".format(
            self.__class__.__name__, prefix)] if n_layers > 1 else 0

        self.gru = nn.GRU(input_size=self.input_size, num_layers=n_layers,
                          hidden_size=hidden_size, batch_first=True, dropout=p)
        self.fc = nn.Linear(hidden_size, self.output_size)

    def forward(self, x):
        o, _ = self.gru(x[:, -self.lookback*24:, self.f_idxs])
        x = self.fc(o)
        return x[:, -1]


class MLP(nn.Module):
    def __init__(self, output_size=None, mode=0, name=None):
        super(MLP, self).__init__()
        self.input_size = None
        self.output_size = output_size
        self.layers = None
        self.mode = mode
        self.f_idxs = None
        self.name = name if name else self.__class__.__name__

    def define_model(self, trial, f_idxs):
        # Let the TPE algorithm suggest model hyperparameters.
        series_f_idxs, static_f_idxs = f_idxs
        self.f_idxs = series_f_idxs + static_f_idxs
        self.input_size = len(self.f_idxs)

        prefix = chr(ord('@')+self.mode+1)
        n_layers = trial.suggest_int(
            "{}_{}_n_layers".format(self.name, prefix), 1, 8)
        layers = []

        in_features = self.input_size
        for i in range(n_layers):
            out_features = trial.suggest_int(
                "{}_{}_n_units_l{}".format(self.name, prefix, i), 4, 256)
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.LeakyReLU())
            p = trial.suggest_float("{}_{}_dropout_l{}".format(
                self.name, prefix, i), 0.05, 0.5)
            layers.append(nn.Dropout(p))

            in_features = out_features

        layers.append(nn.Linear(in_features, self.output_size))
        self.layers = nn.Sequential(*layers)

    def set_optimized_model(self, params, f_idxs):
        # Set the hyperparameters that maximised performance on the test set.
        series_f_idxs, static_f_idxs = f_idxs
        self.f_idxs = series_f_idxs + static_f_idxs
        self.input_size = len(self.f_idxs)

        prefix = chr(ord('@')+self.mode+1)
        n_layers = params["{}_{}_n_layers".format(self.name, prefix)]
        layers = []

        in_features = self.input_size
        for i in range(n_layers):
            out_features = params["{}_{}_n_units_l{}".format(
                self.name, prefix, i)]
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            p = params["{}_{}_dropout_l{}".format(self.name, prefix, i)]
            layers.append(nn.Dropout(p))

            in_features = out_features

        layers.append(nn.Linear(in_features, self.output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x[:, self.f_idxs])


class TwoHeadedHybridModel(nn.Module):
    def __init__(self, output_size=24, rnn_type='LSTM', mode=0):
        super(TwoHeadedHybridModel, self).__init__()
        self.rnn_type = rnn_type
        self.output_size = output_size

        self.rnn_head = LSTM(
            None, mode) if rnn_type == 'LSTM' else GRU(None, mode)
        self.mlp_head = MLP(None, mode)

        self.ensemble = MLP(output_size, mode, name='Tail')
        self.activation = nn.LeakyReLU()

        self.static_f_idxs = None
        self.series_f_idxs = None

    def forward(self, x):
        x1 = self.rnn_head(x)
        x2 = self.mlp_head(x[:, -1])

        x = self.activation(torch.cat((x1, x2), dim=1))
        return self.ensemble(x)

    def define_model(self, trial, f_idxs):
        # Let the TPE algorithm suggest model hyperparameters.
        series_f_idxs, static_f_idxs = f_idxs
        self.series_f_idxs = series_f_idxs
        self.static_f_idxs = static_f_idxs

        rnn_output_size = trial.suggest_int("RNN_head_output_size", 4, 24)
        mlp_output_size = trial.suggest_int("MLP_head_output_size", 4, 24)

        self.rnn_head.output_size = rnn_output_size
        self.rnn_head.define_model(trial, (self.series_f_idxs, []))

        self.mlp_head.output_size = mlp_output_size
        self.mlp_head.define_model(trial, ([], self.static_f_idxs))

        input_size = rnn_output_size + mlp_output_size
        self.ensemble.define_model(trial, (list(range(input_size)), []))

    def set_optimized_model(self, params, f_idxs):
        # Set the hyperparameters that maximised performance on the test set.
        series_f_idxs, static_f_idxs = f_idxs
        self.series_f_idxs = series_f_idxs
        self.static_f_idxs = static_f_idxs

        rnn_output_size = params["RNN_head_output_size"]
        mlp_output_size = params["MLP_head_output_size"]

        self.rnn_head.output_size = rnn_output_size
        self.rnn_head.set_optimized_model(params, (self.series_f_idxs, []))

        self.mlp_head.output_size = mlp_output_size
        self.mlp_head.set_optimized_model(params, ([], self.static_f_idxs))

        input_size = rnn_output_size + mlp_output_size
        self.ensemble.set_optimized_model(
            params, (list(range(input_size)), []))


class EnsembleModel(nn.Module):
    def __init__(self, models):
        super(EnsembleModel, self).__init__()
        self.models = models

    def define_model(self, trial):
        for i in range(len(self.models)):
            self.models[i].define_model(trial)

    def forward(self, x):
        # Calculate the arthimetic average of all models in the ensemble.
        y = self.models[0](x)
        for m in self.models[1:]:
            y = torch.add(y, m(x))
        return torch.div(y, len(self.models))
