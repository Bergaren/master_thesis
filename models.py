import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, output_size=24, mode=0):
        super(LSTM, self).__init__()
        self.input_size     = None
        self.output_size    = output_size
        self.mode           = mode

        self.f_idxs         = None
        self.lstm           = None
        self.fc             = None
        self.lookback       = None
    
    def define_model(self, trial, f_idxs):
        series_f_idxs, static_f_idxs    = f_idxs
        self.f_idxs                     = series_f_idxs + static_f_idxs
        self.input_size                 = len(self.f_idxs)
        self.lookback                   = trial.params['lookback']

        prefix      = chr(ord('@')+self.mode+1)
        n_layers    = trial.suggest_int("{}_n_layers".format(prefix), 1, 3)
        hidden_size = trial.suggest_int("{}_n_units".format(prefix), 4, 256)
        p           = trial.suggest_float("{}_dropout".format(prefix), 0.05, 0.5)

        self.lstm   = nn.LSTM(input_size=self.input_size, num_layers=n_layers, hidden_size=hidden_size, batch_first=True, dropout=p)                
        self.fc     = nn.Linear(hidden_size, self.output_size)
    
    def set_optimized_model(self, params, f_idxs):
        series_f_idxs, static_f_idxs    = f_idxs
        self.f_idxs                     = series_f_idxs + static_f_idxs
        self.input_size                 = len(self.f_idxs)
        self.lookback                   = params['lookback']

        prefix      = chr(ord('@')+self.mode+1)
        n_layers    = params["{}_n_layers".format(prefix)]
        hidden_size = params["{}_n_units".format(prefix)]
        p           = params["{}_dropout".format(prefix)]

        self.lstm   = nn.LSTM(input_size=self.input_size, num_layers=n_layers, hidden_size=hidden_size, batch_first=True, dropout=p)                
        self.fc     = nn.Linear(hidden_size, self.output_size)

    def forward(self, x):
        o, _ = self.lstm(x[:,-self.lookback:,self.f_idxs])
        x = self.fc(o)
        return x[:,-1]

class GRU(nn.Module):
    def __init__(self, output_size=None, mode=0):
        super(GRU, self).__init__()
        self.input_size     = None
        self.output_size    = output_size
        self.gru            = None
        self.fc             = None
        self.mode           = mode

        self.lookback       = None
    
    def define_model(self, trial, f_idxs):
        series_f_idxs, static_f_idxs    = f_idxs
        self.f_idxs                     = series_f_idxs + static_f_idxs
        self.input_size                 = len(self.f_idxs)
        self.lookback                   = trial.params['lookback']
        
        prefix      = chr(ord('@')+self.mode+1)
        n_layers    = trial.suggest_int("{}_n_layers".format(prefix), 1, 3)
        hidden_size = trial.suggest_int("{}_n_units".format(prefix), 4, 256)
        p           = trial.suggest_float("{}_dropout".format(prefix), 0.05, 0.5)

        self.gru    = nn.GRU(input_size=self.input_size, num_layers=n_layers, hidden_size=hidden_size, batch_first=True, dropout=p)                
        self.fc     = nn.Linear(hidden_size, self.output_size)

    def set_optimized_model(self, params, f_idxs):
        series_f_idxs, static_f_idxs    = f_idxs
        self.f_idxs                     = series_f_idxs + static_f_idxs
        self.input_size                 = len(self.f_idxs)

        prefix      = chr(ord('@')+self.mode+1)
        n_layers    = params["{}_n_layers".format(prefix)]
        hidden_size = params["{}_n_units".format(prefix)]
        p           = params["{}_dropout".format(prefix)]

        self.gru    = nn.GRU(input_size=self.input_size, num_layers=n_layers, hidden_size=hidden_size, batch_first=True, dropout=p)                
        self.fc     = nn.Linear(hidden_size, self.output_size)

    def forward(self, x):
        o, _ = self.gru(x[:,-self.lookback:,self.f_idxs])
        x = self.fc(o)
        return x[:,-1]

class MLP(nn.Module):
    def __init__(self, output_size=None, mode=0):
        super(MLP, self).__init__()
        self.input_size     = None
        self.output_size    = output_size
        self.layers         = None
        self.mode           = mode
        self.f_idxs         = None

    def define_model(self, trial, f_idxs):
        series_f_idxs, static_f_idxs    = f_idxs
        self.f_idxs                     = series_f_idxs + static_f_idxs
        self.input_size                 = len(self.f_idxs)

        prefix = chr(ord('@')+self.mode+1)
        n_layers = trial.suggest_int("{}_n_layers".format(prefix), 1, 3)
        layers = []

        in_features = self.input_size
        for i in range(n_layers):
            out_features = trial.suggest_int("{}_n_units_l{}".format(prefix, i), 4, 256)
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            p = trial.suggest_float("{}_dropout_l{}".format(prefix, i), 0.05, 0.5)
            layers.append(nn.Dropout(p))

            in_features = out_features

        layers.append(nn.Linear(in_features, self.output_size))
        self.layers = nn.Sequential(*layers)

    def set_optimized_model(self, params, f_idxs):
        series_f_idxs, static_f_idxs    = f_idxs
        self.f_idxs                     = series_f_idxs + static_f_idxs
        self.input_size                 = len(self.f_idxs)

        prefix      = chr(ord('@')+self.mode+1)
        n_layers    = params["{}_n_layers".format(prefix)]
        layers      = []

        in_features = self.input_size
        for i in range(n_layers):
            out_features = params["{}_n_units_l{}".format(prefix, i)]
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            p = params["{}_dropout_l{}".format(prefix, i)]
            layers.append(nn.Dropout(p))

            in_features = out_features

        layers.append(nn.Linear(in_features, self.output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x[:,self.f_idxs])

class EnsembleModel(nn.Module):
    def __init__(self, models, n_features, output_size):
        super(EnsembleModel, self).__init__()
        self.n_features = n_features
        self.models = models
        
        self.aggregator = nn.Sequential(
            nn.ReLU(),
            nn.Linear(sum([m.output_size for m in self.models]), output_size)
        )
    
    def define_model(self, trial):
        for i in range(len(self.models)):
            self.models[i].define_model(trial)

    def forward(self, x):
        y = self.models[0](x)

        for i, model in enumerate(self.models[1:]):
            y2 = model(x)
            y = torch.cat((y, y2), dim=1)

        return self.aggregator(y)
        
class TwoHeadedHybridModel(nn.Module):
    def __init__(self, output_size=24, rnn_type='LSTM'):
        super(TwoHeadedHybridModel, self).__init__()
        self.rnn_type       = rnn_type
        self.output_size    = output_size

        self.rnn_head       = LSTM(None, 0) if rnn_type=='LSTM' else GRU(None, 0)
        self.mlp_head       = MLP(None, 1)

        self.ensemble       = MLP(output_size, 2)
        self.activation     = nn.ReLU()

        self.static_f_idxs  = None
        self.series_f_idxs  = None
    
    def forward(self, x):
        x1 = self.rnn_head(x)
        x2 = self.mlp_head(x[:,0])

        x = self.activation( torch.cat((x1, x2), dim=1) )
        return self.ensemble(x)
    
    def define_model(self, trial, f_idxs):
        series_f_idxs, static_f_idxs    = f_idxs
        self.series_f_idxs              = series_f_idxs
        self.static_f_idxs              = static_f_idxs

        rnn_output_size = trial.suggest_int("RNN_head_output_size", 4, 24)
        mlp_output_size = trial.suggest_int("MLP_head_output_size", 4, 24)

        self.rnn_head.output_size = rnn_output_size
        self.rnn_head.define_model(trial, (self.series_f_idxs, []))

        self.mlp_head.output_size = mlp_output_size
        self.mlp_head.define_model(trial, ([], self.static_f_idxs))

        input_size = rnn_output_size + mlp_output_size
        self.ensemble.define_model(trial, (list(range(input_size)),[]))

    def set_optimized_model(self, params, f_idxs):
        series_f_idxs, static_f_idxs    = f_idxs
        self.series_f_idxs              = series_f_idxs
        self.static_f_idxs              = static_f_idxs

        rnn_output_size = params["RNN_head_output_size"]
        mlp_output_size = params["MLP_head_output_size"]

        self.rnn_head.output_size = rnn_output_size
        self.rnn_head.set_optimized_model(params, (self.series_f_idxs, []))

        self.mlp_head.output_size = mlp_output_size
        self.mlp_head.set_optimized_model(params, ([], self.static_f_idxs))

        input_size = rnn_output_size + mlp_output_size
        self.ensemble.set_optimized_model(params, (list(range(input_size)),[]))
