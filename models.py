import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, output_size, mode=0):
        super(LSTM, self).__init__()
        self.input_size     = input_size
        self.output_size    = output_size
        self.mode           = mode
        self.lstm           = None
        self.fc             = None
    
    def define_model(self, trial):
        prefix      = chr(ord('@')+self.mode+1)
        n_layers    = trial.suggest_int("{}_n_layers".format(prefix), 1, 10)
        hidden_size = trial.suggest_int("{}_n_units".format(prefix), 4, 256)
        p           = trial.suggest_float("{}_dropout".format(prefix), 0.05, 0.5)

        self.lstm   = nn.LSTM(input_size=self.input_size, num_layers=n_layers, hidden_size=hidden_size, batch_first=True, dropout=p)                
        self.fc     = nn.Linear(hidden_size, self.output_size)
    
    def set_optimized_model(self, params):
        prefix      = chr(ord('@')+self.mode+1)
        n_layers    = params["{}_n_layers".format(prefix)]
        hidden_size = params["{}_n_units".format(prefix)]
        p           = params["{}_dropout".format(prefix)]

        self.lstm   = nn.LSTM(input_size=self.input_size, num_layers=n_layers, hidden_size=hidden_size, batch_first=True, dropout=p)                
        self.fc     = nn.Linear(hidden_size, self.output_size)

    def forward(self, x):
        x = x[:,:, self.input_size*self.mode : self.input_size*(self.mode+1)]
        o, _ = self.lstm(x)
        x = self.fc(o)
        return x[:,-1]

class GRU(nn.Module):
    def __init__(self, input_size, output_size, mode=0):
        super(GRU, self).__init__()
        self.input_size     = input_size
        self.output_size    = output_size
        self.gru            = None
        self.fc             = None
        self.mode           = mode
    
    def define_model(self, trial):
        prefix      = chr(ord('@')+self.mode+1)
        n_layers    = trial.suggest_int("{}_n_layers".format(prefix), 1, 10)
        hidden_size = trial.suggest_int("{}_n_units".format(prefix), 4, 256)
        p           = trial.suggest_float("{}_dropout".format(prefix), 0.05, 0.5)

        self.gru    = nn.GRU(input_size=self.input_size, num_layers=n_layers, hidden_size=hidden_size, batch_first=True, dropout=p)                
        self.fc     = nn.Linear(hidden_size, self.output_size)

    def set_optimized_model(self, params):
        prefix      = chr(ord('@')+self.mode+1)
        n_layers    = params["{}_n_layers".format(prefix)]
        hidden_size = params["{}_n_units".format(prefix)]
        p           = params["{}_dropout".format(prefix)]

        self.gru    = nn.GRU(input_size=self.input_size, num_layers=n_layers, hidden_size=hidden_size, batch_first=True, dropout=p)                
        self.fc     = nn.Linear(hidden_size, self.output_size)

    def forward(self, x):
        o, _ = self.gru(x)
        x = self.fc(o)
        return x[:,-1]

class MLP(nn.Module):
    def __init__(self, input_size, output_size, mode):
        super(MLP, self).__init__()
        self.layers = None
        self.input_size = input_size
        self.output_size = output_size
        self.mode = mode

    def define_model(self, trial):
        prefix = chr(ord('@')+self.mode+1)
        n_layers = trial.suggest_int("{}_n_layers".format(prefix), 1, 10)
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

    def set_optimized_model(self, params):
        prefix = chr(ord('@')+self.mode+1)
        n_layers = params["{}_n_layers".format(prefix)]
        layers = []

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
        x = x[:,self.input_size*self.mode : self.input_size*(self.mode+1)]
        return self.layers(x)

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
        
class HybridModel(nn.Module):
    def __init__(self, modelA, modelB):
        super(HybridModel, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
    
    def forward(self, x):
        x = self.modelA(x)
        x = self.modelB(x)
        return x


class Autoencoder(nn.Module):
    def __init__(self, input_size, output_size, decode=True):
        super(Autoencoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.decode
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, self.output_size)
        )

    def forward(self, x):
        encoding = self.encoder(x)
        if self.decode:
            return self.decoder(encoding)
        else:
            return encoding