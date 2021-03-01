import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(RNN, self).__init__()
        self.input_size     = input_size
        self.output_size    = output_size
        self.lstm           = None
        self.fc             = None
    
    def define_model(self, trial):
        n_layers    = trial.suggest_int("n_layers", 1, 5)
        hidden_size = trial.suggest_int("n_units", 4, 128)
        p           = trial.suggest_float("dropout", 0.05, 0.5)

        self.lstm   = nn.LSTM(input_size=self.input_size, num_layers=n_layers, hidden_size=hidden_size, batch_first=True, dropout=p)
        self.fc     = nn.Linear(hidden_size, self.output_size)

    def forward(self, x):
        o, (h, c) = self.lstm(x)
        x = self.fc(o)
        return x

""" class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.layers(x) """

class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.layers = None
        self.input_size = input_size
        self.output_size = output_size

    def define_model(self, trial):
        n_layers = trial.suggest_int("n_layers", 1, 5)
        layers = []

        in_features = self.input_size
        for i in range(n_layers):
            out_features = trial.suggest_int("n_units_l{}".format(i), 4, 128)
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            p = trial.suggest_float("dropout_l{}".format(i), 0.05, 0.5)
            layers.append(nn.Dropout(p))

            in_features = out_features
        layers.append(nn.Linear(in_features, self.output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

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

class EnsembleModel(nn.Module):
    def __init__(self, models, n_features):
        super(EnsembleModel, self).__init__()
        self.models = models
        self.n_features = n_features

    def forward(self, x):
        output = []
        for i, model in enumerate(self.models):
            y = model(x[:,:,i*self.n_features:(i+1)*self.n_features])[:,-1]
            output.append(y)
        
        return sum(output) / len(output)

class HybridModel(nn.Module):
    def __init__(self, modelA, modelB):
        super(HybridModel, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
    
    def forward(self, x):
        x = self.modelA(x)
        x = self.modelB(x)
        
        return x