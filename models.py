import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, num_layers, hidden_size, output_size):
        super(RNN, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, num_layers=num_layers, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        o, (h, c) = self.lstm(x)
        x = self.fc(o)
        return x

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.layers(x)

class Autoencoder(nn.Module):
    def __init__(self, size, hidden_size):
        super(Autoencoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, size)
        )

    def forward(self, x):
        return self.layers(x)

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
        