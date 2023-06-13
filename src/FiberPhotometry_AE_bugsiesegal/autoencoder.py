import torch
import torch.nn as nn


class Encoder(nn.Module):
    pass


class Decoder(nn.Module):
    pass


class AutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(input_size, hidden_size, num_layers=num_layers)
        self.decoder = Decoder(hidden_size, input_size, num_layers=num_layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def fit(self, data):
        pass
