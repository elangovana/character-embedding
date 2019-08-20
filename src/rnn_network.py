from torch import nn


class RnnNetwork(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size, n_layers=1):
        super(RnnNetwork, self).__init__()
        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
        # Fully connected layer
        self.fc = nn.Sequential(nn.Linear(hidden_dim, output_size), nn.ReLU())

    def forward(self, X):
        _, out = self.rnn(X)
        out = self.fc(out)
        return out
