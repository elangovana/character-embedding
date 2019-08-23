############################################################################
# * Copyright 2019 Amazon.com, Inc. and its affiliates. All Rights Reserved.
#
# Licensed under the Amazon Software License (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
# http://aws.amazon.com/asl/
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
# #############################################################################

import torch
from torch import nn


class RnnNetwork(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size, n_layers=1):
        super(RnnNetwork, self).__init__()
        self.n_layers = n_layers
        self.input_size = input_size
        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True, nonlinearity='tanh')
        # Fully connected layer
        self.fc = nn.Sequential(nn.Linear(hidden_dim, output_size), nn.ReLU())
        # softmax
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        """

        :param X: is a 2 d tensor

                    torch.tensor([
                        # 3 rows of character to index all fixed to same length
                        [29, 14, 28, 29, 1],
                        [29, 14, 28, 29, 2],
                        [29, 14, 28, 29, 2]
                    ]),


        :return:
        """
        last_timestep = X.shape[1] - 1

        # One hot encode..
        input = self._one_hot_encode(X, self.input_size)

        # Pass through network
        out, hidden = self.rnn(input)

        # # Get final rnn layer output
        # # Obtain the last timestep output
        out = out[:, last_timestep, ]

        # FC
        out = self.fc(out)

        # Softmax
        out = self.softmax(out)

        return out

    @staticmethod
    def _one_hot_encode(batch, input_size_depth):
        # Change shape from  (batch, features) to (batch, features, 1)
        view = batch.view(batch.size() + torch.Size([1]))
        # Create zero tensor of shape (batch, features, depth)
        zeros = torch.zeros((batch.size() + torch.Size([input_size_depth])))

        input = zeros.scatter_(2, view, 1)
        return input
