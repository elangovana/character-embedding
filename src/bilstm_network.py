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


class BiLstmNetwork(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size, n_layers=1):
        super(BiLstmNetwork, self).__init__()
        self.n_layers = n_layers
        self.input_size = input_size
        # Lstm Layer
        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True, bidirectional=True)
        # Fully connected layer, 2 is for bi-directional
        fc_size = 30
        self.fc = nn.Sequential(nn.Linear(hidden_dim * 2, fc_size), nn.ReLU(), nn.Linear(fc_size, output_size))

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
        out, (_, _) = self.lstm(input, )

        # Get final timestep
        # TODO: check what the final is for bilstm
        out = out[:, last_timestep, ]

        # FC
        out = self.fc(out)
        return out

    @staticmethod
    def _one_hot_encode(batch, input_size_depth):
        # Change shape from  (batch, features) to (batch, features, 1)
        view = batch.view(batch.size() + torch.Size([1]))
        # Create zero tensor of shape (batch, features, depth)

        zeros = torch.zeros((batch.size() + torch.Size([input_size_depth])), device=batch.device)

        input = zeros.scatter_(2, view, 1)
        return input
