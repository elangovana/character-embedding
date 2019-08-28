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

from unittest import TestCase

import torch

from src.rnn_network import RnnNetwork


class TestRnnNetwork(TestCase):
    def test__call__(self):
        # Arrange

        batch_size = 2
        one_hot_vec_size = 27
        hidden_layer_size = 10
        output_size = 2
        n_layers = 2

        input = torch.randint(0, one_hot_vec_size - 1, (batch_size, 15))
        expected_shape = (batch_size, output_size)

        # sut
        sut = RnnNetwork(input_size=one_hot_vec_size, hidden_dim=hidden_layer_size, output_size=output_size,
                         n_layers=n_layers)

        # Act
        acutal = sut(input)

        # Assert
        self.assertSequenceEqual(expected_shape, acutal.shape)

    def test__one_hot_encode(self):
        one_hot_vec_size = 10
        input = torch.tensor([[1, 5]])
        expected = [[[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]]]
        # Act
        actual = RnnNetwork._one_hot_encode(input, one_hot_vec_size)

        self.assertEqual(expected, actual.tolist())
