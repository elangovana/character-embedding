from unittest import TestCase

import torch

from src.RnnNetwork import RnnNetwork


class TestRnnNetwork(TestCase):
    def test__call__(self):
        # Arrange
        input_length = 50
        batch_size = 32
        one_hot_vec_size = 27
        hidden_layer_size = 10
        output_size = 2
        n_layers = 2

        input = torch.rand((batch_size, input_length, one_hot_vec_size))
        expected_shape = (n_layers, batch_size, output_size)

        # sut
        sut = RnnNetwork(input_size=one_hot_vec_size, hidden_dim=hidden_layer_size, output_size=output_size,
                         n_layers=n_layers)

        # Act
        acutal = sut(input)

        # Assert
        self.assertSequenceEqual(expected_shape, acutal.shape)
