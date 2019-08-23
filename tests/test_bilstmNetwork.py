from unittest import TestCase

import torch

from bilstm_network import BiLstmNetwork


class TestBiLstmNetwork(TestCase):
    def test__call__(self):
        # Arrange

        batch_size = 32
        one_hot_vec_size = 27
        hidden_layer_size = 10
        output_size = 2
        n_layers = 2

        max_input_length = 15
        input = torch.randint(0, one_hot_vec_size - 1, (batch_size, max_input_length))
        expected_shape = (batch_size, output_size)

        # sut
        sut = BiLstmNetwork(input_size=one_hot_vec_size, hidden_dim=hidden_layer_size, output_size=output_size,
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
        actual = BiLstmNetwork._one_hot_encode(input, one_hot_vec_size)

        self.assertEqual(expected, actual.tolist())
