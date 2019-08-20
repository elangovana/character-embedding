from io import StringIO
from unittest import TestCase

from datasets.email_dataset import EmailDataset


class TestEmailDataset(TestCase):
    def test__len__(self):
        # Arrange
        input_handle = StringIO("""yes, test@yahoo.com""")
        sut = EmailDataset(file_or_handle=input_handle)
        expected_len = 1

        # Act
        actual_length = len(sut)

        # Assert
        self.assertEqual(expected_len, actual_length)

    def test__getitem__(self):
        # Arrange
        input_handle = StringIO("""yes,test@yahoo.com""")
        sut = EmailDataset(file_or_handle=input_handle)
        expected_feature = ['test@yahoo.com'], 'yes'

        # Act
        actual_feature = sut[0]

        # Assert
        self.assertEqual(expected_feature, actual_feature)
