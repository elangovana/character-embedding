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
