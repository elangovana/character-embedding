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

import logging

from sklearn import preprocessing


class TransformLabelEncoder:
    """
    Extracts vocab from data frame columns which have already been tokenised into words
    """

    def __init__(self):
        self._encoder = preprocessing.LabelEncoder()

    @property
    def logger(self):
        return logging.getLogger(__name__)

    def fit(self, data_loader):
        self.logger.info("Running TransformLabelEncoder")
        y = []
        for idx, (bx, by) in enumerate(data_loader):
            b_y = by
            y.extend(b_y)
        self._encoder.fit(y)
        self.logger.info("Complete TransformLabelEncoder")

    def transform(self, data_loader):
        # Check if iterable
        try:
            iter(data_loader)
            iterable = not isinstance(data_loader, str)

        except TypeError:
            iterable = False

        # Looks like single value
        if not iterable:
            return self._encoder.transform([data_loader])[0]

        batches = []
        for idx, (b_x, b_y) in enumerate(data_loader):
            encoded_y = self._encoder.transform(b_y)
            batches.append([b_x, encoded_y])
        return batches

    def inverse_transform(self, Y):
        # Check if iterable
        try:
            int(Y)
            return self._encoder.inverse_transform([Y])[0]
        except TypeError:
            pass

        i = []
        for _, b in enumerate(Y):
            i.append(b)
        return self._encoder.inverse_transform(i)

    def fit_transform(self, data_loader):

        self.fit(data_loader)
        return self.transform(data_loader)
