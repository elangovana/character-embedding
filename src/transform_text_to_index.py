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

import string

import torch


class TransformTextToIndex:

    def __init__(self, feature_lens):
        self.feature_lens = feature_lens

    @property
    def all_letters(self):
        return string.printable

    @property
    def pad_index(self):
        return len(self.all_letters)

    @property
    def max_index(self):
        # All characters + pad
        return len(self.all_letters) + 1

    def fit(self, dataloader):
        pass

    def transform(self, dataloader):
        """
Expects a list of batches where each batch a 2-tuple, bx and by.
len of bx is equal to number of columns. And each column contains a list fo values

        :param dataloader:
        """
        result = []
        for _, (bx, by) in enumerate(dataloader):
            transformed_cols = []
            for ci, c in enumerate(bx):
                col_len = self.feature_lens[ci]
                transformed_rc = []
                for r in c:
                    transformed_rc.append(self._transform_text(r, col_len))
                transformed_rc = torch.Tensor(transformed_rc).long()
                transformed_cols.append(transformed_rc)
            result.append([transformed_cols, by])

        return result

    def _transform_text(self, text, length):
        result = [self.all_letters.find(c) for c in text[0:length]]
        result = result + [self.pad_index] * (length - len(result))

        return result
