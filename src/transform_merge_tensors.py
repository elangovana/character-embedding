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


class TransformMergeTensors:

    def __init__(self):
        pass

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
            result.append([torch.cat(bx), by])

        return result
