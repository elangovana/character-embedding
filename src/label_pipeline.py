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

from sklearn.pipeline import Pipeline


class LabelPipeline:

    def __init__(self, label_reshaper, label_encoder):
        self._label_encoder = label_encoder
        self._label_reshaper = label_reshaper
        self._label_pipeline = None

    @property
    def label_reverse_encoder_func(self):
        return self._label_encoder.inverse_transform

    def transform(self, data_loader):
        # Unbatch Y
        return self._label_pipeline.transform(data_loader)

    def fit_transform(self, data_loader):
        self.fit(data_loader)
        return self.transform(data_loader)

    def fit(self, data_loader):
        self._label_pipeline = Pipeline(steps=
                                        [("label_encoder", self._label_encoder)
                                            , ("label_reshaper", self._label_reshaper)])

        for name, p in self._label_pipeline.steps:
            p.fit(data_loader)
