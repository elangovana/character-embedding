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

import os
import tempfile
from unittest import TestCase

from experiment_email import ExperimentEmail


class TestSitExperimentEmail(TestCase):
    def test_run(self):
        """
        Check no exceptions are throwm
        """
        sut = ExperimentEmail()
        train = os.path.join(os.path.dirname(__file__), "data", "sample.csv")
        val = os.path.join(os.path.dirname(__file__), "data", "sample.csv")
        outdir = tempfile.mkdtemp()

        # Act
        sut.run(train, val, outdir, batch_size=32, epochs=2)
