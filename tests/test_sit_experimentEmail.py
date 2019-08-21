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
