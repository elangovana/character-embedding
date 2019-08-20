import logging
import os

import torch


class Snapshotter(object):

    @property
    def logger(self):
        return logging.getLogger(__name__)

    def save(self, model, output_dir, prefix="Snapshot"):
        snapshot_prefix = os.path.join(output_dir, prefix)
        snapshot_path = snapshot_prefix + 'model.pt'

        self.logger.info("Snappshotting model to {} ".format(snapshot_path))

        torch.save(model, snapshot_path)
