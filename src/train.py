import logging

import torch

from model_snapshotter import Snapshotter


class Train:

    def __init__(self, device=None, snapshotter=None, early_stopping=True, patience_epochs=10, epochs=10):
        # TODO: currently only single GPU
        self.epochs = epochs
        self.patience_epochs = patience_epochs
        self.early_stopping = early_stopping
        self.snapshotter = snapshotter
        self.device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')

    @property
    def logger(self):
        return logging.getLogger(__name__)

    @property
    def snapshotter(self):
        self._snapshotter = self._snapshotter or Snapshotter()
        return self._snapshotter

    @snapshotter.setter
    def snapshotter(self, value):
        self._snapshotter = value

    def run(self, train_data, val_data, model, loss_func, optimiser, output_dir):

        model.to(device=self.device)
        best_loss = None

        patience = 0
        previous_loss = None
        for e in range(self.epochs):

            n_batches = 0
            train_total_loss = 0
            total_correct = 0
            total_items = 0

            for i, (b_x, target) in enumerate(train_data):
                n_batches = i + 1
                # Set up train mode
                model.train()

                # Copy to device
                b_x = b_x.to(device=self.device)
                target = target.to(device=self.device)

                # Forward pass
                predicted_prob = model(b_x)
                loss = loss_func(predicted_prob, target)
                train_total_loss += loss.item()

                # Backward
                optimiser.zero_grad()
                loss.backward()

                # Update weights
                optimiser.step()

                # compute accuracy
                predicted_item = torch.max(predicted_prob, dim=1)[1]
                correct = (predicted_item == target).sum()
                total_correct += correct
                total_items += predicted_item.shape[0]

            train_loss = train_total_loss / n_batches
            train_accuracy = total_correct * 100.0 / total_items

            # Validation loss
            val_loss, val_accuracy = self._compute_validation_loss(val_data, model, loss_func)

            if best_loss is None:
                best_loss = val_loss

            # Save snapshots
            if val_loss < best_loss:
                self.snapshotter.save(model, output_dir=output_dir, prefix="snapshot_lowest_loss_")

            # Patience, early stopping
            if previous_loss is None:
                previous_loss = val_loss

            if val_loss > previous_loss:
                patience += 1
            else:
                # Reset patience if loss decreases
                patience = 0

            print("{} {} {} {} {}".format(e, train_loss, val_loss, train_accuracy, val_accuracy))

            if self.early_stopping and patience > self.patience_epochs:
                self.logger.info("No decrease in loss for {} epochs and hence stopping".format(self.patience_epochs))
                break

            previous_loss = val_loss

    def _compute_validation_loss(self, val_data, model, loss_func):
        # Model Eval mode
        model.eval()

        total_loss = 0
        n_batches = 0

        # No grad
        with torch.no_grad():
            total_correct = 0
            total_items = 0
            for i, (b_x, target) in enumerate(val_data):
                # Copy to device
                b_x = b_x.to(device=self.device)
                target = target.to(device=self.device)

                predicted_score = model(b_x)
                val_loss = loss_func(predicted_score, target)

                # Total loss
                total_loss += val_loss
                n_batches = i + 1

                # Accuracy
                predicted_item = torch.max(predicted_score, dim=1)[1]
                correct = (predicted_item == target).sum()
                total_correct += correct
                total_items += predicted_item.shape[0]

        average_loss = total_loss / n_batches
        accuracy = total_correct * 100.0 / total_items
        return average_loss, accuracy
