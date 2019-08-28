import logging

import torch

from model_snapshotter import Snapshotter
from result_writer import ResultWriter


class Train:

    def __init__(self, device=None, snapshotter=None, early_stopping=True, patience_epochs=10, epochs=10,
                 results_writer=None):
        # TODO: currently only single GPU
        self.results_writer = results_writer
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

    @property
    def results_writer(self):
        self._results_writer = self._results_writer or ResultWriter()
        return self._results_writer

    @results_writer.setter
    def results_writer(self, value):
        self._results_writer = value

    def run(self, train_data, val_data, model, loss_func, optimiser, output_dir):
        self.logger.info("Running training...")

        model.to(device=self.device)
        best_loss = None

        patience = 0
        previous_loss = None
        result_logs = []

        for e in range(self.epochs):

            predicted_items = []
            target_items = []

            n_batches = 0
            train_total_loss = 0
            total_correct = 0
            total_items = 0

            for i, (b_x, target) in enumerate(train_data):
                # use this for confusion matrix later
                target_items.extend(target.tolist())

                # Set up train mode
                model.train()
                len = b_x.shape[0]

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
                predicted_items.extend(predicted_item.tolist())

                self.logger.debug(
                    "Batch {}/{}, total correct {}. loss {}".format(i, e, (correct * 100 / len), loss.item()))

            train_loss = train_total_loss
            train_accuracy = (total_correct.float() * 100.0 / total_items).item()
            # Train confusion matrix
            self._print_confusion_matrix(target_items, predicted_items, "Train")

            # Validation loss
            val_loss, val_accuracy = self._compute_validation_loss(val_data, model, loss_func)

            if best_loss is None:
                best_loss = val_loss

            # Save snapshots
            # TODO Check negative loss
            if val_loss < best_loss:
                self.snapshotter.save(model, output_dir=output_dir, prefix="snapshot_lowest_loss_")

            # Patience, early stopping
            if previous_loss is None:
                previous_loss = val_loss

            if val_loss >= previous_loss:
                patience += 1
            else:
                # Reset patience if loss decreases
                patience = 0

            print("###score: train_loss### {}".format(train_loss))
            print("###score: val_loss### {}".format(val_loss))
            print("###score: train_accuracy### {}".format(train_accuracy))
            print("###score: val_accuracy### {}".format(val_accuracy))
            # TODO: ADD F-SCORE
            # print("###score: train_f_score### {}").format("")
            # print("###score: val_f_score### {}".format(""))

            self.logger.info(
                "epoch: {}, train_loss {}, val_loss {}, train_accuracy {}, val_accuracy {}".format(e, train_loss,
                                                                                                   val_loss,
                                                                                                   train_accuracy,
                                                                                                   val_accuracy))
            result_logs.append([e, train_loss, val_loss, train_accuracy, val_accuracy])

            if self.early_stopping and patience > self.patience_epochs:
                self.logger.info("No decrease in loss for {} epochs and hence stopping".format(self.patience_epochs))
                break

            previous_loss = val_loss

        self.results_writer.dump_object(result_logs, output_dir, "epochs_loss")

    def _compute_validation_loss(self, val_data, model, loss_func):
        # Model Eval mode
        model.eval()

        total_loss = 0
        n_batches = 0

        # No grad
        predicted_items = []
        target_items = []
        with torch.no_grad():
            total_correct = 0
            total_items = 0
            for i, (b_x, target) in enumerate(val_data):
                target_items.extend(target.tolist())
                # Copy to device
                b_x = b_x.to(device=self.device)
                target = target.to(device=self.device)

                predicted_score = model(b_x)
                val_loss = loss_func(predicted_score, target)

                # Total loss
                total_loss += val_loss.item()

                # Accuracy
                predicted_item = torch.max(predicted_score, dim=1)[1]
                correct = (predicted_item == target).sum()
                total_correct += correct
                total_items += predicted_item.shape[0]
                predicted_items.extend(predicted_item.tolist())

        average_loss = total_loss
        accuracy = (total_correct.float() * 100.0 / total_items).item()
        self._print_confusion_matrix(target_items, predicted_items, "Validation ")

        return average_loss, accuracy

    @staticmethod
    def _print_confusion_matrix(y_actual, y_pred, title):
        from sklearn.metrics import confusion_matrix
        cnf_matrix = confusion_matrix(y_actual, y_pred)

        print("{} Confusion matrix,  \n{}".format(title, cnf_matrix))

    # TODO: implement this, f-score..
    #  @staticmethod
    # def _get_f_score(y_acutal, y_pred, num_classes):
    #     F1Score = torch.zeros(len(classes))
    #     for cls in range(len(classes)):
    #         try:
    #             F1Score[cls] = 2. * confusion[cls, cls] / (np.sum(confusion[cls, :]) + np.sum(confusion[:, cls]))
    #         except:
    #             pass
    #     print("F1Score: ")
    #     for cls, score in enumerate(F1Score):
    #         print("{}: {:.2f}".format(classes[cls], score))
