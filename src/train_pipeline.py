import os

from torch.utils.data import DataLoader


class TrainPipeline:

    def __init__(self, trainer, model, loss_func, optimiser, data_pipeline, label_pipeline, num_workers=None,
                 batch_size=32):
        self.label_pipeline = label_pipeline
        self.data_pipeline = data_pipeline
        self.batch_size = batch_size
        self.num_workers = num_workers
        if self.num_workers is None:
            self.num_workers = 1 if os.cpu_count() == 1 else os.cpu_count() - 1
        self.trainer = trainer
        self.optimiser = optimiser
        self.loss_func = loss_func
        self.model = model

    def run(self, train_dataset, val_dataset, output_dir):
        train_data = DataLoader(train_dataset, num_workers=self.num_workers, shuffle=True, batch_size=self.batch_size)
        val_data = DataLoader(val_dataset, num_workers=self.num_workers, shuffle=False, batch_size=self.batch_size)

        # Transform features
        transformed_train = self.data_pipeline.fit_transform(train_data)
        transformed_val = self.data_pipeline.transform(val_data)

        # Transform labels
        transformed_train = self.label_pipeline.fit_transform(transformed_train)
        transformed_val = self.label_pipeline.transform(transformed_val)

        self.trainer.run(transformed_train, transformed_val, self.model, self.loss_func, self.optimiser, output_dir)
