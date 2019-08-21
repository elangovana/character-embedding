from torch.nn import NLLLoss
from torch.optim import SGD

from data_pipeline import DataPipeline
from label_pipeline import LabelPipeline
from rnn_network import RnnNetwork
from train import Train
from train_pipeline import TrainPipeline
from transform_label_encoder import TransformLabelEncoder
from transform_label_rehaper import TransformLabelReshaper
from transform_merge_tensors import TransformMergeTensors
from transform_text_to_index import TransformTextToIndex


class CharacterRnnTrainBuilder:

    def __init__(self, epochs=50, early_stopping=True, patience_epochs=10, batch_size=32, num_workers=None, **kwargs):
        self.batch_size = batch_size
        self.patience_epochs = patience_epochs
        self.early_stopping = early_stopping
        self.epochs = epochs

        self.learning_rate = kwargs.get("learning_rate", .01)
        self.hidden_dim = kwargs.get("hidden_dim", 20)
        self.num_workers = num_workers

    def get_pipeline(self, train_dataset):
        trainer = Train(patience_epochs=self.patience_epochs, early_stopping=self.early_stopping, epochs=self.epochs)

        max_feature_lens = train_dataset.max_feature_lens
        num_classes = train_dataset.num_classes

        text_to_index = TransformTextToIndex(feature_lens=max_feature_lens)

        # data pipeline
        merge_tensor = TransformMergeTensors()
        post_process_steps = [("merge_tensor", merge_tensor)]
        data_pipeline = DataPipeline(text_to_index=text_to_index, postprocess_steps=post_process_steps)

        # Label pipeline
        label_encoder = TransformLabelEncoder()
        label_reshaper = TransformLabelReshaper(num_classes=num_classes)

        label_pipeline = LabelPipeline(label_encoder=label_encoder, label_reshaper=label_reshaper)

        # Train pipeline
        model = RnnNetwork(input_size=text_to_index.max_index,
                           hidden_dim=self.hidden_dim,
                           output_size=train_dataset.num_classes)

        optimiser = SGD(lr=self.learning_rate, params=model.parameters())
        loss_func = NLLLoss()
        train_pipeline = TrainPipeline(batch_size=self.batch_size,
                                       optimiser=optimiser,
                                       trainer=trainer,
                                       data_pipeline=data_pipeline,
                                       label_pipeline=label_pipeline,
                                       num_workers=self.num_workers,
                                       loss_func=loss_func,
                                       model=model)

        return train_pipeline