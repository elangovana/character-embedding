import logging


class DataPipeline:

    def __init__(self, text_to_index, preprocess_steps=None, postprocess_steps=None):
        self.postprocess_steps = postprocess_steps or []
        self.text_to_index = text_to_index
        self.preprocess_steps = preprocess_steps or []

    @property
    def logger(self):
        return logging.getLogger(__name__)

    def transform(self, dataloader):
        transformed_x = dataloader
        for name, p in self.feature_pipeline:
            transformed_x = p.transform(transformed_x)
        return transformed_x

    def update_vocab_dict(self, vocab_dict):
        self.text_to_index.vocab_dict = vocab_dict

    def fit_transform(self, dataloader):
        self.fit(dataloader)
        return self.transform(dataloader)

    def fit(self, dataloader):

        # set up pipeline
        self.feature_pipeline = self.preprocess_steps + [("text_to_index", self.text_to_index)] + self.postprocess_steps

        # load count vectoriser after loading pretrained vocab
        for name, p in self.feature_pipeline:
            self.logger.info("Running transform {}".format(name))
            p.fit(dataloader)
            self.logger.info("Completed transform {}".format(name))
