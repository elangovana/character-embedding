import argparse
import logging
import os
import sys

from character_rnn_train_builder import CharacterRnnTrainBuilder
from datasets.email_dataset import EmailDataset


class ExperimentEmail:

    def __init__(self):
        pass

    def run(self, train_file, val_file, out_dir, batch_size=32, epochs=10, **kwargs):
        # Set up dataset
        train_dataset = EmailDataset(train_file)
        val_dataset = EmailDataset(val_file)

        # Get trainpipeline
        builder = CharacterRnnTrainBuilder(epochs=epochs, batch_size=batch_size, kwargs=kwargs)
        train_pipeline = builder.get_pipeline(train_dataset)

        # Start training
        train_pipeline.run(train_dataset, val_dataset, out_dir)


def list_csv_files(path, limit):
    files = [os.path.join(path, f) for f in
             filter(lambda x: os.path.isfile(os.path.join(path, x)) and x.endswith(".csv"), os.listdir(path))]
    assert len(files) == limit, "Expected exactly {} .csv file in {}, but found {}".format(limit, path, len(files))

    return files

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--traindir",
                        help="The input train  data", default=os.environ.get("SM_CHANNEL_TRAIN", None))
    parser.add_argument("--valdir",
                        help="The input val data", default=os.environ.get("SM_CHANNEL_VAL", None))

    parser.add_argument("--outdir", help="The output dir", default=os.environ.get("SM_MODEL_DIR", None))

    parser.add_argument("--batchsize", help="The batch size", type=int, default=32)

    parser.add_argument("--epochs", help="The number of epochs", type=int, default=10)

    parser.add_argument("--log-level", help="Log level", default="INFO", choices={"INFO", "WARN", "DEBUG", "ERROR"})

    args = parser.parse_args()

    print(args.__dict__)

    # Set up logging
    logging.basicConfig(level=logging.getLevelName(args.log_level), handlers=[logging.StreamHandler(sys.stdout)],
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    train_csv_file = list_csv_files(args.traindir, limit=1)[0]
    val_csv_file = list_csv_files(args.valdir, limit=1)[0]

    ExperimentEmail().run(train_csv_file,
                          val_csv_file,
                          args.outdir,
                          args.batchsize,
                          args.epochs)
