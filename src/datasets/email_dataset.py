import csv

from torch.utils.data import Dataset

"""
Simple email dataset, csv separated. E.g.

myemail@gmail.com,"N"
test@yahoo.com,"Y"
"""


class EmailDataset(Dataset):

    def __init__(self, file_or_handle):
        self._features, self._labels = self._load_csv_file(file_or_handle)

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, index):
        return self._features[index], self._labels[index]

    @property
    def num_classes(self):
        return 2

    @property
    def max_feature_lens(self):
        return [254]

    def _load_csv_file(self, file_or_handle):
        if isinstance(file_or_handle, str):
            with open(file_or_handle, "r") as f:
                return self._load_csv_handle(f)

        return self._load_csv_handle(file_or_handle)

    def _load_csv_handle(self, handle):
        x = []
        y = []
        label_index = 0
        for fields in csv.reader(handle):
            x.append(fields[1:])
            y.append(fields[label_index])
        return x, y
