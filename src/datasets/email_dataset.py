import csv

from torch.utils.data import Dataset

"""
Simple email dataset, csv separated. E.g.

Y,afhg11@domain.com
N,afhg22@domain.com
Y,afhg33@domain.com
N,afhg44@domain.com

"""


class EmailDataset(Dataset):

    def __init__(self, file_or_handle):
        self._features, self._labels = self._load_csv_file(file_or_handle)

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, index):
        name = self._features[index]

        # Get the name from the email, e.g aaa from aa@domain.com
        if "@" in name:
            name = name.split("@")[0]
        return name, self._labels[index]

    @property
    def num_classes(self):
        return 2

    @property
    def max_feature_lens(self):
        return [100]

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
