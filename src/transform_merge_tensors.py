import torch


class TransformMergeTensors:

    def __init__(self):
        pass

    def fit(self, dataloader):
        pass

    def transform(self, dataloader):
        """
Expects a list of batches where each batch a 2-tuple, bx and by.
len of bx is equal to number of columns. And each column contains a list fo values

        :param dataloader:
        """
        result = []
        for _, (bx, by) in enumerate(dataloader):
            result.append([torch.cat(bx), by])

        return result
