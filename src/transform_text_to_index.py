import string


class TransformTextToIndex:

    def __init__(self, feature_lens):
        self.feature_lens = feature_lens

    @property
    def all_letters(self):
        return string.printable

    @property
    def pad_index(self):
        return len(self.all_letters)

    @property
    def max_index(self):
        # All characters + pad
        return len(self.all_letters) + 1

    def transform(self, dataloader):
        """
Expects a list of batches where each batch a 2-tuple, bx and by.
Size of bx is equal to number of columns. And each column contains a list fo values

        :param dataloader:
        """
        result = []
        for _, (bx, by) in enumerate(dataloader):
            transformed_cols = []
            for ci, c in enumerate(bx):
                col_len = self.feature_lens[ci]
                transformed_rc = []
                for r in c:
                    transformed_rc.append(self._transform_text(r, col_len))
                transformed_cols.append(transformed_rc)
            result.append([transformed_cols, by])

        return result

    def _transform_text(self, text, length):
        result = [self.all_letters.find(c) for c in text[0:length]]
        result = result + [self.pad_index] * (length - len(result))

        return result
