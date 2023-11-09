from core.data.converters.abstract_converter import AbstractDataConverter
from core.data.converters.helpers import convert_to_tensor


class PaddedTensorConverter(AbstractDataConverter):
    def __init__(self, size, padding_values=0, types='int32'):
        super(PaddedTensorConverter, self).__init__()
        self.size = size
        if isinstance(types, list):
            self.types = types
        else:
            self.types = [types] * size
        if isinstance(padding_values, list):
            self.padding_values = padding_values
        else:
            self.padding_values = [padding_values] * size

    def convert(self, batch):
        results = list()
        for item, type, value in zip(batch, self.types, self.padding_values):
            if item is None:
                results.append(item)
            else:
                results.append(convert_to_tensor(item, type, value))
        return results
