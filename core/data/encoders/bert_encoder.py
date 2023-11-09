import os
from bert.tokenization import FullTokenizer

from core.data.encoders.abstract_encoder import AbstractEncoder


def recursive_apply_new(fn, sequence, num=1):
    results = list()
    for _ in range(num):
        results.append(list())

    if not isinstance(sequence, list):
        return None
    elif not isinstance(sequence[0], list):
        return fn(sequence)
    else:
        for item in sequence:
            outputs = recursive_apply_new(fn, item)
            for result, output in zip(results, outputs):
                result.append(output)
    return results


class BertEncoder(AbstractEncoder):
    def __init__(self, bert_path, lowercase):
        super(BertEncoder, self).__init__()

        self.tokenizer = None
        self.__init(bert_path, lowercase)

        self.CLS = '[CLS]'
        self.SEP = '[SEP]'

    def __init(self, bert_path, lowercase):
        vocab_file = os.path.join(bert_path, 'vocab.txt')
        self.tokenizer = FullTokenizer(vocab_file, lowercase)

    def __encode_fn(self, tokens):
        subtokens = list()
        indices = list()
        k = 1
        for token in tokens:
            token_subtokens = self.tokenizer.tokenize(token)
            subtokens.extend(token_subtokens)
            token_indices = list()
            for _ in range(len(token_subtokens)):
                token_indices.append(k)
                k += 1
            indices.append(token_indices)
        subtokens = [self.CLS] + subtokens + [self.SEP]
        ids = self.tokenizer.convert_tokens_to_ids(subtokens)
        mask = [1] * len(ids)
        type_ids = [0] * len(ids)
        return ids, mask, type_ids, indices

    def __decode_fn(self, ids):
        ids = ids[1:-1]
        subtokens = self.tokenizer.convert_ids_to_tokens(ids)
        return subtokens,

    def encode(self, sequence):
        return recursive_apply_new(self.__encode_fn, sequence, 4)

    def decode(self, sequence):
        return recursive_apply_new(self.__decode_fn, sequence, 1)


if __name__ == '__main__':
    bert_dir = '/home/sapphire/PycharmProjects/bert-biaffine-parser/data/uncased_L-2_H-128_A-2'
    bert_encoder = BertEncoder(bert_dir, True)

    s = '[ROOT] Who was Jim Henson frailty ?'.split(' ')
    encoded = bert_encoder.encode([s, s])
    print(encoded)
    # decoded = bert_encoder.decode(encoded[0])
    # print(decoded)
