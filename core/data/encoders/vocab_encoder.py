from core.data.encoders.abstract_encoder import AbstractEncoder
from core.data.encoders.helpers import recursive_apply


class Vocab(dict):
    def __init__(self, default_value=None):
        super(Vocab, self).__init__()
        self.default_value = default_value

    def __getitem__(self, item):
        return self.get(item, self.default_value)


class VocabEncoder(AbstractEncoder):
    def __init__(self, vocab, oov_token='[UNK]'):
        super(VocabEncoder, self).__init__()

        self.str2id = None
        self.id2str = None
        self.oov_token = oov_token
        self.oov_id = 0

        self.__init(vocab)

    def __init(self, vocab):
        self.str2id = Vocab()
        self.id2str = list()

        id = 0
        for item in vocab:
            self.str2id[item] = id
            self.id2str.append(item)
            id += 1
        if self.oov_token not in vocab:
            self.str2id[self.oov_token] = id
            self.id2str.append(self.oov_token)
            self.oov_id = id
        else:
            self.oov_id = self.str2id[self.oov_token]

        self.str2id.default_value = self.oov_id

    def __getitem__(self, token):
        return self.str2id[token]

    def __len__(self):
        return len(self.str2id)

    def lookup(self, id):
        return self.id2str[id]

    def __encode_fn(self, token):
        return self.str2id[token]

    def __decode_fn(self, id):
        return self.id2str[id]

    def encode(self, sequence):
        return recursive_apply(self.__encode_fn, sequence)

    def decode(self, sequence):
        return recursive_apply(self.__decode_fn, sequence)
