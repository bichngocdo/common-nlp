class AbstractEncoder(object):
    def __init__(self):
        pass

    def encode(self, sequence):
        raise NotImplementedError

    def decode(self, sequence):
        raise NotImplementedError
