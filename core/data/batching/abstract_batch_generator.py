class AbstractBatchGenerator(object):
    def __init__(self):
        pass

    def get_batch_indexes(self, batch_size, shuffle=False):
        raise NotImplementedError
