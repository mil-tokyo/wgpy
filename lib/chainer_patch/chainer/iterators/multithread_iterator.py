from chainer.iterators.serial_iterator import SerialIterator

class MultithreadIterator(SerialIterator):
    def __init__(self, dataset, batch_size, repeat=True, shuffle=None,
                 n_threads=1, order_sampler=None):
        super().__init__(dataset, batch_size, repeat, shuffle, order_sampler)
