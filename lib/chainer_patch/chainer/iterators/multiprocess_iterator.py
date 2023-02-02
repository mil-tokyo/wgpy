from chainer.iterators.serial_iterator import SerialIterator

class MultiprocessIterator(SerialIterator):
    def __init__(self, dataset, batch_size, repeat=True, shuffle=None,
                 n_processes=None, n_prefetch=1, shared_mem=None,
                 order_sampler=None, dataset_timeout=30.0):
        super().__init__(dataset, batch_size, repeat, shuffle, order_sampler)
