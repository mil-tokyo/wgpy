import chainer
from chainer.functions.activation.relu import relu
from chainer.functions.array.reshape import reshape
from chainer.functions.pooling.average_pooling_2d import average_pooling_2d
from chainer import link
from chainer.links.connection.convolution_2d import Convolution2D
from chainer.links.connection.linear import Linear
from chainer.links.normalization.batch_normalization import BatchNormalization

class Block(link.Chain):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        with self.init_scope():
            self.conv1 = Convolution2D(in_channels, out_channels, 3, stride, 1, nobias=True)
            self.bn1 = BatchNormalization(out_channels)
            self.conv2 = Convolution2D(out_channels, out_channels, 3, 1, 1, nobias=True)
            self.bn2 = BatchNormalization(out_channels)
            if stride != 1 or in_channels != out_channels:
                self.shortcut = Convolution2D(in_channels, out_channels, 1, stride, nobias=True)
            else:
                self.shortcut = lambda x: x
    
    def forward(self, x):
        h = relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        h += self.shortcut(x)
        return relu(h)

def _make_blocks(n_blocks, in_channels, out_channels, stride):
    blocks = []
    for i in range(n_blocks):
        blocks.append(Block(in_channels if i == 0 else out_channels, out_channels, stride if i == 0 else 1))
    return chainer.Sequential(*blocks)

def _global_average_pooling_2d(x):
    n, channel, rows, cols = x.shape
    h = average_pooling_2d(x, (rows, cols), stride=1)
    h = reshape(h, (n, channel))
    return h

class ResNet18(link.Chain):
    def __init__(self, n_class=100):
        super().__init__()
        with self.init_scope():
            self.conv1 = Convolution2D(None, 64, 3, 1, 1, nobias=True)
            self.bn1 = BatchNormalization(64)
            self.block2 = _make_blocks(2, 64, 64, 1)
            self.block3 = _make_blocks(2, 64, 128, 2)
            self.block4 = _make_blocks(2, 128, 256, 2)
            self.block5 = _make_blocks(2, 256, 512, 2)
            self.fc1 = Linear(None, n_class)
        
    def forward(self, x):
        h = relu(self.bn1(self.conv1(x)))
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = _global_average_pooling_2d(h)
        h = self.fc1(h)

        return h
