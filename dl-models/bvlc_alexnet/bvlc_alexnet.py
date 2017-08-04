
import chainer
import chainer.functions as F
import chainer.links as L


class net(chainer.Chain):

    def __init__(self):
        super(net, self).__init__(
            conv1=L.Convolution2D(3,  96, 11, stride=4),
            conv2=L.Convolution2D(96, 256,  5, pad=2),
            conv3=L.Convolution2D(256, 384,  3, pad=1),
            conv4=L.Convolution2D(384, 384,  3, pad=1),
            conv5=L.Convolution2D(384, 256,  3, pad=1),
            fc6=L.Linear(256 * 6 * 6, 4096),
            fc7=L.Linear(4096, 4096),
            fc8=L.Linear(4096, 1000),
        )
        self.train = False
        self.insize = 224
    def __call__(self, x, t=None):
        h = self.conv1(x)
        h.name = 'conv1'
        h = F.relu(h)
        h.name = 'relu1'
        h = F.local_response_normalization(h)
        h.name = 'norm1'
        h = F.max_pooling_2d(h, 3, stride=2)
        h.name = 'pool1'
        
        h = self.conv2(h)
        h.name = 'conv2'
        h = F.relu(h)
        h.name = 'relu2'
        h = F.local_response_normalization(h)
        h.name = 'norm2'
        h = F.max_pooling_2d(h, 3, stride=2)
        h.name = 'pool2'
        
        h = self.conv3(h)
        h.name = 'conv3'
        h = F.relu(h)
        h.name = 'relu3'
        
        h = self.conv4(h)
        h.name = 'conv4'
        h = F.relu(h)
        h.name = 'relu4'
        
        h = self.conv5(h)
        h.name = 'conv5'
        h = F.relu(h)
        h.name = 'relu5'
        h = F.max_pooling_2d(h, 3, stride=2)
        h.name = 'pool5'

        h = self.fc6(h)
        h.name = 'fc6'
        h = F.relu(h)
        h.name = 'relu6'
        h = F.dropout(h)
        h.name = 'drop6'

        h = self.fc7(h)
        h.name = 'fc7'
        h = F.relu(h)
        h.name = 'relu7'
        h = F.dropout(h)
        h.name = 'drop7'

        h = self.fc8(h)
        h.name = 'fc8'
        h = F.softmax(h)
        h.name = 'prob'
        
        self.score = h

        if t is None:
            assert not self.train
            return h

        self.loss = F.softmax_cross_entropy(self.score, t)
        self.accuracy = F.accuracy(self.score, t)
        chainer.report({'loss': self.loss, 'accuracy': self.accuracy})

        return self.loss
