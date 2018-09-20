from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        conv_weights = np.random.normal(0, weight_scale, (num_filters, input_dim[0], filter_size, filter_size))
        conv_biases = np.zeros((num_filters))

        w2_size = input_dim[1]//2
        W2 = np.random.normal(0, weight_scale, (num_filters*w2_size*w2_size,hidden_dim))
        b2 = np.zeros(hidden_dim)

        W3 = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
        b3 = np.zeros(num_classes)

        self.params['W1'] = conv_weights
        self.params['b1'] = conv_biases
        self.params['W2'] = W2
        self.params['b2'] = b2
        self.params['W3'] = W3
        self.params['b3'] = b3

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None

        i1, conv_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        #print(i1.shape)
        h1, hcache = affine_relu_forward(i1, W2, b2)
        scores, scache = affine_forward(h1, W3, b3)


        if y is None:
            return scores

        loss, grads = 0, {}
        loss, dscores = softmax_loss(scores, y)
        loss += self.reg * (0.5*np.sum(self.params['W1']**2) + 0.5*np.sum(self.params['W2']**2) + 0.5*np.sum(self.params['W3']**2))

        dh1, dW3, db3 = affine_backward(dscores, scache)

        di1, dW2, db2 = affine_relu_backward(dh1, hcache)
        dx, dW1, db1 = conv_relu_pool_backward(di1, conv_cache)

        grads['W1'] = dW1
        grads['b1'] = db1
        grads['W2'] = dW2
        grads['b2'] = db2
        grads['W3'] = dW3
        grads['b3'] = db3


        return loss, grads
