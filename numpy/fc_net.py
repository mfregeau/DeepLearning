from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg


        self.params['W1'] = np.random.normal(0, weight_scale, (input_dim, hidden_dim))
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
        self.params['b2'] = np.zeros(num_classes)



    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None

        h1, cache1 = affine_relu_forward(X, self.params['W1'], self.params['b1'])
        scores, cache2 = affine_forward(h1, self.params['W2'], self.params['b2'])


        if y is None:
            return scores

        loss, grads = 0, {}

        loss, dscores = softmax_loss(scores, y)
        loss += self.reg * (0.5*np.sum(self.params['W1']**2) + 0.5*np.sum(self.params['W2']**2))

        dh1, dW2, db2 = affine_backward(dscores, cache2)
        dx, dW1, db1 = affine_relu_backward(dh1, cache1)

        dW1 += self.reg*self.params['W1']
        dW2 += self.reg*self.params['W2']

        grads['W1'] = dW1 
        grads['W2'] = dW2
        grads['b1'] = db1
        grads['b2'] = db2



        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}


        w_strings = ["W{}".format(i) for i in range(1,self.num_layers+1)]
        b_strings = ["b{}".format(i) for i in range(1,self.num_layers+1)]


        self.params[w_strings[0]] = np.random.normal(0, weight_scale, (input_dim, hidden_dims[0]))
        self.params[b_strings[0]] = np.zeros(hidden_dims[0])
        
        if(self.use_batchnorm):
            gamma_s = ["gamma{}".format(i) for i in range(1, self.num_layers)]
            beta_s = ["beta{}".format(i) for i in range(1, self.num_layers)]
            self.params[gamma_s[0]] = np.ones(hidden_dims[0])
            self.params[beta_s[0]] = np.zeros(hidden_dims[0])

        for i in range(1,self.num_layers-1):
            self.params[w_strings[i]] = np.random.normal(0, weight_scale, (hidden_dims[i-1], hidden_dims[i]))
            self.params[b_strings[i]] = np.zeros(hidden_dims[i])
            if(self.use_batchnorm):
                self.params[gamma_s[i]] = np.ones(hidden_dims[i])
                self.params[beta_s[i]] = np.zeros(hidden_dims[i])

        self.params[w_strings[-1]] = np.random.normal(0, weight_scale, (hidden_dims[-1], num_classes))
        self.params[b_strings[-1]] = np.zeros(num_classes)


        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed


        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None

        caches = []
        outs = [X]
        

        for i in range(self.num_layers-1):
            if self.use_batchnorm:
                gamma = self.params['gamma{}'.format(i+1)]
                beta = self.params['beta{}'.format(i+1)]
                #print(gamma.shape, beta.shape, outs[i].shape)
                if self.use_dropout:
                   new_out, new_cache = affine_bnorm_relu_forward(outs[i], self.params['W{}'.format(i+1)], self.params['b{}'.format(i+1)], gamma, beta, self.bn_params[i], self.dropout_param) 
                else:
                    new_out, new_cache = affine_bnorm_relu_forward(outs[i], self.params['W{}'.format(i+1)], self.params['b{}'.format(i+1)], gamma, beta, self.bn_params[i])
            else:
                if self.use_dropout:
                    new_out, new_cache = affine_relu_forward(outs[i], self.params['W{}'.format(i+1)], self.params['b{}'.format(i+1)], self.dropout_param)
                else:
                    new_out, new_cache = affine_relu_forward(outs[i], self.params['W{}'.format(i+1)], self.params['b{}'.format(i+1)])
            
            outs += [new_out]
            caches += [new_cache]

            if self.use_dropout:
                pass
            
            

        

        scores, last_cache = affine_forward(outs[-1], self.params['W{}'.format(self.num_layers)], self.params['b{}'.format(self.num_layers)])

        if mode == 'test':
            return scores

        loss, grads = 0.0, {}

        loss, dscores = softmax_loss(scores, y)
        

        new_grad, grads['W{}'.format(self.num_layers)], grads['b{}'.format(self.num_layers)] = affine_backward(dscores, last_cache)
        
        for i in range(self.num_layers-1):
            if(self.use_batchnorm):
                new_grad, grads['W{}'.format(self.num_layers-i-1)], grads['b{}'.format(self.num_layers-i-1)], grads['gamma{}'.format(self.num_layers-i-1)], grads['beta{}'.format(self.num_layers-i-1)] = affine_bnorm_relu_backward(new_grad, caches[self.num_layers-i-2])
            else:    
                new_grad, grads['W{}'.format(self.num_layers-i-1)], grads['b{}'.format(self.num_layers-i-1)] = affine_relu_backward(new_grad, caches[self.num_layers-i-2])

        for i in range(self.num_layers):
            grads['W{}'.format(i+1)] += self.reg*self.params["W{}".format(i+1)]
            loss += 0.5*self.reg*np.sum(self.params["W{}".format(i+1)]**2)


        return loss, grads
