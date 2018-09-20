from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    shp = x.shape
    X = x.reshape((shp[0], np.prod(shp[1:])))
    out = np.dot(X, w) + b
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None

    shp = x.shape
    X = x.reshape((shp[0], np.prod(shp[1:])))
    dw = np.dot(X.T, dout)
    db = np.sum(dout, axis=0, keepdims=True)[0]
    dx = np.dot(dout, w.T).reshape(shp)
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    out = np.maximum(0, x)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    dout[x<=0]=0
    dx = dout
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        sample_mean = np.mean(x)
        sample_var = np.var(x)
        x_hat = (x-sample_mean)/(np.sqrt(sample_var) + eps)
        out = gamma*x_hat + beta

        running_mean = momentum*running_mean + (1-momentum) * sample_mean
        running_var = momentum*running_var + (1-momentum) * sample_var

        cache = (x, gamma, beta, sample_mean, sample_var, eps, N*D)

    elif mode == 'test':
        out = (gamma/np.sqrt(running_var + eps))*x + (beta - ((gamma*running_mean)/(np.sqrt(running_var+eps))))
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    x, gamma, beta, sample_mean, sample_var, epsilon, size = cache
    dl_dy = dout
    dl_dxh = dl_dy * gamma

    dl_dvar = np.sum(dl_dxh * (x-sample_mean) * (-0.5) * (sample_var+epsilon)**(-3/2))
    dl_dmean = (np.sum(dl_dxh * (-1.0/np.sqrt(sample_var+epsilon)))) + (dl_dvar) * ((np.sum(-2*(x-sample_mean)))/size)

    dgamma = np.sum(dout * ((x-sample_mean)/(np.sqrt(sample_var+epsilon))), axis=0)
    dbeta = np.sum(dout, axis=0)

    dx = dl_dxh * (1.0/np.sqrt(sample_var+epsilon)) + dl_dvar * (2*(x-sample_mean))/size + dl_dmean/size
    return dx, dgamma, dbeta



def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = np.random.choice([0, 1], size=x.shape, p=[p, 1-p])
    out = None

    if mode == 'train':
        out = x*mask
    elif mode == 'test':

        out=x


    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':

        dx = dout*mask

    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    pad = conv_param['pad']
    stride = conv_param['stride']
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    H_ = int(1+ (H+ 2*pad-HH)/stride)
    W_ = int(1 + (W + 2*pad-WW)/stride)

    padded = np.pad(x, ((0, 0),(0, 0),(pad, pad),(pad,pad)), 'constant', constant_values = 0)

    out = np.zeros((N, F, H_, W_))
    for i in range(N):
        V = np.zeros((F, H_, W_))
        for f in range(F):
            for h in range(H_):
                for w_ in range(W_):
                    V[f, h, w_] += np.sum(padded[i, :, stride*h:(stride*h)+HH, stride*w_:(stride*w_)+WW] * w[f]) + b[f]
        out[i] += V

    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None

    x, w, b, conv_param = cache

    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    pad = conv_param['pad']
    stride = conv_param['stride']
    H_ = int(1+ (H+ 2*pad-HH)/stride)
    W_ = int(1 + (W + 2*pad-WW)/stride)
    #print(H_, W_)
    padded = np.pad(x, ((0, 0),(0, 0),(pad, pad),(pad,pad)), 'constant', constant_values = 0)
    dout_padded = np.pad(dout, ((0, 0),(0, 0),(pad, pad),(pad,pad)), 'constant', constant_values = 0)
    
    dw = np.zeros((F, C, HH, WW))
    db = np.zeros(F)
    dx = np.zeros((N, C, H, W))
    dx_padded = np.pad(dx, ((0, 0),(0, 0),(pad, pad),(pad,pad)), 'constant', constant_values = 0)
    for i in range(N):
        for f in range(F):
            for h in range(H_):
                for w_ in range(W_):

                    dw[f] += dout[i, f, h, w_] * padded[i, :,  stride*h:(stride*h)+HH, stride*w_:stride*w_+WW]
                    db[f] += dout[i, f, h, w_]
                    dx_padded[i , :, stride*h:(stride*h)+HH, stride*w_:(stride*w_)+WW] += w[f]*dout[i, f, h, w_]


    dx = dx_padded[:, :, pad:-pad, pad:-pad]
    print(dx.shape)

    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    
    N, C, H, W = x.shape
    hh = pool_param['pool_height']
    ww = pool_param['pool_width']
    stride = pool_param['stride']
    W_ = int(1+(W-ww)/stride)
    H_ = int(1+(H-hh)/stride)
    out = np.zeros((N, C, H_, W_))

    for n in range(N):
        for h in range(H_):
            for w in range(W_):
                out[n, :, h, w] = np.amax(np.amax(x[n, :, h*stride:(h*stride)+hh, w*stride:(w*stride)+ww], axis=1), axis=1)

    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    x, pool_param = cache
    N, C, H, W = x.shape
    hh = pool_param['pool_height']
    ww = pool_param['pool_width']
    stride = pool_param['stride']
    W_ = int(1+(W-ww)/stride)
    H_ = int(1+(H-hh)/stride)
    dx = np.zeros((N, C, H, W))


    for n in range(N):
        for h in range(H_):
            for w in range(W_):

                idx = np.argmax(x[n, :, h*stride:(h*stride)+hh, w*stride:(w*stride)+ww], axis=1)
                temp = np.amax(x[n, :, h*stride:(h*stride)+hh, w*stride:(w*stride)+ww], axis=1)
                idy = np.argmax(temp, axis=1)
                idx = idx[list(range(len(idx))),idy]
                idxy = np.concatenate((idx, idy)).reshape(2, C)
                dx[n, list(range(C)), idxy[0]+(h*stride), idxy[1]+(w*stride)] = dout[n, :, h, w]


    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (C,) giving running mean of features
      - running_var Array of shape (C,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    N, C, H, W = x.shape
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)
    running_mean = bn_param.get('running_mean', np.zeros(C, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(C, dtype=x.dtype))
    sample_mean = np.zeros(C)
    sample_var = np.zeros(C)
    out = np.zeros((N, C, H, W))
    for channel in range(C):
        cur_channel = x[:, channel, :, :]
        channel_mean = np.mean(cur_channel)
        channel_var = np.var(cur_channel)
        sample_mean[channel] = channel_mean
        sample_var[channel] = channel_var
        x_hat = (cur_channel - channel_mean)/np.sqrt(channel_var + eps)
        channel_out = gamma[channel]*x_hat + beta[channel]
        out[:, channel, :, :] += channel_out
        running_mean[channel] = momentum*running_mean[channel] + (1-momentum) * sample_mean[channel]
        running_var[channel] = momentum*running_var[channel] + (1-momentum) * sample_var[channel]
    
    cache = (x, gamma, beta, sample_mean, sample_var, eps, N*H*W)

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None
    N, C, H, W = dout.shape
    dx = np.zeros((N, C, H, W))
    dgamma = np.zeros(C)
    dbeta = np.zeros(C)

    
    x, gamma, beta, sample_mean, sample_var, eps, size = cache
    for channel in range(C):
        cur_cache = (x[:, channel, :, :], gamma[channel], beta[channel], sample_mean[channel], sample_var[channel], eps, size)
        dcx, dcgamma, dcbeta = batchnorm_backward(dout[:, channel, :, :], cur_cache)
        dx[:, channel, :, :] += dcx
        dgamma[channel] += np.sum(dcgamma)
        dbeta[channel] += np.sum(dcbeta)

    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
