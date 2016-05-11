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

        #######################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        #######################################################################
        channel_num = input_dim[0]
        H = input_dim[1]
        W = input_dim[2]
        HH = filter_size
        WW = filter_size
        conv_weights = np.random.normal(
            scale=weight_scale, size=(num_filters, channel_num, HH, WW))
        conv_biases = np.zeros(num_filters)
        stride = 1
        pad = (filter_size - 1) / 2
        Hp = 1 + (H + 2 * pad - HH) / stride
        Wp = 1 + (W + 2 * pad - WW) / stride
        conv_maxpool_size = (num_filters * Hp/2 * Wp/2)
        hid_aff_weights = np.random.normal(scale=weight_scale, size= (conv_maxpool_size, hidden_dim))
        hid_aff_bias = np.zeros(hidden_dim)
        out_aff_weights = np.random.normal(scale=weight_scale, size=(hidden_dim, num_classes))
        out_aff_bias = np.zeros(num_classes)
        self.params['W1'] = conv_weights
        self.params['b1'] = conv_biases
        self.params['W2'] = hid_aff_weights
        self.params['b2'] = hid_aff_bias
        self.params['W3'] = out_aff_weights
        self.params['b3'] = out_aff_bias
        #######################################################################
        #                             END OF YOUR CODE                             #
        #######################################################################

        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        conv - relu - 2x2 max pool - affine - relu - affine - softmax

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        # print W1, b1
        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}
        conv_out, conv_cache = conv_forward_naive(X, W1, b1, conv_param)
        # conv_shape = conv_out.shape
        relu1_out, relu1_cache = relu_forward(conv_out)
        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        pool_out, pool_cache = max_pool_forward_naive(relu1_out, pool_param)
        aff1_out, aff1_cache = affine_forward(pool_out, W2, b2)
        relu2_out, relu2_cache = relu_forward(aff1_out)
        aff2_out, aff2_cache = affine_forward(relu2_out, W3, b3)

        scores = aff2_out
        #######################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        #######################################################################
        #######################################################################
        #                             END OF YOUR CODE                             #
        #######################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        #######################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #######################################################################
        loss, dscores = softmax_loss(scores, y)
        dr2o, dw3, db3 = affine_backward(dscores, aff2_cache)
        da1o = relu_backward(dr2o, relu2_cache)
        dpoolo, dw2, db2 = affine_backward(da1o, aff1_cache)
        dr1o = max_pool_backward_naive(dpoolo, pool_cache)
        da1o = relu_backward(dr1o, relu1_cache)
        dco, dw1, db1 = conv_backward_naive(da1o, conv_cache)
        loss+=self.reg*(np.sum(W1*W1) + np.sum(W2*W2) + np.sum(W3*W3)) / 2
        grads['W1'] = dw1 + self.reg * W1
        grads['b1'] = db1
        grads['W2'] = dw2 + self.reg * W2
        grads['b2'] = db2
        grads['W3'] = dw3 + self.reg * W3
        grads['b3'] = db3

        #######################################################################
        #                             END OF YOUR CODE                             #
        #######################################################################

        return loss, grads


pass
