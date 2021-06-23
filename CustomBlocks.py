import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet.gluon.loss import Loss
import mxnet.ndarray as F
from gluoncv.loss import FocalLoss
import numpy as np

class RbfBlock(nn.Block):
    def __init__(self, units, in_units, mu_init=mx.init.Constant(0.25), gamma_init=mx.init.One(), kernel="TriWeight", norm="l2", priors=True, **kwargs):
        super(RbfBlock, self).__init__(**kwargs)
        self._with_priors = priors
        with self.name_scope():
            self.norm = self.initialize_norm(norm)
            self.kernel = self.initialize_kernel(kernel)

            self.mu = self.params.get('mu', shape=(units, in_units), init=mu_init)
            self.gamma = self.params.get('gamma', shape=(units, ), init=gamma_init)

            if priors:
                self.priors = self.params.get('prior', shape=(units, ), init=mx.init.Normal(0.5))

    def initialize_norm(self, norm):
        if norm == "l2":
            return L2NormBlock()
        else:
            return L1NormBlock()

    def initialize_kernel(self, kernel):
        if kernel == "TriWeight":
            return TriWeightKernelBlock()
        else:
            return GaussianKernelBlock()

    def diff(self, x):
        x = x.swapaxes(1,3)
        return F.log(F.expand_dims(x, axis=3)) - F.log(self.mu.data())

    def get_rbf_kernel_stats(self):
        return self.mu.data().asnumpy(), self.gamma.data().asnumpy()

    def forward(self, x):
        x = x.swapaxes(1,3)

        norm = self.norm(x, self.mu.data())
        likelihood = self.kernel(norm, self.gamma.data())

        if self._with_priors:
            kde = F.softmax(self.priors.data()) * likelihood
            return kde.swapaxes(1,3)
        else:
            return likelihood.swapaxes(1,3)

class CosRbfBlock(nn.Block):
    def __init__(self, units, in_units, kernel="TriWeight", norm="l2", softmax_cos=True, **kwargs):
        super(CosRbfBlock, self).__init__(**kwargs)
        mu_init = mx.init.Xavier(magnitude=1.0)
        self.softmax_cos = softmax_cos
        with self.name_scope():
            self.cos_kernel = CosKernelBlock()
            self.rbf_block = RbfBlock(units, in_units, mu_init=mu_init, kernel=kernel, norm=norm, priors=False)

    def _sigmoid(self, x):
        return (1 / (1 + F.exp(-6 * x)))

    def diff(self, x):
        return self.rbf_block.diff(x)

    def get_mu(self):
        return F.softmax(self.rbf_block.mu.data(), axis=1).asnumpy()
    
    def get_gamma(self):
        return self.rbf_block.gamma.data().asnumpy()

    def get_rbf_kernel_stats(self):
        return self.get_mu(), self.get_gamma()

    def forward(self, x):
        likelihood = self.rbf_block(x)
        cos_kernel = self.cos_kernel(x.swapaxes(1,3), self.rbf_block.mu.data())
       
        if self.softmax_cos:
            cos_kernel = F.softmax(cos_kernel, axis=3)
        
        return cos_kernel.swapaxes(1,3) * likelihood #F.softmax(likelihood*60, axis=1)

class LocalLinearBlock(nn.Block):
    def __init__(self, units, in_units, kernel="TriWeight", norm="l2", **kwargs):
        super(LocalLinearBlock, self).__init__(**kwargs)
        mu_init = mx.init.Xavier(magnitude=1.0)
        with self.name_scope():
            self.weight = self.params.get('weight', shape=(units, in_units), init=mx.init.Xavier(magnitude=2.24))
            self.bias = self.params.get('bias', shape=(units, ), init=mx.init.One())

            self.rbf_block = RbfBlock(units, in_units, mu_init=mu_init, kernel=kernel, norm=norm, priors=False)

    def get_mu(self):
        return self.rbf_block.mu.data().asnumpy()
    
    def get_gamma(self):
        return self.rbf_block.gamma.data().asnumpy()

    def get_rbf_kernel_stats(self):
        return self.get_mu(), self.get_gamma()

    def forward(self, x):

        rbf = self.rbf_block(x)
        diff = self.rbf_block.diff(x)

        x = F.sum(diff * self.weight.data(), axis=4)
        x = x + self.bias.data()
        return x.swapaxes(1,3) * rbf

class CustomKdeBlock(nn.Block):
    def __init__(self, units, in_units, w_softmax=False, **kwargs):
        super(CustomKdeBlock, self).__init__(**kwargs)
        self.n_nodes = in_units
        self.w_softmax = w_softmax

        with self.name_scope():
            self.weight = self.params.get('weight', shape=(units, in_units), init=mx.init.Xavier(magnitude=2.24))

    def forward(self, x):
        x = x.swapaxes(1,3)

        x = F.sum(F.expand_dims(x, axis=3) * self.weight.data(), axis=4)

        x = x.swapaxes(1,3) 

        if self.w_softmax:
           x = F.softmax(x*10, axis=1)

        return x

class SobelBlock(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(SobelBlock, self).__init__(**kwargs)
        self.sobel_kernel = F.array([[1, 2, 1]], ctx=mx.gpu(0)).T * F.array([[1, 0, -1]], ctx=mx.gpu(0))

    def forward(self, x, in_units):
        self.kx = F.ones(shape=(in_units, 1, 3, 3), ctx=mx.gpu(0)) * self.sobel_kernel
        self.ky = F.ones(shape=(in_units, 1, 3, 3), ctx=mx.gpu(0)) * self.sobel_kernel.T

        gx = F.Convolution(x, self.kx, kernel=(3, 3), num_filter=in_units, no_bias=True, pad=(1,1), layout='NCHW')
        gy = F.Convolution(x, self.ky, kernel=(3, 3), num_filter=in_units, no_bias=True, pad=(1,1), layout='NCHW')
        
        out = F.abs(gx) + F.abs(gy)
        return out

class CosKernelBlock(nn.Block):
    def __init__(self, **kwargs):
        super(CosKernelBlock, self).__init__(**kwargs)

    def forward(self, x, mu):
        cos_kernel = ( F.sum(F.expand_dims(x, axis=3) * mu, axis=4) ) \
            / (F.sqrt((F.expand_dims(F.sum(x**2, axis=3), axis=3))) * F.sqrt(F.sum(mu ** 2, axis=1)))
        return cos_kernel

class TriWeightKernelBlock(nn.Block):
    def __init__(self, **kwargs):
        super(TriWeightKernelBlock, self).__init__(**kwargs)

    def forward(self, norm, gamma):
        x = F.relu((1 - gamma * norm**2)**5)
        x = 9 * (F.exp(x) - 1) /5
        return x

class GaussianKernelBlock(nn.Block):
    def __init__(self, **kwargs):
        super(GaussianKernelBlock, self).__init__(**kwargs)

    def forward(self, norm, gamma):
        return F.exp(-1 * gamma * norm)
        
class L1NormBlock(nn.Block):
    def __init__(self, **kwargs):
        super(L1NormBlock, self).__init__(**kwargs)

    def forward(self, x, mu):
        x = F.expand_dims(x, axis=3) - mu
        return F.sum(F.abs(x), axis=4)

class L2NormBlock(nn.Block):
    def __init__(self, **kwargs):
        super(L2NormBlock, self).__init__(**kwargs)

    def forward(self, x, mu):
        x = F.expand_dims(x, axis=3) - mu
        return F.sqrt(F.sum(x ** 2, axis=4))

class TriWeightActivation(nn.HybridBlock):
    r"""
    Parameters
    ----------
    beta : float
        Tw(x) = (1 - x^2)^3

    Inputs:
        - **data**: input tensor with arbitrary shape.

    Outputs:
        - **out**: output tensor with the same shape as `data`.
    """

    def __init__(self, beta=1.0, **kwargs):
        super(TriWeightActivation, self).__init__(**kwargs)
        self._beta = beta

    def hybrid_forward(self, F, x):
        return x * F.relu((x**2)**3)

class ContrastiveLoss(Loss):
    def __init__(self, margin=6., weight=None, batch_axis=0, **kwargs):
        super(ContrastiveLoss, self).__init__(weight, batch_axis, **kwargs)
        self.margin = margin

    def hybrid_forward(self, F, image1, image2, label):
        distances = image1 - image2
        distances_squared = F.sum(F.square(distances), 1, keepdims=True)
        euclidean_distances = F.sqrt(distances_squared + 0.0001)
        d = F.clip(self.margin - euclidean_distances, 0, self.margin)
        loss = (1 - label) * distances_squared + label * F.square(d)
        loss = 0.5*loss
        return loss

class LogCoshDiceLoss(Loss):
    r"""Computes the log cosh dice loss.
    Inputs:
        - **pred**: the prediction tensor, where the `batch_axis` dimension
          ranges over batch size and `axis` dimension ranges over the number
          of classes.
        - **label**: the truth tensor.  
    Outputs:
        - **loss**: loss tensor with shape (batch_size,). 
    """
    def __init__(self, num_classes, tune_model=False, axis=1, weight=None, batch_axis=0, **kwargs):
        super(LogCoshDiceLoss, self).__init__(weight, batch_axis, **kwargs)
        self.softmax_cross_entropy_loss = gluon.loss.SoftmaxCELoss(axis=axis, from_logits=True, sparse_label=False, weight=0.1)
        self.mse = gluon.loss.L2Loss()
        self.mae = gluon.loss.L1Loss()
        self._num_classes = num_classes
        self._axis = axis
        self._tune_model = tune_model
        if tune_model:
            self.sobel = SobelBlock()

    def log_cosh_diceloss(self, F, pred, label, weight):
        smooth = 1
        pred_y = F.softmax(pred, axis=1)
        intersection = pred_y * label * weight
        tp = F.mean(intersection, axis=self._batch_axis, exclude=True)
        fp = F.mean(label * weight, axis=self._batch_axis, exclude=True) 
        fn = F.mean(pred_y * weight, axis=self._batch_axis, exclude=True)
        score = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
        return F.log(F.cosh(1-score))

    def bayes_pred(self, F, x):
        evidence = F.expand_dims(F.sum(x, axis=1), axis=1)
        posterior= x/evidence
        return posterior  

    def hybrid_forward(self, F, pred, label, syn, signal):
        weight = F.Cast(label + 1, np.float32)/self._num_classes
        one_hot = F.one_hot(label, self._num_classes).swapaxes(1, 4)
        one_hot = F.squeeze(one_hot, axis=4)

        log_cosh_diceloss = self.log_cosh_diceloss(F, pred, one_hot, weight)
        mse_loss = self.mse(syn, signal)

        #logits = F.log_softmax(pred, axis=self._axis) * (1 - F.softmax(pred, axis=self._axis))
        #softmax_cross_entropy_loss = self.softmax_cross_entropy_loss(logits, one_hot)

        loss = 100*log_cosh_diceloss + mse_loss
        
        if self._tune_model:
            #pred_labels = F.expand_dims(F.argmax(F.softmax(pred, axis=1), axis=1), axis=1)
            pred_labels = F.expand_dims(F.argmax(self.bayes_pred(F, pred), axis=1), axis=1)
            TvNorm_pred = F.stop_gradient(F.mean(self.sobel(pred_labels, pred.shape[0]), axis=self._batch_axis, exclude=True))
            TvNorm_label = F.stop_gradient(F.mean(self.sobel(label, pred.shape[0]), axis=self._batch_axis, exclude=True))
            TvNorm = self.mae(TvNorm_pred, TvNorm_label)
            loss = loss + TvNorm

        return loss