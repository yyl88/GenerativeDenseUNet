import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet.gluon.loss import Loss
import mxnet.ndarray as F
from gluoncv.loss import FocalLoss
import numpy as np

class CosKernelBlock(nn.Block):
    def __init__(self, **kwargs):
        super(CosKernelBlock, self).__init__(**kwargs)

    def forward(self, x, mu):
        cos_kernel = (F.sum(F.expand_dims(x, axis=3) * mu ** 2, axis=4)) \
            / (F.sqrt((F.expand_dims(F.sum(x**2, axis=3), axis=3))) *F.sqrt(F.sum(mu ** 2, axis=1)))
        return cos_kernel

class TriWeightKernelBlock(nn.Block):
    def __init__(self, **kwargs):
        super(TriWeightKernelBlock, self).__init__(**kwargs)

    def forward(self, norm, gamma):
        return F.relu((1 - gamma * norm**2)**3)

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
        return F.expand_dims(x, axis=3) - self.mu.data()

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
    def __init__(self, units, in_units, kernel="TriWeight", norm="l2", **kwargs):
        super(CosRbfBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.cos_kernel = CosKernelBlock()
            self.rbf_block = RbfBlock(units, in_units, mu_init=mx.init.Xavier(magnitude=1), kernel=kernel, norm=norm, priors=False)

    def _sigmoid(self, x):
        return (1 / (1 + F.exp(-6 * x)))

    def diff(self, x):
        return self.rbf_block.diff(x)

    def forward(self, x):
        likelihood = self.rbf_block(x)
        cos_kernel = self.cos_kernel(x.swapaxes(1,3), self.rbf_block.mu.data())
        cos_kernel = self._sigmoid(cos_kernel).swapaxes(1,3)
        return cos_kernel * likelihood

class LocalLinearBlock(nn.Block):
    def __init__(self, units, in_units, kernel="TriWeight", norm="l2", **kwargs):
        super(LocalLinearBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.weight = self.params.get('weight', shape=(units, in_units), init=mx.init.Xavier(magnitude=2.24))
            self.bias = self.params.get('bias', shape=(units, ), init=mx.init.One())

            self.RbfBlock = RbfBlock(units, in_units, mu_init=mx.init.Xavier(magnitude=1), kernel=kernel, norm=norm, priors=False)
            #self.RbfBlock = CosRbfBlock(units, in_units)

    def forward(self, x):

        rbf = self.RbfBlock(x)
        diff = self.RbfBlock.diff(x)

        x = F.sum(diff * self.weight.data(), axis=4)
        x = x + self.bias.data()
        return x.swapaxes(1,3) * rbf

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
    r"""Computes the focal log cosh dice loss.
    Inputs:
        - **pred**: the prediction tensor, where the `batch_axis` dimension
          ranges over batch size and `axis` dimension ranges over the number
          of classes.
        - **label**: the truth tensor.  
        - **gamma**: hyper parameter for focal loss 
    Outputs:
        - **loss**: loss tensor with shape (batch_size,). 
    """
    def __init__(self, num_classes, axis=1, weight=None, batch_axis=0, **kwargs):
        super(LogCoshDiceLoss, self).__init__(weight, batch_axis, **kwargs)
        #self.softmax_cross_entropy_loss = gluon.loss.SoftmaxCELoss(axis=axis, from_logits=True, sparse_label=False, weight=0.1)
        self._num_classes = num_classes
        self._axis = axis

    def log_cosh_diceloss(self, F, pred, label, weight):
        smooth = 1
        pred_y = F.softmax(pred, axis=1)
        intersection = pred_y * label * weight
        tp = F.mean(intersection, axis=self._batch_axis, exclude=True)
        fp = F.mean(label * weight, axis=self._batch_axis, exclude=True) 
        fn = F.mean(pred_y * weight, axis=self._batch_axis, exclude=True)
        score = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
        return F.log(F.cosh(1-score))

    def hybrid_forward(self, F, pred, label):
        weight = F.Cast(label + 1, np.float32)
        one_hot = F.one_hot(label, self._num_classes).swapaxes(1, 4)
        one_hot = F.squeeze(one_hot, axis=4)

        log_cosh_diceloss = self.log_cosh_diceloss(F, pred, one_hot, weight)

        #logits = F.log_softmax(pred, axis=self._axis) * (1 - F.softmax(pred, axis=self._axis))**2
        #softmax_cross_entropy_loss = self.softmax_cross_entropy_loss(logits, one_hot)
        return 2*log_cosh_diceloss #+ softmax_cross_entropy_loss