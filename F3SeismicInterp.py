import numpy as np
import mxnet as mx
from mxnet import gluon
from gluoncv.utils.metrics import SegmentationMetric
from mxnet.gluon.data.vision import transforms
from skimage.util import random_noise
from skimage import filters
from skimage.filters import unsharp_mask
from skimage.filters.rank import entropy, modal
from skimage.morphology import disk, cube

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from DataSet import *
from DenseUnet_2D import DenseUNet
from CustomBlocks import *
from Train import *
import time

class JoelsSegNet(gluon.Block):
    r"""
    RBFDenseUNet
    DenseUNet with an RBF output layer.
    """
    def __init__(self, **kwargs):
        super(JoelsSegNet, self).__init__(**kwargs)
        with self.name_scope():
            self.d0 = nn.HybridSequential()
            self.d0.add(nn.Conv2D(6, kernel_size=7, strides=2, use_bias=False),
                        nn.Activation('relu'),
                        nn.MaxPool2D(pool_size=3, strides=2, padding=1))

            self.DenseUNetBlock = DenseUNet(block_config=[2, 4, 8, 16], growth_rate=[4, 8, 16, 32], dropout=0)
            self.ConvTranspose = nn.Conv2DTranspose(channels=4, kernel_size=9, strides=4, use_bias=False)
            #self.RbfBlock = RbfBlock(6, 4, mu_init=mx.init.Xavier(magnitude=1))
            self.RbfBlock = CosRbfBlock(6, 4)
            #self.RbfBlock = LocalLinearBlock(6, 4)
            self.BatchNormBlock = nn.BatchNorm()

    def forward(self, x):
        x = self.Rbf(x)
        x = self.BatchNormBlock(x)
        return x

    def embeddings(self, x): 
        x0 = self.d0(x)
        x1 = self.DenseUNetBlock(x0)
        x2 = F.concat(x0, x1, dim=1)

        x3 = self.ConvTranspose(x2)
        x4 = nd.Crop(*[x3,x], center_crop=True) 
        return x4

    def Rbf(self, x):
        x = self.embeddings(x)
        x = self.RbfBlock(x)
        return x

    def calculate_probabilities(self, x):
        evidence = F.expand_dims(F.sum(x, axis=1), axis=1)
        posterior= x/evidence
        return posterior        

    def posterior(self, x):
        x = self.Rbf(x)
        x = self.calculate_probabilities(x)
        return x

    def bayes_error_rate(self, x):
        probability = self.posterior(x)
        error = 1 - F.max(probability, axis=1)
        return error

np_datasets = LoadSeismicNumpyFiles(
    ['data/train/train_seismic.npy', 'data/train/train_labels.npy'],
    ['data/test_once/test1_seismic.npy', 'data/test_once/test1_labels.npy'],
    ['data/test_once/test2_seismic.npy', 'data/test_once/test2_labels.npy']
)

def get_sample():
    global np_datasets
    return np_datasets.training['seismic'][0].T

def plot_xample(xLine):
    fig = px.imshow(xLine, color_continuous_scale='cividis')
    fig.show()

#sample_slice = get_sample()

#import pdb; pdb.set_trace()

# set the context on GPU is available otherwise CPU
ctx = [mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()]

net = JoelsSegNet()
net.initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
#net.load_parameters("DenseRBFUNet_2", ctx)
net.hybridize()

trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 0.001})

# Use Accuracy as the evaluation metric.
metric = mx.metric.Accuracy()
log_cosh_dice_loss = LogCoshDiceLoss(num_classes=6) 
#gluon.loss.SoftmaxCrossEntropyLoss(axis=1, weight=0.5)

train_data = np_datasets.create_gluon_loader(np_datasets.training, batch_transforms=False)
val_data = np_datasets.create_gluon_loader(np_datasets.validation)
test_data = np_datasets.create_gluon_loader(np_datasets.testing)

"""
train_data = np_datasets.create_gluon_loader(np_datasets.training, batch_size=32, swapaxes=True)
val_data = np_datasets.create_gluon_loader(np_datasets.validation, batch_size=32, swapaxes=True)
test_data = np_datasets.create_gluon_loader(np_datasets.testing, batch_size=32, swapaxes=True)
"""

start = time.time()
training_instance = Fit(net,ctx, trainer, metric, log_cosh_dice_loss, 64, train_data, val_data)
stop = time.time()
print("%s seconds" % (start-stop))

#net.save_parameters("DenseRBFUnet_1")

label, prediction = training_instance.val_data_iterator(inference=True)
prediction = prediction.asnumpy()

fig = px.imshow(np_datasets.training['label'][0].T, color_continuous_scale='jet')
fig.show()

fig = px.imshow(np_datasets.training['seismic'][0].T, color_continuous_scale='cividis')
fig.show()

fig = px.imshow(label[0][0].T, color_continuous_scale='jet')
fig.show()

fig = px.imshow(np.argmax(prediction[0], axis=0).T, color_continuous_scale='jet')
fig.show()

denoised_img = modal( (np.argmax(prediction[0], axis=0).T), disk(3))

fig = px.imshow(denoised_img, color_continuous_scale='jet')
fig.show()