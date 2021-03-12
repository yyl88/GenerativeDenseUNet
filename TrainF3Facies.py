import numpy as np
import mxnet as mx
from mxnet import gluon
from gluoncv.utils.metrics import SegmentationMetric
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
from Fit import *
import time

#--------------------------------------------------------------------------------------------------

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

            self.DenseUNetBlock = DenseUNet(block_config=[2, 4, 8, 16], 
                                            growth_rate=[4, 8, 16, 32], 
                                            dropout=0)

            self.ConvTranspose = nn.Conv2DTranspose(channels=4, 
                                                    kernel_size=9, 
                                                    strides=4, 
                                                    use_bias=False)
            
            #self.RbfBlock = RbfBlock(6, 4, mu_init=mx.init.Xavier(magnitude=1))
            self.RbfBlock = CosRbfBlock(6, 4)
            #self.RbfBlock = LocalLinearBlock(6, 4)
            
            self.BatchNormBlock_0 = nn.BatchNorm()
            self.BatchNormBlock = nn.BatchNorm()

    def forward(self, x):
        x = self.rbf_output(x)
        x = self.BatchNormBlock(x)
        return x

    def embeddings(self, x): 
        x0 = self.d0(x)
        x1 = self.DenseUNetBlock(x0)
        x2 = F.concat(x0, x1, dim=1)

        x3 = self.ConvTranspose(x2)
        x4 = nd.Crop(*[x3,x], center_crop=True) 
        return x4

    def rbf_output(self, x):
        x = self.embeddings(x)
        x = self.BatchNormBlock_0(x)
        x = self.RbfBlock(x)
        return x

    def calculate_probabilities(self, x):
        evidence = F.expand_dims(F.sum(x, axis=1), axis=1)
        posterior= x/evidence
        return posterior        

    def posterior(self, x):
        x = self.rbf_output(x)
        x = self.calculate_probabilities(x)
        return x

    def bayes_error_rate(self, x):
        probability = self.posterior(x)
        error = 1 - F.max(probability, axis=1)
        return error

#--------------------------------------------------------------------------------------------------

np_datasets = LoadSeismicNumpyFiles(
    ['data/train/train_seismic.npy', 'data/train/train_labels.npy'],
    ['data/test_once/test1_seismic.npy', 'data/test_once/test1_labels.npy'],
    ['data/test_once/test2_seismic.npy', 'data/test_once/test2_labels.npy']
)

#--------------------------------------------------------------------------------------------------

def get_sample():
    global np_datasets
    return np_datasets.training['seismic'][0].T

def plot_xample(xLine):
    fig = px.imshow(xLine, color_continuous_scale='cividis')
    fig.show()

sample_slice = get_sample()
plot_xample(sample_slice)
#import pdb; pdb.set_trace()

#--------------------------------------------------------------------------------------------------

net = JoelsSegNet()


# set the context on GPU is available otherwise CPU
ctx = [mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()]

#net.load_parameters("DenseRBFUNet_1", ctx)

net.initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
net.hybridize()

trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 0.001})

# Use Accuracy as the evaluation metric.
metric = mx.metric.Accuracy()
log_cosh_dice_loss = LogCoshDiceLoss(num_classes=6) 

#--------------------------------------------------------------------------------------------------

train_data = np_datasets.create_gluon_loader(np_datasets.training, batch_transforms=True, shuffle=True)
val_data = np_datasets.create_gluon_loader(np_datasets.validation)
test_data = np_datasets.create_gluon_loader(np_datasets.testing)

"""
train_data = np_datasets.create_gluon_loader(np_datasets.training, batch_size=32, swapaxes=True)
val_data = np_datasets.create_gluon_loader(np_datasets.validation, batch_size=32, swapaxes=True)
test_data = np_datasets.create_gluon_loader(np_datasets.testing, batch_size=32, swapaxes=True)
"""

#--------------------------------------------------------------------------------------------------

start = time.time()
training_instance = Fit(net,ctx, trainer, metric, log_cosh_dice_loss, 60, train_data, val_data)
stop = time.time()
print("%s seconds" % (start-stop))

#--------------------------------------------------------------------------------------------------

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

#--------------------------------------------------------------------------------------------------

mu, gamma = net.RbfBlock.get_rbf_kernel_stats()

fig = px.scatter_3d(x=mu[:,0], y=mu[:,1], z=mu[:,2], opacity=0.8, color=gamma, color_continuous_scale='jet')
fig.update_layout(template="plotly_dark", title="Rbf nodes")
fig.show()

#--------------------------------------------------------------------------------------------------

code, labels = training_instance.latent_space()

fig = go.Figure()
fig.add_trace( 
    go.Scatter3d(
        x=code[0,0,:,:].flatten(),
        y=code[0,1,:,:].flatten(), 
        z=code[0,2,:,:].flatten(),
        name='class',
        mode='markers',
        marker=dict(
            size=0.9, 
            color=labels[0].flatten(),
            colorscale='jet'
        ) 
    )
)

fig.add_trace(
    go.Scatter3d(
        x=mu[:,0],
        y=mu[:,1], 
        z=mu[:,2],
        name='centers',
        mode='markers',
        marker=dict(
            color=gamma,
            opacity=1,
            colorscale='jet'
        ) 
    )
)

fig.update_layout(template="plotly_dark", title="Embeddings and Rbf nodes")

fig.show()

#--------------------------------------------------------------------------------------------------

err = training_instance.error()

plot_xample(err[0].T)

import pdb; pdb.set_trace()

net.save_parameters("DenseRBFUnet_1")