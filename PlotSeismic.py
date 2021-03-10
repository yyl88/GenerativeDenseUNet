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

from DenseUnet_2D import DenseUNet
from CustomBlocks import *
from Train import *
import pdb
import time

class LoadSeismicNumpyFiles():
    def __init__(self, training_path_list, validation_path_list, testing_path_list, **kwargs):

        self.training_dataset_keys = ['seismic', 'label']
        self.validation_dataset_keys = ['seismic', 'label']
        self.testing_dataset_keys = ['seismic', 'label']

        self.training = {}
        self.validation = {}
        self.testing = {}

        self.load_seismic_label_set(self.training, training_path_list, self.training_dataset_keys)
        self.load_seismic_label_set(self.validation, validation_path_list, self.validation_dataset_keys)
        self.load_seismic_label_set(self.testing, testing_path_list, self.testing_dataset_keys)

        #self.rearange_labels(self.training['label'])
        #self.rearange_labels(self.validation['label'])
        #self.augment_training_dataset()

    def load_seismic_label_set(self, dic, path, keys):
        # load .npy files in pairs (data/label)
        for i in range(0, len(path), 2):
            dic[keys[0]] = np.float32(np.load(path[i]))
            dic[keys[1]] = np.load(path[i+1])

    def augment_training_dataset(self):
        #self.training['seismic'] = self.random_noise_aug(self.training['seismic'])
        self.training['seismic'] = self.sobel_aug(self.training['seismic'])
        self.validation['seismic'] = self.sobel_aug(self.validation['seismic'])
        #self.testing['seismic'] = self.sato_aug(self.testing['seismic'])

    def rearange_labels(self, x):
        x[x == 4] = 10
        x[x == 3] = 4
        x[x == 10] = 3
        x[x == 0] = 10
        x[x == 2] = 0
        x[x == 10] = 2
        #x[x == 1] = 10
        #x[x == 2] = 1
        #x[x == 10] = 2

    def create_gluon_loader(self, dataset_dic, batch_size=32, shuffle=False, swapaxes=False, timeslice=False, batch_transforms=False):
        dataset_iter = iter(dataset_dic.keys())
        for i in range(0, int(len(dataset_dic)/2)):
            data = dataset_dic[next(dataset_iter)]
            label = dataset_dic[next(dataset_iter)]
            print(f'Generating gluon dataset from {dataset_dic.keys()}')

        if swapaxes:
            data, label = self.swapaxes(data, label, timeslice)

        data = np.expand_dims(data, axis=1)
        label = np.expand_dims(label, axis=1)

        dataset = gluon.data.dataset.ArrayDataset(data, label)
        
        if batch_transforms:
            dataset = dataset.transform(self.aug_transform, lazy=True)
        
        return gluon.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def swapaxes(self, data, label, timeslice):
        if timeslice:
            return data.swapaxes(0,2), label.swapaxes(0,2)
        else:
            return data.swapaxes(0,1), label.swapaxes(0,1)

    def random_noise_aug(self, data):
        sigma = 0.123
        data = random_noise(data, var=sigma**2)
        return np.float32(data)

    def sharpen_aug(self, data, radius=5, amount=2):
        data = unsharp_mask(data, radius=5, amount=amount)
        return np.float32(data)

    def entropy_aug(self, data, radius=13):
        for img_slice in data:
            img_slice = entropy(img_slice, disk(13))
        return np.float32(data)

    def sobel_aug(self, data):
        data = filters.sobel(data)
        return np.float32(data)

    def sato_aug(self, data):
        data = filters.sato(data)
        return np.float32(data)

    def aug_transform(self, data, label):
        # augment images using mx.image augmentators
        data = nd.array(data)
        label = nd.array(label)
        augs = [mx.image.HorizontalFlipAug(0.2)]
        
        for aug in augs:
            data= aug(data)
            label = aug(label)

        return data, label

class JoelSegNet(gluon.Block):
    r"""
    RBFDenseUNet
    DenseUNet with an RBF output layer.
    """
    def __init__(self, **kwargs):
        super(JoelSegNet, self).__init__(**kwargs)
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

net = JoelSegNet()
net.initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
#net.load_parameters("DenseRBFUNet_2", ctx)
net.hybridize()

trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 0.001})

# Use Accuracy as the evaluation metric.
metric = mx.metric.Accuracy()
focal_log_cosh_dice_loss = FocalLogCoshDiceLoss(num_classes=6) 
#gluon.loss.SoftmaxCrossEntropyLoss(axis=1, weight=0.5)

train_data = np_datasets.create_gluon_loader(np_datasets.training, batch_transforms=True)
val_data = np_datasets.create_gluon_loader(np_datasets.validation)
test_data = np_datasets.create_gluon_loader(np_datasets.testing)

"""
train_data = np_datasets.create_gluon_loader(np_datasets.training, batch_size=32, swapaxes=True)
val_data = np_datasets.create_gluon_loader(np_datasets.validation, batch_size=32, swapaxes=True)
test_data = np_datasets.create_gluon_loader(np_datasets.testing, batch_size=32, swapaxes=True)
"""

start = time.time()
training_instance = Fit(net,ctx, trainer, metric, focal_log_cosh_dice_loss, 64, train_data, val_data)
stop = time.time()
print("%s seconds" % (start-stop))

net.save_parameters("DenseRBFUnet_1")

label, prediction = training_instance.val_data_iterator(inference=True)
prediction = prediction.asnumpy()
#prediction = np.argmax(prediction[0], axis=0)

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