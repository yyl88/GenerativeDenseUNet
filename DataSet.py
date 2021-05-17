import numpy as np
import mxnet as mx
import mxnet.ndarray as nd
from mxnet import gluon
from skimage import filters
from scipy.fft import fft, dct

from skimage.util import random_noise
from skimage.filters import unsharp_mask
from skimage.morphology import disk, cube
from mxnet.gluon.data.vision import transforms
from skimage.filters.rank import entropy, modal
from gluoncv.utils.metrics import SegmentationMetric

class LoadSeismicNumpyFiles():
    def __init__(self, training_path_list, validation_path_list, testing_path_list, **kwargs):

        self._dataset_keys = ['seismic', 'label']

        # dic for datasets
        self.training = {}
        self.validation = {}
        self.testing = {}

        # load each volume separately
        self.load_seismic_label_set(self.training, training_path_list, self._dataset_keys)
        self.load_seismic_label_set(self.validation, validation_path_list, self._dataset_keys)
        self.load_seismic_label_set(self.testing, testing_path_list, self._dataset_keys)

        self._planeswap = [self._XLine, self._ZSlice]

    def normalize_seismic(self, data):
        data = (data + 1)/2
        return np.float32(data)

    def load_seismic_label_set(self, dic, path, keys):
        # load .npy files in pairs (data/label)
        for i in range(0, len(path), 2):
            dic[keys[0]] = self.normalize_seismic(np.load(path[i]))
            dic[keys[1]] = np.load(path[i+1])

    def universal_preprocessing(self, data, label, plane):
        # application of filters and expansion of dims to all datasets
        assert(plane <= 2)
        if plane > 0:
            data, label = self._planeswap[plane - 1](data, label)

        #indices = np.indices((701, 255))
        #inx_1 = np.expand_dims(np.float32(indices[0])/255, axis=0)
        #inx_0 = np.expand_dims(np.float32(indices[1])/701, axis=0)
        
        #index_features = np.repeat(np.expand_dims(np.vstack((inx_0, inx_1)), 0), 401, axis=0)

        #sobel = np.expand_dims(self.sobel_aug(data), axis=1)
        
        label = np.expand_dims(label, axis=1)
        data = np.expand_dims(data, axis=1)

        #data = np.vstack((data, index_features.swapaxes(0,1))).swapaxes(0,1)

        #data = np.concatenate((sobel, data), axis=1)
        return data, label

    def _XLine(self, data, label):
        return data.swapaxes(0,1), label.swapaxes(0,1)

    def _ZSlice(self, data, label):
        return data.swapaxes(0,2), label.swapaxes(0,2)

    def rearange_labels(self, x):
        x[x == 4] = 10
        x[x == 3] = 4
        x[x == 10] = 3
        x[x == 0] = 10
        x[x == 2] = 0
        x[x == 10] = 2

    def create_gluon_loader(self, dataset_dic, batch_size=32, plane=0, shuffle=False, aug_transforms=False):
        
        dataset_iter = iter(dataset_dic.keys())
        for i in range(0, int(len(dataset_dic)/2)):
            data = dataset_dic[next(dataset_iter)]
            label = dataset_dic[next(dataset_iter)]
            print(f'Generating gluon dataset from {dataset_dic.keys()}')

        data, label = self.universal_preprocessing(data, label, plane)

        dataset = gluon.data.dataset.ArrayDataset(data, label)
        
        if aug_transforms:
            dataset = dataset.transform(self.aug_transform, lazy=True)
        
        return gluon.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def random_noise_aug(self, data):
        sigma = 0.16
        data = random_noise(data, var=sigma**2)
        return np.float32(data)

    def sharpen_aug(self, data, amount=16):
        data = unsharp_mask(data, amount=amount)
        return np.float32(data)

    def entropy_aug(self, data, radius=13):
        for img_slice in data:
            img_slice = entropy(img_slice, disk(13))
        return np.float32(data)

    def sobel_aug(self, data):
        data = filters.sobel(data)
        return np.float32(data)

    def meijering_aug(self, data):
        data = filters.meijering(data)
        return np.float32(data)

    def sato_aug(self, data):
        data = filters.sato(data)
        return np.float32(data)

    def aug_transform(self, data, label):
        data = nd.array(data)
        label = nd.array(label)

        data, label = self.joint_transform(data, label)
        return data, label

    def positional_augmentation(self, joint):
        # Random crop
        crop_height = 255
        crop_width = 255
        aug = mx.image.RandomCropAug(size=(crop_width, crop_height))
        aug_joint = aug(joint)
        # Horizontal Flip
        aug = mx.image.HorizontalFlipAug(0.1)
        aug_joint = aug(aug_joint)
        # Add more translation/scale/rotation augmentations here...
        return aug_joint

    def joint_transform(self, base, mask):
        base = base.swapaxes(0,2).swapaxes(0,1)
        mask = mask.swapaxes(0,2).swapaxes(0,1)
        ### Join
        # Concatinate on channels dim, to obtain an 6 channel image
        # (3 channels for the base image, plus 3 channels for the mask)
        base_channels = base.shape[2] # so we know where to split later on
        joint = mx.nd.concat(base, mask, dim=2)

        ### Augmentation Part 1: positional
        aug_joint = self.positional_augmentation(joint)
        
        ### Split
        aug_base = aug_joint[:, :, :base_channels].swapaxes(0,1).swapaxes(0,2)
        aug_mask = aug_joint[:, :, base_channels:].swapaxes(0,1).swapaxes(0,2)
        
        ### Augmentation Part 2: color
        aug_base = self.color_augmentations(aug_base)

        return aug_base, aug_mask

    def color_augmentations(self, data):
        # augment images using mx.image augmentators
        color_augs = [mx.image.BrightnessJitterAug(0.46),
                      mx.image.SaturationJitterAug(0.36)]

        data = data.swapaxes(0,2)
        for aug in color_augs:
            data = aug(data)

        return data.swapaxes(0,2)