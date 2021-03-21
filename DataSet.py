import numpy as np
import mxnet as mx
import mxnet.ndarray as nd
from mxnet import gluon
from skimage import filters
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

        #sobel = np.expand_dims(self.sobel_aug(data), axis=1)
        
        label = np.expand_dims(label, axis=1)
        data = np.expand_dims(data, axis=1)

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

    def swapaxes(self, data, label, timeslice):
        # swap the axes of the volumes to train on XLine
        # if timeslice is true, swap axes of volume to train on time slices
        if timeslice:
            return data.swapaxes(0,2), label.swapaxes(0,2)
        else:
            return data.swapaxes(0,1), label.swapaxes(0,1)

    def random_noise_aug(self, data):
        sigma = 0.223
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
        """
        augs_list = [self.sharpen_aug, 
                     self.random_noise_aug]

        for aug in augs_list:
            data = aug(data)
        """
        data = nd.array(data)
        label = nd.array(label)

        #data, label = self.spatial_augmentations(data, label)
        data = self.color_augmentations(data)
        return data, label

    def spatial_augmentations(self, data, label):
        # augment images using mx.image augmentators
        spatial_augs = [mx.image.HorizontalFlipAug(0.36)]
         
        for aug in spatial_augs:
            data= aug(data)
            label = aug(label)

        return data, label

    def color_augmentations(self, data):
        # augment images using mx.image augmentators
        color_augs = [#mx.image.ContrastJitterAug(0.9),
                      mx.image.BrightnessJitterAug(0.36),
                      mx.image.SaturationJitterAug(0.36),
                      #mx.image.HueJitterAug(0.3),
                      mx.image.ColorJitterAug(0.36, 0.36, 0.36)]

        data = data.swapaxes(0,2)
        for aug in color_augs:
            data = aug(data)

        return data.swapaxes(0,2)