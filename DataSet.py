import numpy as np
import mxnet as mx
import mxnet.ndarray as nd
from mxnet import gluon
from gluoncv.utils.metrics import SegmentationMetric
from mxnet.gluon.data.vision import transforms
from skimage.util import random_noise
from skimage import filters
from skimage.filters import unsharp_mask
from skimage.filters.rank import entropy, modal
from skimage.morphology import disk, cube

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

    def load_seismic_label_set(self, dic, path, keys):
        # load .npy files in pairs (data/label)
        for i in range(0, len(path), 2):
            dic[keys[0]] = np.float32(np.load(path[i]))
            dic[keys[1]] = np.load(path[i+1])

    def preprocessing(self, data, label, swapaxes, timeslice):
        if swapaxes:
            data, label = self.swapaxes(data, label, timeslice)

        sobel = np.expand_dims(self.sobel_aug(data), axis=1)
        
        label = np.expand_dims(label, axis=1)
        data = np.expand_dims(data, axis=1)

        data = np.concatenate((sobel, data), axis=1)
        return data, label

    def rearange_labels(self, x):
        x[x == 4] = 10
        x[x == 3] = 4
        x[x == 10] = 3
        x[x == 0] = 10
        x[x == 2] = 0
        x[x == 10] = 2

    def create_gluon_loader(self, dataset_dic, batch_size=32, shuffle=False, swapaxes=False, timeslice=False, batch_transforms=False):
        dataset_iter = iter(dataset_dic.keys())
        for i in range(0, int(len(dataset_dic)/2)):
            data = dataset_dic[next(dataset_iter)]
            label = dataset_dic[next(dataset_iter)]
            print(f'Generating gluon dataset from {dataset_dic.keys()}')

        data, label = self.preprocessing(data, label, swapaxes, timeslice)

        dataset = gluon.data.dataset.ArrayDataset(data, label)
        
        if batch_transforms:
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
        sigma = 0.123
        data = random_noise(data, var=sigma**2)
        return np.float32(data)

    def sharpen_aug(self, data, amount=3):
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
        # augment images using mx.image augmentators
        data = nd.array(data)
        label = nd.array(label)
        augs = [mx.image.HorizontalFlipAug(0.2)]
        
        for aug in augs:
            data= aug(data)
            label = aug(label)

        return data, label