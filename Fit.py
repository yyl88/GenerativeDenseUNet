import mxnet as mx
import numpy as np
from mxnet import gluon
from mxnet.gluon import nn
import mxnet.ndarray as nd
from skimage.util import random_noise
from mxnet import autograd as ag

class Fit():
    r"""
    Inputs:
        - **net**: gluon block/neural network model.
        - **ctx**: Context, either mx.gpu or mx.cpu.  
        - **trainer**: gluon.Trainer object. 
        - **metric**: mx.metric object. 
        - **training_loss**: gluon.loss object. 
        - **epoch**: int. 
        - **train_data**: gluon.DataLoader object. 
        - **val_data**: gluon.DataLoader object. 
    """
    def __init__(self, net, ctx, trainer, metric, training_loss, epoch, train_data, val_data, **kwargs):
        self.net = net
        self.ctx = ctx
        self.trainer = trainer
        self.metric = metric
        self.training_loss = training_loss
        self.epoch = epoch
        self.train_data = train_data
        self.val_data = val_data

        self.train_acc = []
        self.val_acc = []

        self._fit()

    def _fit(self):
        lr_reduced = False
        lr = 0.001
        thresh_hold = 0.88
        
        for i in range(self.epoch):
            t_acc = self.train_data_iterator()
            v_acc = self.val_data_iterator()
            b_acc = self.val_data_iterator(bayes=True)
    
            print('epoch %d: train_acc = %f: val_acc_softmax = %f: val_acc_bayes = %f' % (i, t_acc*100, v_acc*100, b_acc*100))

            if b_acc > thresh_hold and not lr_reduced:
                thresh_hold += 0.06
                lr_reduced = True

                lr = 0.0006
                self.trainer = gluon.Trainer(self.net.collect_params(), 'adam', {'learning_rate': lr})                
                print("learning rate reduced")

    def latent_space(self):
         # Loop over the test data iterator.
        for (data, label) in self.val_data:
            data_batch = gluon.utils.split_and_load(data, ctx_list=self.ctx, batch_axis=0)

            label_batch = gluon.utils.split_and_load(label, ctx_list=self.ctx, batch_axis=0)

            embeddings = self.net.embeddings(data_batch[0])

        return nd.softmax(embeddings, axis=1).asnumpy(), label_batch[0].asnumpy()

    def error(self):
         # Loop over the test data iterator.
        for (data, label) in self.val_data:
            data_batch = gluon.utils.split_and_load(data, ctx_list=self.ctx, batch_axis=0)

            label_batch = gluon.utils.split_and_load(label, ctx_list=self.ctx, batch_axis=0)

            model_error = self.net.bayes_error_rate(data_batch[0])

        return model_error.asnumpy()

    def val_data_iterator(self, inference=False, bayes=False):
        r"""
        Inputs:
            - **inference**: bool.
                Option to use the function for inference.
                If true, will return softmax ndarray.
            - **bayes**: bool.
                If true, Bayes theorem will be use to calculate posterior probability volume.
        Outputs:
            - **labels**: numpy nd array
            - **probability volume**: numpy nd.array
        """
        # Loop over the test data iterator.
        for (data, label) in self.val_data:
            data_batch = gluon.utils.split_and_load(data, ctx_list=self.ctx, batch_axis=0)

            label_batch = gluon.utils.split_and_load(label, ctx_list=self.ctx, batch_axis=0)

            if not bayes:
                output = self.net(data_batch[0])
            else:
                output = self.net.posterior(data_batch[0])

            # Updates internal evaluation
            self.metric.update(label_batch[0], output)

        name, v_acc = self.metric.get()
        self.val_acc.append(v_acc)

        # Reset evaluation result to initial state.
        self.metric.reset()
        
        if inference:
            if bayes:
                return label.asnumpy(), output
            else:
                return label.asnumpy(), nd.softmax(output, axis=1)
        else:
            return v_acc


    def train_data_iterator(self):
        # Loop over the train data iterator.
        for data, label in self.train_data:
            # Splits train data into multiple slices along batch_axis
            # and copy each slice into a context.
            data_batch = gluon.utils.split_and_load(data, ctx_list=self.ctx, batch_axis=0)
            # Splits train labels into multiple slices along batch_axis
            # and copy each slice into a context.
            label_batch = gluon.utils.split_and_load(label, ctx_list=self.ctx, batch_axis=0)

            # Inside training scope
            with ag.record():
                output = self.net(data_batch[0])
                loss = self.training_loss(output, label_batch[0])
                loss.backward()

            assert(output.shape[-2:] == label.shape[-2:])

            # Updates internal evaluation
            self.metric.update(label_batch[0], output)

            # Make one step of parameter update. Trainer needs to know the
            # batch size of data to normalize the gradient by 1/batch_size.
            self.trainer.step(data.shape[0])
        
        # Gets the evaluation result.
        name, t_acc = self.metric.get()
        self.train_acc.append(t_acc)
        
        # Reset evaluation result to initial state.
        self.metric.reset()
        return t_acc