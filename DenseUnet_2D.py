import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
import mxnet.ndarray as nd
from mxnet.gluon.contrib.nn import HybridConcurrent, Identity

class DenseUNet(nn.HybridBlock):
    r"""
    DenseUNet
    A UNet architecture with Dense blocks and upsampled downstream feature maps.
    Paper: "Densely Connected Convolutional Networks"
    Parameters:
        - **block_config**: List of int.
            List of integers for number of layers in each dense block
        - **growth_rate**: List of int.
            Number of filters to add to each layer ('k' in the paper')  
        - **dropout**: Rate of dropout after each dense layer 
    """
    def __init__(self, block_config=[6,6,6,6], growth_rate=[6,6,6,6], dropout=0.0, **kwargs):
        
        super(DenseUNet, self).__init__(**kwargs)
        with self.name_scope():
            
            self.d1 = nn.HybridSequential()
            self.d1.add(
                make_dense_block(num_layers=block_config[0], bn_size=1, growth_rate=growth_rate[0], dropout=dropout),
                make_transition(growth_rate[0]), 
            )
                        
            self.d2 = nn.HybridSequential()
            self.d2.add(
                make_dense_block(num_layers=block_config[1], bn_size=1, growth_rate=growth_rate[1], dropout=dropout), 
                make_transition(growth_rate[1]),
            )
            
            self.d3 = nn.HybridSequential()
            self.d3.add(
                make_dense_block(num_layers=block_config[2], bn_size=1, growth_rate=growth_rate[2], dropout=dropout), 
                make_transition(growth_rate[2]),
            )
            
            self.d4 = nn.HybridSequential()
            self.d4.add(
                make_dense_block(num_layers=block_config[3], bn_size=1, growth_rate=growth_rate[3], dropout=dropout), 
                make_transition(growth_rate[3]),
            )

            #self.f4 = nn.Conv2D(1, kernel_size=3, padding=1, activation="relu")
            self.f3 = nn.Conv2D(1, kernel_size=1, padding=0, use_bias=False)
            self.f2 = nn.Conv2D(1, kernel_size=1, padding=0, use_bias=False)
            self.f1 = nn.Conv2D(1, kernel_size=1, padding=0, use_bias=False)
            
            self.u4 = up_block(block_config[3], growth_rate[3], dropout=dropout)
            self.u3 = up_block(block_config[2], growth_rate[2], dropout=dropout)
            self.u2 = up_block(block_config[1], growth_rate[1], dropout=dropout)
            self.u1 = up_block(block_config[0], growth_rate[0], dropout=dropout)
            
    def hybrid_forward(self, F, x):
        x1 = self.d1(x)
        x2 = self.d2(x1)
        x3 = self.d3(x2)
        x4 = self.d4(x3)
        
        y4 = self.u4(x4,x3)
        y3 = self.u3(y4,x2)
        y2 = self.u2(y3,x1)
        y = self.u1(y2,x)
        
        #y4 = F.UpSampling(self.f4(x4), sample_type="nearest", scale=21)
        #y4 = F.Crop(*[y4,x], center_crop=True)

        y3 = F.UpSampling(self.f3(y4), sample_type="nearest", scale=9)
        y3 = F.Crop(*[y3,x], center_crop=True)

        y2 = F.UpSampling(self.f2(y3), sample_type="nearest", scale=5)
        y2 = F.Crop(*[y2,x], center_crop=True)

        y1 = F.UpSampling(self.f1(y2), sample_type="nearest", scale=3)
        y1 = F.Crop(*[y1,x], center_crop=True)

        y = F.concat(y, y1, y2, y3, dim=1)
        return y

class up_block(nn.HybridBlock):
    def __init__(self, channels, growth_rate, dropout, **kwargs):
        super(up_block, self).__init__(**kwargs)
        with self.name_scope():
            self.transition_up = nn.HybridSequential()
            self.transition_up.add(
                nn.BatchNorm(),
                nn.Activation("relu"),
                nn.Conv2DTranspose(growth_rate, kernel_size=3, strides=1, use_bias=False),
            )

            #self.drop = nn.Dropout(0.2)

            self.dense = make_dense_block(channels, 1, growth_rate, dropout)

    def hybrid_forward(self, F, x, s):
        x = self.transition_up(x)
        x = F.UpSampling(x, sample_type="nearest", scale=2)

        x = F.Crop(*[x,s], center_crop=True)
        x = F.concat(s, x, dim=1)

        x = self.dense(x)
        return x

# Helper functions to build DenseNet backbone layers
def make_dense_block(num_layers, bn_size, growth_rate, dropout):
    out = nn.HybridSequential()
    for _ in range(num_layers):
        out.add(make_dense_layer(growth_rate, bn_size, dropout))
    return out

def make_dense_layer(growth_rate, bn_size, dropout):
    new_features = nn.HybridSequential()
    new_features.add(
        nn.BatchNorm(),
        nn.Activation('relu'),
        nn.Conv2D(bn_size * growth_rate, kernel_size=1, use_bias=False),
        nn.Activation('relu'),
        nn.Conv2D(growth_rate, kernel_size=3, padding=1))
    
    if dropout:
        new_features.add(nn.Dropout(dropout))

    out = HybridConcurrent(axis=1)
    out.add(Identity())
    out.add(new_features)

    return out

def make_transition(num_output_features):
    out = nn.HybridSequential()
    out.add(
        nn.BatchNorm(),
        nn.Activation('relu'),
        nn.Conv2D(num_output_features, kernel_size=3, padding=1),
        nn.MaxPool2D(pool_size=2, strides=2))
    return out