import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
import mxnet.ndarray as F
from CustomBlocks import RbfBlock, CosRbfBlock, CustomKdeBlock
from mxnet.gluon.contrib.nn import HybridConcurrent, Identity

class DenseUNet(nn.Block):
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
            self.down_1 = nn.Sequential()
            self.down_1.add(
                make_dense_block(num_layers=block_config[0], bn_size=4, growth_rate=growth_rate[0], dropout=dropout),
                #make_transition(growth_rate[0]), 
            )
                        
            self.down_2 = nn.Sequential()
            self.down_2.add(
                make_transition(growth_rate[1]), 
                SelfAttentionBlock(growth_rate[1]),
                make_dense_block(num_layers=block_config[1], bn_size=4, growth_rate=growth_rate[1], dropout=dropout), 
                #make_transition(growth_rate[1]),
            )
            
            self.down_3 = nn.Sequential()
            self.down_3.add(
                make_transition(growth_rate[2]), 
                SelfAttentionBlock(growth_rate[2]),
                make_dense_block(num_layers=block_config[2], bn_size=4, growth_rate=growth_rate[2], dropout=dropout), 
                #make_transition(growth_rate[2]),
            )
            
            self.down_4 = nn.Sequential()
            self.down_4.add(
                make_transition(growth_rate[3]), 
                SelfAttentionBlock(growth_rate[3]),
                make_dense_block(num_layers=block_config[3], bn_size=4, growth_rate=growth_rate[3], dropout=dropout), 
                #make_transition(growth_rate[3]),
            )

            #self.attention_1_up = AttentionBlock(block_config[0])
            #self.attention_2_up = AttentionBlock(block_config[1])
            #self.attention_3_up = AttentionBlock(block_config[2])

            self.up_4 = UpBlock(block_config[3], growth_rate[3], dropout=dropout)
            self.up_3 = UpBlock(block_config[2], growth_rate[2], dropout=dropout)
            self.up_2 = UpBlock(block_config[1], growth_rate[1], dropout=dropout)
            #self.up_1 = UpBlock(block_config[0], growth_rate[0], dropout=dropout)

            self.feature_3 = nn.Conv2D(3, kernel_size=1, padding=0, use_bias=False)
            self.feature_2 = nn.Conv2D(2, kernel_size=1, padding=0, use_bias=False)
            self.feature_1 = nn.Conv2D(1, kernel_size=1, padding=0, use_bias=False)
            
    def forward(self, x):
        x1 = self.down_1(x)
        x2 = self.down_2(x1)
        x3 = self.down_3(x2)

        center = self.down_4(x3)
        
        #a3 = self.attention_3_up(x3, center)
        y3 = self.up_4(center,x3)

        #a2 = self.attention_2_up(x2, y3)
        y2 = self.up_3(y3,x2)

        #a1 = self.attention_1_up(x1, y2)
        y = self.up_2(y2,x1)

        #y = self.up_1(y2,x)
        
        y3 = F.UpSampling(self.feature_3(center), sample_type="nearest", scale=9)
        y3 = F.Crop(*[y3,x], center_crop=True)

        y2 = F.UpSampling(self.feature_2(y3), sample_type="nearest", scale=5)
        y2 = F.Crop(*[y2,x], center_crop=True)

        y1 = F.UpSampling(self.feature_1(y2), sample_type="nearest", scale=3)
        y1 = F.Crop(*[y1,x], center_crop=True)

        y = F.concat(y, y1, y2, y3, dim=1)
        return y

class SelfAttentionBlock(nn.Block):
    def __init__(self, n_filters, **kwargs):
        super(SelfAttentionBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.Conv_g = nn.Conv2D(3, kernel_size=3, padding=1, use_bias=True)
            
            self.swish = nn.Swish()
            self.RbfBlock = nn.Sequential()
            self.RbfBlock.add(RbfBlock(n_filters, 3, priors=False),
                              CustomKdeBlock(n_filters, n_filters))

    def forward(self, x):
        g = self.Conv_g(x)
        out = self.swish(g)
        out = self.RbfBlock(out)

        return out*x

class AttentionBlock(nn.Block):
    def __init__(self, in_num_channels, **kwargs):
        super(AttentionBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.Conv_x = nn.Conv2D(1, kernel_size=3, strides=2, use_bias=False)
            self.Conv_g = nn.Conv2D(1, kernel_size=1, use_bias=True)
            self.Conv_r = nn.Conv2D(3, kernel_size=1, padding=0, use_bias=True)
            self.BatchNorm_r = nn.BatchNorm()
            self.swish = nn.Swish()
            self.RbfBlock = nn.Sequential()
            self.RbfBlock.add(RbfBlock(3, 3, priors=False),
                              CustomKdeBlock(1, 3, w_softmax=True))

    def forward(self, x, g):
        x1 = self.Conv_x(x)
        g = self.Conv_g(g)

        if x[0][0].shape != g[0][0].shape:
            x1 = F.contrib.BilinearResize2D(x1, height=g.shape[2], width=g.shape[3])

        out = self.Conv_r(x1+g)
        out = self.BatchNorm_r(out)

        out = F.contrib.BilinearResize2D(out, height= x.shape[2], width= x.shape[3])

        out = self.swish(out)
        out = self.RbfBlock(out)

        return out*x

class UpBlock(nn.HybridBlock):
    def __init__(self, channels, growth_rate, dropout, **kwargs):
        super(UpBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.transition_up = nn.HybridSequential()
            self.transition_up.add(
                nn.BatchNorm(),
                nn.Swish(),
                nn.Conv2DTranspose(growth_rate, kernel_size=3, strides=2, use_bias=False),
            )

            self.dense = make_dense_block(channels, 4, growth_rate, dropout)

    def forward(self, x, s):
        x = self.transition_up(x)
    
        if x[0][0].shape != s[0][0].shape:
            x = F.contrib.BilinearResize2D(x, height= s.shape[2], width=s.shape[3])

        #x = F.UpSampling(x, sample_type="nearest", scale=2)
        #x = F.Crop(*[x,s], center_crop=True)
        x = F.concat(s, x, dim=1)
        x = self.dense(x)
        
        return x

def make_dense_block(num_layers, bn_size, growth_rate, dropout):
    out = nn.HybridSequential()
    for _ in range(num_layers):
        out.add(make_dense_layer(growth_rate, bn_size, dropout))
    return out

def make_dense_layer(growth_rate, bn_size, dropout):
    new_features = nn.HybridSequential()
    new_features.add(
        nn.BatchNorm(),
        nn.Swish(),
        nn.Conv2D(bn_size * growth_rate, kernel_size=1, use_bias=False),
        nn.Swish(),
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
        nn.Swish(),
        nn.Conv2D(num_output_features, kernel_size=3, padding=1),
        nn.MaxPool2D(pool_size=2, strides=2))
    return out