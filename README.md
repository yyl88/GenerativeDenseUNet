# RBFDenseUNet
A UNet architecture with Dense blocks, and an RBF output layer.

An RBF based artificial neural networks used for the purpose of semantic image segmentation. The backbone of this model is a DenseUNe. The head of the convolutional neural network is an RBF layer. Inference can be done using Bayes Theorem, or Softmax. 

The dataset used for training and validation can be found in this repo: https://github.com/yalaudah/facies_classification_benchmark

Papers: "Densely Connected Convolutional Networks",
"U-Net: Convolutional Networks for Biomedical Image Segmentation", 
"A Probabilistic RBF Network for Classification", 
"A survey of loss functions for semantic segmentation".

https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.77.1134&rep=rep1&type=pdf

https://arxiv.org/pdf/1608.06993.pdf

https://arxiv.org/pdf/1505.04597.pdf

https://arxiv.org/pdf/2006.14822.pdf
